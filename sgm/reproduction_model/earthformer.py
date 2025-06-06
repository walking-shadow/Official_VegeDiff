import torch
import torch.nn as nn
from torch.jit import Final
from torch.nn import functional as F
from typing import Optional, Union

import argparse
import ast
import numpy as np
import timm

class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class PVT_embed(nn.Module):

    def __init__(self,in_channels, out_channels ,pretrained = True ,frozen = False):
        super().__init__()

        self.pvt = timm.create_model("pvt_v2_b0.in1k", pretrained = pretrained, features_only=True, in_chans = in_channels)
        if frozen:
            timm.utils.freeze(self.pvt)
        self.pvt_project = nn.Conv2d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):

        B, T, C, H, W = x.shape
        
        x_feats = self.pvt(x.reshape(B*T, C, H, W))

        x_feats = [F.interpolate(feat, size = x_feats[0].shape[-2:]) for feat in x_feats]

        x = self.pvt_project(torch.cat(x_feats, dim = 1))

        _, C, H, W = x.shape

        # Patchify

        x_patches = x.reshape(B, T, C, H, W).permute(0,3,4,1,2).reshape(B * H * W, T, C)

        return x_patches

class ContextFormer(nn.Module):

    def __init__(self, param1=None):
        super().__init__()


        self.context_length = 10
        self.target_length = 20
        self.patch_size = 8
        self.n_image = 9
        self.n_weather = 24
        self.n_hidden = 256
        self.n_out = 4
        self.n_heads = 8
        self.depth = 3
        self.mlp_ratio = 4.0
        self.mtm = True
        self.leave_n_first = 3
        self.p_mtm = 0.3
        

        self.embed_images = Mlp(in_features=self.n_image * self.patch_size * self.patch_size, hidden_features=self.n_hidden, out_features=self.n_hidden)

        self.embed_weather = Mlp(in_features=self.n_weather, hidden_features=self.n_hidden, out_features=self.n_hidden)

        self.mask_token = nn.Parameter(torch.zeros(self.n_hidden))

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.n_hidden,
                    self.n_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.depth)
            ]
        )

        self.head = Mlp(in_features=self.n_hidden, hidden_features=self.n_hidden, out_features=self.n_out * self.patch_size * self.patch_size)
        

    def forward(self, data):
        
        # Input handling
        context_length = self.context_length


        # Get the dimensions of the input data. Shape: batch size, temporal size, number of channels, height, width
        preds_length = self.target_length

        c_l = self.context_length

        hr_dynamic_inputs = data["imgs"][:, :context_length, ...]  # B, T, 4, H, W

        B, T, C, H, W = hr_dynamic_inputs.shape

        if T == c_l: # If Only given Context images, add zeros (later overwritten by token mask)
            hr_dynamic_inputs = torch.cat((hr_dynamic_inputs, torch.zeros(B, preds_length, C, H, W, device = hr_dynamic_inputs.device)), dim = 1)
            B, T, C, H, W = hr_dynamic_inputs.shape

        static_inputs = data["highres_condition_image"]  # B, 5, H, W
        static_inputs = static_inputs.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B, T, 5, H, W

        weather = data["meso_condition_image"]  # B, T, 24
        _, t_m, c_m = weather.shape




        images = torch.cat([hr_dynamic_inputs, static_inputs], dim = 2)  # B, T, 9, H, W
        B, T, C, H, W = images.shape

        # Patchify

        image_patches = images.reshape(B, T, C, H//self.patch_size,self.patch_size,W//self.patch_size,self.patch_size).permute(0,3,5,1,2,4,6).reshape(B * H//self.patch_size * W//self.patch_size, T, C * self.patch_size * self.patch_size)
        B_patch, N_patch, C_patch = image_patches.shape
        image_patches_embed = self.embed_images(image_patches)

        
        weather_patches = weather.reshape(B, 1, t_m, c_m).repeat(1, H//self.patch_size * W//self.patch_size, 1, 1).reshape(B_patch, t_m, c_m)

        # Embed Patches
        weather_patches_embed = self.embed_weather(weather_patches)
        
        # Add Token Mask, 过去部分值为0，未来部分值为1

        token_mask = torch.ones(B_patch, N_patch,device=weather_patches.device).type_as(weather_patches).reshape(B_patch, N_patch, 1).repeat(1, 1, self.n_hidden)
        token_mask[:,:c_l] = 0

        # 未来部分的每个像素点，即通道部分都是相同的可训练参数，相当于相同的初始化参数
        image_patches_embed[token_mask.bool()] = (self.mask_token).reshape(1, 1, self.n_hidden).repeat(B_patch, N_patch, 1)[token_mask.bool()]


        # Add Image and Weather Embeddings
        patches_embed = image_patches_embed + weather_patches_embed


        # Add Positional Embedding
        pos_embed = get_sinusoid_encoding_table(N_patch, self.n_hidden).to(patches_embed.device).unsqueeze(0).repeat(B_patch, 1, 1)

        x = patches_embed + pos_embed

        # Then feed all into Transformer Encoder
        for blk in self.blocks:
            x = blk(x)

        # Decode image patches
        x_out = self.head(x)

        # Mask Non-masked inputs
        x_out[~token_mask.bool()[:,:,:self.n_out*self.patch_size*self.patch_size]] = -1

        # unpatchify images
        images_out = x_out.reshape(B, H//self.patch_size, W//self.patch_size, N_patch, self.n_out, self.patch_size, self.patch_size).permute(0, 3, 4, 1, 5, 2, 6).reshape(B, N_patch, self.n_out, H, W)

        images_out = images_out[:,-preds_length:]  # B,20,4,H,W

        return images_out
