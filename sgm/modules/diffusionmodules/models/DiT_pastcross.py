import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from sgm.modules.diffusionmodules.models.video_module.motion_module import PositionalEncoding, get_motion_module_layer
from einops import rearrange
# from sgm.modules.diffusionmodules.models.video_module.motion_module import 
from torch.jit import Final
from timm.layers import use_fused_attn
import torch.nn.functional as F


class Conv_Past_Future(nn.Module):
    # 一开始先把已知的图像和预测的图像进行卷积，只保留预测图像个数的图像数据，然后去预测噪声

    def __init__(
        self,
        in_channel,
        out_channel,
    ):
        # 因为只需要把过去图像的简单时序信息注入到未来加噪的图像里面，因此不宜把middle channel设置得太大，和把网络设置得太深
        super().__init__()
        import math
        middle_channel = 2 ** (int(math.log2(in_channel))+1)  # 设置中间通道数为比输入通道数大的最近的一个2的幂次方数，保证维数相当
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=middle_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=middle_channel),
            nn.ReLU(inplace=True)
        )
        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=middle_channel[0], out_channels=middle_channel[0], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=middle_channel[0]),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=middle_channel[1], out_channels=middle_channel[1], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(num_features=middle_channel[1]),
        #     nn.ReLU(inplace=True)
        # )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=middle_channel, out_channels=out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv_in(x)
        # x = torch.cat([self.conv_1(x), x], dim=1)
        # x = torch.cat([self.conv_2(x), x], dim=1)
        out = self.conv_out(x)
        
        return out

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            noise_image_num=None, 
            image_time_length=None,
            temporal_module_kwargs=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.noise_image_num = noise_image_num
        self.image_time_length = image_time_length
        self.temporal_transformer_block = get_motion_module_layer(
            in_channels=dim,
            num_transformer_block=1,
            noise_image_num=noise_image_num,
            motion_module_kwargs=temporal_module_kwargs
        )

    def forward(self, without_noise_input: torch.Tensor, with_noise_input: torch.Tensor) -> torch.Tensor:
        # 输入的with_noise_input形状为b*t2, h*w, c, without_noise_input形状为b*t1,h*w,c

        # 只对未来图像部分进行多头自注意力操作
        x = with_noise_input

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
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

        # 单点时序注意力,把时序注意力变成交叉注意力，即Q=未来预测图像，KV=历史图像
        with_noise_input = x  # b*t2,h*w,c
        h_noise = int(with_noise_input.shape[1]**0.5)  # 要求输入图像长宽相等
        without_noise_input = rearrange(without_noise_input, "(b t) (h w) c -> b c t h w", \
                                        t=(self.image_time_length - self.noise_image_num), h=h_noise).contiguous()
        with_noise_input = rearrange(with_noise_input, "(b t) (h w) c -> b c t h w", t=self.noise_image_num, h=h_noise).contiguous()
        output = torch.cat([without_noise_input, with_noise_input], dim=2)  # b c t h w

        # 时序交叉注意力的输出只有未来图像部分，不需要把过去部分拼接回去
        output = self.temporal_transformer_block(output, temb=None, encoder_hidden_states=None)  # b c t h w
        # output = rearrange(output, "b c t h w -> (b t) (h w) c").contiguous()
        # output = output + x
        output = output + with_noise_input
        output = rearrange(output, "b c t h w -> (b t) (h w) c").contiguous()

        return output


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         embeddings = self.embedding_table(labels)
#         return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, noise_image_num=None, image_time_length=None, temporal_module_kwargs=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, noise_image_num=noise_image_num, image_time_length=image_time_length,
                              temporal_module_kwargs=temporal_module_kwargs, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, without_noise_input, with_noise_input, c):
        # without_noise_input形状为b*t1,h*w,c,  with_noise_input形状为b*t2,h*w,c, c的形状为b*t2,c， 输出形状为b*t,h*w,c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        with_noise_input = with_noise_input + gate_msa.unsqueeze(1) * self.attn(without_noise_input, modulate(self.norm1(with_noise_input), shift_msa, scale_msa))
        with_noise_input = with_noise_input + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(with_noise_input), shift_mlp, scale_mlp))
        return without_noise_input, with_noise_input


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_checkpoint=False,
        noise_image_num=20,
        image_time_length=30,  # 时间序列长度
        temporal_module_kwargs=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.noise_image_num = noise_image_num
        self.image_time_length = image_time_length

        # # Conv in 一开始先把已知的图像和预测的图像进行卷积，只保留预测图像个数的图像数据，然后去预测噪声
        # self.conv_past_future = Conv_Past_Future(in_channel=image_time_length*in_channels, 
        #                                          out_channel=noise_image_num*in_channels,
        #                                          )

        # self.context_pos_encoder = PositionalEncoding(
        #     context_dim,
        #     dropout=0., 
        #     max_len=image_time_length,
        # )
        self.input_pos_encoder = PositionalEncoding(
            in_channels,
            dropout=0., 
            max_len=image_time_length,
        )

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                     noise_image_num=noise_image_num, image_time_length=image_time_length,
                     temporal_module_kwargs=temporal_module_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, without_noise_input, with_noise_input, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        原始的代码只是单图像生成，要生成序列图像，需要把N变成N*T,并在特定的部分加上时间编码，更新后的输入形状如下
        """
        # without_noise_input,没加噪的历史图像，形状为b,t,c,h,w
        # with_noise_input = 加了噪声的预测图像序列, 形状为b,t,c,h,w
        # t = timesteps = c_noise = sigma = sigma = sqrt((1-A)/A),A为噪声累计影响变量，为一维向量，形状为b
        # context = 图像condition，有4个不同大小的condition，对应Unet下采样3次的4中预测变量图像，形状为b*h*w,l,d, 暂时不加
        # y = 图像大小信息，形状为b,d，目前为None

        # # 首先给输入图像进行卷积，然后加入时间编码信息
        # # 把过去图像和加噪预测图像卷积在一起
        x = torch.cat([without_noise_input, with_noise_input], dim=1)
        b_in,t_in,c_in,h_in,w_in = x.shape
        # x = rearrange(x, "b t c h w -> b (t c) h w").contiguous()  # b,t*c,h,w
        # x = self.conv_past_future(x)  # b,t2*c,h,w
        # x = rearrange(x, "b (t c) h w -> b t c h w", t=self.noise_image_num).contiguous()  # b,t2,c,h,w


        # # 给condition图像加上时间编码，不需要加空间位置编码，因为condition适合遥感图像逐像素作用的
        # for i in range(len(context)):
        #     context[i] = self.context_pos_encoder(context[i])
        
        # 给输入的图像添加时间编码, 相邻形状的合并可以使用view，否则需要交换维度的时候需要使用transpose，统一一点可以用rearrange函数
        x = self.input_pos_encoder(rearrange(x, "b t c h w -> (b h w) t c").contiguous())  # b*h*w, t2, c
        x = rearrange(x, "(b h w) t c -> (b t) c h w", b=b_in, h=h_in, w=w_in).contiguous()  # b*t2,c,h,w



        # x = rearrange(x, "b (t c) h w -> (b t) c h w", t=self.noise_image_num).contiguous()  # b*t2,c,h,w

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / (patch_size ** 2), N = b*t
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        # c = t + y                                # (N, D)
        c = t                                # (N, D) y=None
        # c = c.repeat(self.image_time_length, 1) # b*t, c
        c = c.repeat(self.noise_image_num, 1) # b*t2, c
        print(f'shape of condition:{c.shape}')

        x = rearrange(x, "(b t) n c -> b t n c", t=t_in).contiguous()
        without_noise_input, with_noise_input = x[:,:-self.noise_image_num, ...], x[:,-self.noise_image_num:, ...]  # b,t,n,c
        without_noise_input = rearrange(without_noise_input, "b t n c -> (b t) n c").contiguous()  # b*t1,n,c
        with_noise_input = rearrange(with_noise_input, "b t n c -> (b t) n c").contiguous()  # b*t2,n,c



        for block in self.blocks:
            if self.use_checkpoint:
                without_noise_input, with_noise_input = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), without_noise_input, with_noise_input, c)       # (N, T, D)
            else:
                without_noise_input, with_noise_input = block(without_noise_input, with_noise_input, c)       # (N, T, D)
        with_noise_input = self.final_layer(with_noise_input, c)                # (N, T, patch_size ** 2 * out_channels)
        with_noise_input = self.unpatchify(with_noise_input)                   # (N, out_channels, H, W)
        # x = rearrange(x, "(b t) c h w -> b t c h w", t=self.noise_image_num).contiguous()  # b,t,c,h,w

        with_noise_input = rearrange(with_noise_input, "(b t) c h w -> b t c h w", t=self.noise_image_num).contiguous()  # b,t,c,h,w
        # x = x[:,-self.noise_image_num:,...]  # b,t2,c,h,w
        return with_noise_input

    # def forward_with_cfg(self, x, t, y, cfg_scale):
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, y)
    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

# def DiT_XL_2(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def DiT_XL_4(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

# def DiT_XL_8(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# def DiT_L_2(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# def DiT_L_4(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# def DiT_L_8(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# def DiT_B_2(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# def DiT_B_4(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# def DiT_B_8(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# def DiT_S_2(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# def DiT_S_4(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# def DiT_S_8(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# DiT_models = {
#     'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
#     'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
#     'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
#     'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
# }