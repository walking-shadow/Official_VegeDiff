import math
from abc import abstractmethod
from functools import partial
from typing import Iterable

import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .videoattention import SpatialTransformer, Conv_Past_Future
from ....modules.diffusionmodules.util import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from ....util import default, exists
from .video_module.motion_module import get_motion_module, get_motion_module_layer, PositionalEncoding


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self,
        x,
        emb,
        context=None,
        skip_time_mix=False,
        time_context=None,
        num_video_frames=None,
        time_context_cat=None,
        use_crossframe_attention_in_spatial_layers=False,
    ):
        # print(f'timestep input x:{x[0][0][0][0]}')
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                if isinstance(layer, Upsample):
                    B,T,C,H,W = x.shape
                    x = x.view(B*T, C, H, W)
                    x = layer(x)
                    _, C, H, W = x.shape
                    x = x.view(B, T, C, H, W)
                else:
                    x = layer(x)
        # print(f'timestep output x:{x[0][0][0][0]}')
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        if self.dims == 3:
            t_factor = 1 if not self.third_up else 2
            x = F.interpolate(
                x,
                (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if dtype == torch.bfloat16:
            x = x.to(dtype)
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    "Learned 2x upsampling without padding"

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(
            self.channels, self.out_channels, kernel_size=ks, stride=2
        )

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            print(f"Building a Downsample layer with {dims} dims.")
            print(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            if dims == 3:
                print(f"  --> Downsampling third axis (time): {third_down}")
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        # print(f'downsample input x:{x[0][0][0][0]}')

        assert x.shape[1] == self.channels
        x = self.op(x)
        # print(f'downsample input x:{x[0][0][0][0]}')
        return x

# video version
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        temporal_block_layers=0,
        temporal_module_kwargs=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        
        
        self.temporal_block_layers = temporal_block_layers
        if self.temporal_block_layers != 0: # open temporal_block_layer
            self.temporal_transformer_block = get_motion_module_layer(
                in_channels=self.out_channels,
                num_transformer_block=temporal_block_layers,
                motion_module_kwargs=temporal_module_kwargs
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # print(f'resnet input x:{x[0][0][0][0]}')
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w").contiguous()
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            
            # Before
            # h = rearrange(h.view(B, T, h.shape[-3], h.shape[-2], h.shape[-1]), "b t c h w -> b c (t h) w").contiguous()
            # h = h + emb_out
            # h = rearrange(h, "b c (t h) w -> (b t) c h w ", t=T).contiguous()
            # h = self.out_layers(h)
            
            h = rearrange(h, "(b t) c h w -> b t c h w", t=T).contiguous()
            h = h + emb_out.view(emb_out.shape[0], -1, emb_out.shape[1], emb_out.shape[2], emb_out.shape[3])
            h = rearrange(h, "b t c h w -> (b t) c h w ", t=T).contiguous()
            h = self.out_layers(h)
        output = self.skip_connection(x) + h
        _, C, H, W = output.shape
        output = output.view(B, T, C, H, W).contiguous()
        
        if self.temporal_block_layers != 0: # open temporal_block_layer
            output = rearrange(output, "b t c h w -> b c t h w").contiguous()
            output = self.temporal_transformer_block(output, temb=None, encoder_hidden_states=None)
            output = rearrange(output, "b c t h w -> b t c h w").contiguous()
        # print(f'resnet output x:{x[0][0][0][0]}')
        return output


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, **kwargs):
        # TODO add crossframe attention and use mixed checkpoint
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        # print(f'attentionblock input x:{x[0][0][0][0]}')
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        # print(f'attentionblock output x:{h[0][0][0][0]}')

        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class VideoUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        noise_image_num,
        image_time_length,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="softmax",
        adm_in_channels=None,
        use_fairscale_checkpoint=False,
        offload_to_cpu=False,
        transformer_depth_middle=None,
        temporal_module_kwargs=None,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.noise_image_num = noise_image_num
        self.image_time_length = image_time_length
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        if use_fp16:
            print("WARNING: use_fp16 was dropped and has no effect anymore.")
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        assert use_fairscale_checkpoint != use_checkpoint or not (
            use_checkpoint or use_fairscale_checkpoint
        )

        self.use_fairscale_checkpoint = False
        checkpoint_wrapper_fn = (
            partial(checkpoint_wrapper, offload_to_cpu=offload_to_cpu)
            if self.use_fairscale_checkpoint
            else lambda x: x
        )

        time_embed_dim = model_channels * 4
        self.time_embed = checkpoint_wrapper_fn(
            nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = checkpoint_wrapper_fn(
                    nn.Sequential(
                        Timestep(model_channels),
                        nn.Sequential(
                            linear(model_channels, time_embed_dim),
                            nn.SiLU(),
                            linear(time_embed_dim, time_embed_dim),
                        ),
                    )
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        # Conv in 一开始先把已知的图像和预测的图像进行卷积，只保留预测图像个数的图像数据，然后去预测噪声
        self.conv_past_future = Conv_Past_Future(in_channel=image_time_length*in_channels, 
                                                 out_channel=noise_image_num*in_channels,
                                                 middle_channel=[128,256,512])

        self.context_pos_encoder = PositionalEncoding(
            context_dim,
            dropout=0., 
            max_len=image_time_length,
        )
        self.input_pos_encoder = PositionalEncoding(
            in_channels,
            dropout=0., 
            max_len=noise_image_num,
        )

        # Down
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]): # normal block
                if ds in attention_resolutions:
                    layers = [
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=mult * model_channels,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                temporal_module_kwargs=temporal_module_kwargs
                            )
                        )
                    ]
                else:
                    layers = [
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=mult * model_channels,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                temporal_module_kwargs=temporal_module_kwargs,
                                temporal_block_layers=1
                            )
                        )
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            checkpoint_wrapper_fn(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                            if not use_spatial_transformer
                            else checkpoint_wrapper_fn(
                                SpatialTransformer(
                                    ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth[level],
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_transformer,
                                    attn_type=spatial_transformer_attn_type,
                                    use_checkpoint=use_checkpoint,
                                    temporal_module_kwargs=temporal_module_kwargs,
                                    temporal_block_layers=transformer_depth[level]
                                )
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1: # downsample
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        
        self.middle_block = TimestepEmbedSequential(    
            checkpoint_wrapper_fn(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    temporal_module_kwargs=temporal_module_kwargs
                )
            ),
            checkpoint_wrapper_fn(
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
            )
            if not use_spatial_transformer
            else checkpoint_wrapper_fn(
                SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,  
                    dim_head,
                    depth=transformer_depth_middle,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    attn_type=spatial_transformer_attn_type,
                    use_checkpoint=use_checkpoint,
                    temporal_module_kwargs=temporal_module_kwargs,
                    temporal_block_layers=transformer_depth_middle
                )
            ),
            checkpoint_wrapper_fn(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    temporal_module_kwargs=temporal_module_kwargs
                )
            ),
        )
        self._feature_size += ch

        # UP
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                if ds in attention_resolutions and (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                    layers = [
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch + ich,
                                time_embed_dim,
                                dropout,
                                out_channels=model_channels * mult,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                temporal_module_kwargs=temporal_module_kwargs,
                                temporal_block_layers=1
                            )
                        )
                    ]
                else:
                    layers = [
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch + ich,
                                time_embed_dim,
                                dropout,
                                out_channels=model_channels * mult,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                temporal_module_kwargs=temporal_module_kwargs,
                                temporal_block_layers=1
                            )
                        )
                    ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            checkpoint_wrapper_fn(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                            if not use_spatial_transformer
                            else checkpoint_wrapper_fn(
                                SpatialTransformer(
                                    ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth[level],
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_transformer,
                                    attn_type=spatial_transformer_attn_type,
                                    use_checkpoint=use_checkpoint,
                                    temporal_module_kwargs=temporal_module_kwargs,
                                    temporal_block_layers=transformer_depth[level]
                                )
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = checkpoint_wrapper_fn(
            nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            )
        )
        self.ignored_modules = self.get_ignored_modules(mode='freeze_image')

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    def get_ignored_modules(self, mode='freeze_image'):
        ignored_modules = []
        if mode == 'freeze_image':
            from sgm.modules.diffusionmodules.models.video_module.motion_module import VanillaTemporalModule
            from sgm.modules.diffusionmodules.models.videomodel import ResBlock
            from sgm.modules.diffusionmodules.models.videoattention import SpatialTransformer
            for i in range(len(self.input_blocks)):
                for j in range(len(self.input_blocks[i])):
                    if isinstance(self.input_blocks[i][j], ResBlock):
                        ignored_modules += [
                            self.input_blocks[i][j].in_layers,
                            self.input_blocks[i][j].h_upd,
                            self.input_blocks[i][j].x_upd,
                            self.input_blocks[i][j].emb_layers,
                            self.input_blocks[i][j].out_layers,
                            self.input_blocks[i][j].skip_connection,
                        ]
                    elif isinstance(self.input_blocks[i][j], SpatialTransformer):
                        ignored_modules += [
                            self.input_blocks[i][j].norm,
                            self.input_blocks[i][j].proj_in,
                            self.input_blocks[i][j].transformer_blocks,
                            self.input_blocks[i][j].proj_out,
                        ]
            for i in range(len(self.middle_block)):
                if isinstance(self.input_blocks[i], ResBlock):
                    ignored_modules += [
                        self.input_blocks[i][j].in_layers,
                        self.input_blocks[i][j].h_upd,
                        self.input_blocks[i][j].x_upd,
                        self.input_blocks[i][j].emb_layers,
                        self.input_blocks[i][j].out_layers,
                        self.input_blocks[i][j].skip_connection,
                    ]
                elif isinstance(self.input_blocks[i], SpatialTransformer):
                    ignored_modules += [
                        self.input_blocks[i][j].norm,
                        self.input_blocks[i][j].proj_in,
                        self.input_blocks[i][j].transformer_blocks,
                        self.input_blocks[i][j].proj_out,
                    ]
            for i in range(len(self.output_blocks)):
                for j in range(len(self.output_blocks[i])):
                    if isinstance(self.output_blocks[i][j], ResBlock):
                        ignored_modules += [
                            self.output_blocks[i][j].in_layers,
                            self.output_blocks[i][j].h_upd,
                            self.output_blocks[i][j].x_upd,
                            self.output_blocks[i][j].emb_layers,
                            self.output_blocks[i][j].out_layers,
                            self.output_blocks[i][j].skip_connection,
                        ]
                    elif isinstance(self.output_blocks[i][j], SpatialTransformer):
                        ignored_modules += [
                            self.output_blocks[i][j].norm,
                            self.output_blocks[i][j].proj_in,
                            self.output_blocks[i][j].transformer_blocks,
                            self.output_blocks[i][j].proj_out,
                        ]
            # ignored_modules += [self.out, self.label_emb, self.time_embed]
            ignored_modules += [self.out, self.time_embed]
        else:
            ignored_modules = None
        return ignored_modules
    

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        
        Notification: input: B, T*C, H, W -> output: B, T*C, H, W
        """
        # 这个forward函数是在C:\Users\11859\Desktop\stable_diffusion_video-master\sgm\modules\diffusionmodules\wrappers.py
        # 里的VideoWrapper类中的forward函数进行调用的，所以是有y的
        # x = noised_input = 参考图像和加了噪声的预测图像序列, 形状为b,t,c,h,w
        # timesteps = c_noise = sigma = sigma = sqrt((1-A)/A),A为噪声累计影响变量，为一维向量，形状为b
        # context = 图像condition，有4个不同大小的condition，对应Unet下采样3次的4中预测变量图像，形状为b*h*w,l,d
        # y = 图像大小信息，形状为b,d
        # TODO ZSJ 想办法把图片condition加进来        
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # 在这里噪声强度信息timestep和图片大小信息vector融合在了一起
            assert y.shape[0] == x.shape[0] #TODO check
            emb = emb + self.label_emb(y)
        
        # 把过去图像和加噪预测图像卷积在一起
        b,t,c,h,w = x.shape
        x = x.view(b, t*c, h, w).contiguous()  # b,t*c,h,w
        x = self.conv_past_future(x)  # b,t2*c,h,w

        # 给condition图像加上时间编码，不需要加空间位置编码，因为condition适合遥感图像逐像素作用的
        for i in range(len(context)):
            context[i] = self.context_pos_encoder(context[i])
        # 给输入的图像添加时间编码
        x = self.input_pos_encoder(x.view(b*h*w, self.noise_image_num, c))

        x = x.view(b, self.noise_image_num, self.in_channels, h, w).contiguous()  # b,t2,c,h,w
        h = x
        # print(f'origin h:{h[0][0][0]}')
        # print(f'origin context:{context[0][0]}')
        # # context在时间维度上复制的t次
        # context = rearrange(context.repeat(x.shape[1],1,1,1), "t b c h -> (b t) c h")

        # Down
        for module in self.input_blocks:
            if isinstance(module._modules['0'], nn.Conv2d) or isinstance(module._modules['0'], Downsample) or isinstance(module._modules['0'], Upsample):
                B, T, C, H, W = h.shape
                new_h = h.view(B * T, C, H, W).contiguous()
                new_h = module(new_h, emb, context)
                _, C, H, W = new_h.shape
                h = new_h.view(B, T, C, H, W).contiguous()
            else:
                h = module(h, emb, context)
            hs.append(h) # append BTCHW
            # print(f"down h:{h[0][0][0][0]}")
        
        # Middle
        h = self.middle_block(h, emb, context)
        # print(f"middle h:{h[0][0][0][0]}")
        
        # torch.Size([2, 16, 10, 256, 64]) torch.Size([2, 10, 77, 64]) torch.Size([2, 10, 77, 64])
        # Up
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=2)
            if isinstance(module._modules['0'], nn.Conv2d) or isinstance(module._modules['0'], Downsample) or isinstance(module._modules['0'], Upsample):
                B, T, C, H, W = h.shape
                new_h = h.view(B * T, C, H, W).contiguous()
                new_h = module(new_h, emb, context)
                _, C, H, W = new_h.shape
                h = new_h.view(B, T, C, H, W).contiguous()
            else:
                h = module(h, emb, context)
            # print(f"up h:{h[0][0][0][0]}")
        h = h.type(x.dtype)
        B, T, C, H, W = h.shape
        h = h.view(B * T, C, H, W).contiguous()
        h = self.out(h).view(B, T, -1, H, W).contiguous()
        # print(f"out h:{h[0][0][0][0]}")
        # # 只输出预测图像的预测噪声
        # predict_h = h[:, -self.noise_image_num:, ...]
        return h



class NoTimeUNetModel(VideoUNetModel):
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        timesteps = th.zeros_like(timesteps)
        return super().forward(x, timesteps, context, y, **kwargs)


if __name__ == "__main__":

    class Dummy(nn.Module):
        def __init__(self, in_channels=3, model_channels=64):
            super().__init__()
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(2, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )

    model = VideoUNetModel(
        use_checkpoint=True,
        image_size=64,
        in_channels=4,
        out_channels=4,
        model_channels=128,
        attention_resolutions=[4, 2],
        num_res_blocks=2,
        channel_mult=[1, 2, 4],
        num_head_channels=64,
        use_spatial_transformer=False,
        use_linear_in_transformer=True,
        transformer_depth=1,
        legacy=False,
    ).cuda()
    x = th.randn(11, 4, 64, 64).cuda()
    t = th.randint(low=0, high=10, size=(11,), device="cuda")
    o = model(x, t)
    print("done.")
