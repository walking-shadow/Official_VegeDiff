

import math
import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from torch import nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention as CrossAttention
from einops import rearrange, repeat
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func



def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels, 
    motion_module_kwargs: dict
):
    return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)

def get_motion_module_layer(
    in_channels, 
    num_transformer_block,
    noise_image_num,
    motion_module_kwargs: dict
):
    return VanillaTemporalModule(in_channels=in_channels, num_transformer_block=num_transformer_block, 
                                noise_image_num=noise_image_num, **motion_module_kwargs,)

class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        noise_image_num                    = 20,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        gradient_checkpoint                = False,
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            noise_image_num=noise_image_num,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            gradient_checkpoint=gradient_checkpoint
        )
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)



    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)
        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        noise_image_num,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        gradient_checkpoint                = False
    ):
        super().__init__()
        self.noise_image_num = noise_image_num

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    noise_image_num=noise_image_num,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)   
        self.gradient_checkpoint = gradient_checkpoint 
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        
        without_noise_state, with_noise_state = hidden_states[:,:,:-self.noise_image_num,...], hidden_states[:,:,-self.noise_image_num:,...]

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()

        batch, channel, height, weight = hidden_states.shape  # b*t,c,h,w
        # 运行时序交叉注意力之后，得到的结果只有未来图像部分
        # residual = hidden_states
        with_noise_state = rearrange(with_noise_state, "b c f h w -> (b f) c h w").contiguous()
        residual = with_noise_state

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)  # b*t, h*w, c

        # 分离过去图像和未来图像
        hidden_states_temp = rearrange(hidden_states, "(b t) d c -> b t d c", t=video_length).contiguous()
        without_noise_proj_in, with_noise_state_proj_in = hidden_states_temp[:,:-self.noise_image_num,...], hidden_states_temp[:,-self.noise_image_num:,...]
        # without_noise_proj_in = rearrange(without_noise_proj_in, "b t d c -> (b t) d c").contiguous()
        # with_noise_state_proj_in = rearrange(with_noise_state_proj_in, "b t d c -> (b t) d c").contiguous()


        # Transformer Blocks
        for block in self.transformer_blocks:
            if self.gradient_checkpoint and self.training:
                # 这里运行时序交叉注意力, 结果只有未来图像部分
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), 
                    hidden_states, 
                    encoder_hidden_states, 
                    attention_mask,
                    video_length
                )  # 输出形状为b*t, h*w, c
                
                hidden_states = rearrange(hidden_states, "(b t) d c -> b t d c", t=self.noise_image_num).contiguous()
                hidden_states = torch.cat([without_noise_proj_in, hidden_states], dim=1)  # 注意形状
                hidden_states = rearrange(hidden_states, "b t d c -> (b t) d c").contiguous()
            else:
                hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)

                hidden_states = rearrange(hidden_states, "(b t) d c -> b t d c", t=self.noise_image_num).contiguous()
                hidden_states = torch.cat([without_noise_proj_in, hidden_states], dim=1)  # 注意形状
                hidden_states = rearrange(hidden_states, "b t d c -> (b t) d c").contiguous()

        # output
                
        # 分离过去图像和未来图像,选取未来图像部分作为输出
        hidden_states_temp = rearrange(hidden_states, "(b t) d c -> b t d c", t=video_length).contiguous()
        without_noise_proj_in, with_noise_state_proj_in = hidden_states_temp[:,:-self.noise_image_num,...], hidden_states_temp[:,-self.noise_image_num:,...]
        # without_noise_proj_in = rearrange(without_noise_proj_in, "b t d c -> (b t) d c").contiguous()
        hidden_states = rearrange(with_noise_state_proj_in, "b t d c -> (b t) d c").contiguous()

        hidden_states = self.proj_out(hidden_states)
        # hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=height).contiguous()  # b*t,c,h,w


        output = hidden_states + residual
        # output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length).contiguous()
        output = rearrange(output, "(b f) c h w -> b c f h w", f=self.noise_image_num).contiguous()
        
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        noise_image_num,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        self.noise_image_num = noise_image_num
        
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    noise_image_num=noise_image_num,
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        # hidden_states: (b*t, h*w, c), video_length: t,

        # 分离过去图像和未来图像
        hidden_states_temp = rearrange(hidden_states, "(b t) d c -> b t d c", t=video_length).contiguous()
        without_noise_proj_in, with_noise_state_proj_in = hidden_states_temp[:,:-self.noise_image_num,...], hidden_states_temp[:,-self.noise_image_num:,...]
        # without_noise_proj_in = rearrange(without_noise_proj_in, "b t d c -> (b t) d c").contiguous()
        with_noise_state_proj_in = rearrange(with_noise_state_proj_in, "b t d c -> (b t) d c").contiguous()

        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
                        
            # hidden_states = attention_block(
            #     norm_hidden_states,
            #     encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
            #     attention_mask=attention_mask,
            #     video_length=video_length,
            # ) + hidden_states
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                attention_mask=attention_mask,
                video_length=video_length,
            ) + with_noise_state_proj_in  # 输出形状为b*t, h*w, c

            hidden_states = rearrange(hidden_states, "(b t) d c -> b t d c", t=self.noise_image_num).contiguous()
            hidden_states = torch.cat([without_noise_proj_in, hidden_states], dim=1)  # 注意形状，两个变量形状均为b,t, h*w, c
            hidden_states = rearrange(hidden_states, "b t d c -> (b t) d c").contiguous()

        # 分离过去图像和未来图像
        hidden_states_temp = rearrange(hidden_states, "(b t) d c -> b t d c", t=video_length).contiguous()
        without_noise_proj_in, with_noise_state_proj_in = hidden_states_temp[:,:-self.noise_image_num,...], hidden_states_temp[:,-self.noise_image_num:,...]
        without_noise_proj_in = rearrange(without_noise_proj_in, "b t d c -> (b t) d c").contiguous()
        with_noise_state_proj_in = rearrange(with_noise_state_proj_in, "b t d c -> (b t) d c").contiguous()

        # 只保留未来图像部分作为输出
        # hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        hidden_states = self.ff(self.ff_norm(with_noise_state_proj_in)) + with_noise_state_proj_in


        output = hidden_states  
        return output  # 输出形状为b*t, h*w, c


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# class VersatileAttention(CrossAttention):
#     def __init__(
#             self,
#             attention_mode                     = None,
#             cross_frame_attention_mode         = None,
#             temporal_position_encoding         = False,
#             temporal_position_encoding_max_len = 24,            
#             *args, **kwargs
#         ):
#         super().__init__(*args, **kwargs)
#         assert attention_mode == "Temporal"

#         self.attention_mode = attention_mode
#         self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
#         # true
#         self.pos_encoder = PositionalEncoding(
#             kwargs["query_dim"],
#             dropout=0., 
#             max_len=temporal_position_encoding_max_len
#         ) if (temporal_position_encoding and attention_mode == "Temporal") else None

#     def extra_repr(self):
#         return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

#     def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, attn_type='flash'):
#         # hidden_states: (b*t, h*w, c), video_length: t, others=default
#         if self.attention_mode == "Temporal":
#             d = hidden_states.shape[1]
#             hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)  # b*h*w, t, c
            
#             if self.pos_encoder is not None:
#                 hidden_states = self.pos_encoder(hidden_states)
            
#             encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
#         else:
#             raise NotImplementedError
#         batch_size, sequence_length, _ = hidden_states.shape
#         encoder_hidden_states = encoder_hidden_states

#         if self.group_norm is not None:
#             hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         # query = self.to_q(hidden_states)
#         # dim = query.shape[-1]
#         # # query = self.reshape_heads_to_batch_dim(query)
#         # query = self.head_to_batch_dim(query)

#         # if self.added_kv_proj_dim is not None:
#         #     raise NotImplementedError

#         # encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
#         # key = self.to_k(encoder_hidden_states)
#         # value = self.to_v(encoder_hidden_states)

#         # # key = self.reshape_heads_to_batch_dim(key)
#         # key = self.head_to_batch_dim(key)
#         # # value = self.reshape_heads_to_batch_dim(value)
#         # value = self.head_to_batch_dim(value)

#         # if attention_mask is not None:
#         #     if attention_mask.shape[-1] != query.shape[1]:
#         #         target_length = query.shape[1]
#         #         attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
#         #         attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

#         # # # attention, what we cannot get enough of
#         # # if self._use_memory_efficient_attention_xformers:
#         # #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
#         # #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
#         # #     hidden_states = hidden_states.to(query.dtype)
#         # # else:
#         # #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
#         # #         hidden_states = self._attention(query, key, value, attention_mask)
#         # #     else:
#         # #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

#         # # linear proj
#         # hidden_states = self.to_out[0](hidden_states)

#         # # dropout
#         # hidden_states = self.to_out[1](hidden_states)

#         query = self.to_q(hidden_states)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif self.norm_cross:
#             encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

#         key = self.to_k(encoder_hidden_states)
#         value = self.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // self.heads

#         if attn_type == 'flash':
#             query = query.view(batch_size, -1, self.heads, head_dim)  # b*h*w, t, c -> b*h*w, t, h, c/h
#             key = key.view(batch_size, -1, self.heads, head_dim)
#             value = value.view(batch_size, -1, self.heads, head_dim)
#         elif attn_type == 'pytorch':
#             query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#             key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#             value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         if attn_type == 'flash':
#             hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
#         elif attn_type == 'pytorch':
#             hidden_states = F.scaled_dot_product_attention(
#                 query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#             )
        
#         if attn_type == 'flash':
#             hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim)  # b*h*w, t, h, c/h -> b*h*w, t, c
#         elif attn_type == 'pytorch':
#             hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         # linear proj
#         hidden_states = self.to_out[0](hidden_states)
#         # dropout
#         hidden_states = self.to_out[1](hidden_states)
#         if self.attention_mode == "Temporal":
#             hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

#         return hidden_states
    



class VersatileAttention(CrossAttention):
    def __init__(
            self,
            noise_image_num                    = None,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 24,            
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.noise_image_num = noise_image_num
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        # true
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, attn_type='flash'):
        # hidden_states: (b*t, h*w, c), video_length: t, others=default
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length).contiguous()  # b*h*w, t, c
            
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 分离过去图像和未来图像，此处hidden_states形状为b*h*w, t, c
        without_noise_proj_in, with_noise_state_proj_in = hidden_states[:,:-self.noise_image_num,...], hidden_states[:,-self.noise_image_num:,...]

        # query为未来图像部分
        # query = self.to_q(hidden_states)
        query = self.to_q(with_noise_state_proj_in)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        # K和V为过去图像部分
        # key = self.to_k(encoder_hidden_states)
        # value = self.to_v(encoder_hidden_states)
        key = self.to_k(without_noise_proj_in)
        value = self.to_v(without_noise_proj_in)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        if attn_type == 'flash':
            query = query.view(batch_size, -1, self.heads, head_dim)  # b*h*w, t, c -> b*h*w, t, h, c/h
            key = key.view(batch_size, -1, self.heads, head_dim)
            value = value.view(batch_size, -1, self.heads, head_dim)
        elif attn_type == 'pytorch':
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if attn_type == 'flash':
            # 把causal设置为True，就是只能看到过去的帧，不能看到未来的帧
            # 实验证明False比True更好，因果编码不应该使用
            hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
        elif attn_type == 'pytorch':
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        
        if attn_type == 'flash':
            hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim)  # b*h*w, t, h, c/h -> b*h*w, t, c
        elif attn_type == 'pytorch':
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d).contiguous()

        # 只输出未来图像部分
        return hidden_states