from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (
    ByT5Tokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from ...modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from ...modules.diffusionmodules.model import Encoder
from ...modules.diffusionmodules.openaimodel import Timestep
from ...modules.diffusionmodules.util import extract_into_tensor, make_beta_schedule
from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...util import (
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)
from einops import rearrange


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

# TODO zsj 这个类负责控制SDXL的condition，需要修改
class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1, "condition_image": 3}

    def __init__(self, emb_models: Union[List, ListConfig]):
        # 加载所有conditioner并初始化，丢到embedders列表里面
        # 所有的embedder的ucg_rate都是0，而不是设置文件里面设置的值
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            # 我的condition image通过的是resnet
            # assert isinstance(
            #     embedder, AbstractEmbModel
            # ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            embedder.pretrained_path = embconfig.get("pretrained_path", None)
            embedder.in_channel = embconfig.get("in_channel", None)
            embedder.out_channel = embconfig.get("out_channel", None)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            # if embedder.input_key == 'highres_condition_image':
            #     # 输入的是静态环境变量，形状为B,5,H,W
            #     embedder.load_state_dict(torch.load(embedder.pretrained_path))
            #     # 创建新的特定通道输入卷积层
            #     new_first_conv_layer = torch.nn.Conv2d(embedder.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #     # 用新的卷积层替换原始的第一卷积层
            #     embedder.conv1 = new_first_conv_layer
            #     embedder = nn.Sequential(
            #                 *[child for _, child in list(embedder.named_children())[:6]]  # 控制resnet18下采样8倍
            #             )
            #     # new_embedder = nn.Sequential(
            #     #             # 只提取每个元组的模块部分
            #     #             *[child for _, child in list(embedder.named_children())[:-2]]  # 控制resnet18全部下采样层
            #     #         )
            #     # conv_out_layer = nn.Sequential(
            #     #     torch.nn.Conv2d(128, embedder.out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            #     #     torch.nn.Conv2d(256, embedder.out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            #     #     torch.nn.Conv2d(512, embedder.out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            #     #     torch.nn.Conv2d(512, embedder.out_channel, kernel_size=3, stride=2, padding=1, bias=True)
            #     # )
            #     # new_embedder = nn.Sequential(*(list(new_embedder.children()) + list(conv_out_layer.children())))
            #     # new_embedder.input_key = embedder.input_key
            #     # new_embedder.ucg_rate = embedder.ucg_rate
            #     # new_embedder.legacy_ucg_val = embedder.legacy_ucg_val
            #     # new_embedder.is_trainable = embedder.is_trainable
            #     # embedder = new_embedder

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    # def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
    def possibly_get_ucg_val(self, embedder, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                # all embedder has input_key attribute
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    # embedder.legacy_ucg_val is None
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    # condition_img = batch[embedder.input_key]
                    emb_out = embedder(batch[embedder.input_key])

                    # if embedder.input_key == 'meso_condition_image':
                    #     # 输入的是动态气象变量，形状为B,T,C
                    #     b,t,c = condition_img.shape
                    #     condition_img = condition_img.transpose(1,2)

                    #     # condition_img = condition_img.reshape((b*t,c,h,w))

                    #     emb_out = embedder(condition_img)
                    #     emb_out = emb_out.transpose(1,2)  # 形状为b,l,d

                    # elif embedder.input_key == 'highres_condition_image':
                    #     # 输入的是静态环境变量，形状为B,5,H,W

                    #     b,c,h,w = condition_img.shape
                    #     # condition_img = condition_img.reshape((b*t,c,h,w))
                    #     # print(f'original highres:{condition_img[0]}')
                    #     # 保存各种分辨率下的highres_condition_image，方便之后进行交叉注意力计算，因为预测变量在下采样而condition不变
                    #     n_embedder = len(embedder)
                    #     assert n_embedder == 12, f'resnet of highres_condition should with 12 layers, not {n_embedder}!'
                    #     for i in range(n_embedder-4):
                    #         condition_img = embedder[i](condition_img)
                    #         if i == 5:
                    #             condition_img_d8 = embedder[-4](condition_img)
                    #             # print(f'condition_img_d8:{condition_img_d8[0]}')
                    #         elif i == 6:
                    #             condition_img_d16 = embedder[-3](condition_img)
                    #             # print(f'condition_img_d16:{condition_img_d16[0]}')

                    #         elif i == 7:
                    #             condition_img_d32 = embedder[-2](condition_img)
                    #             condition_img_d64 = embedder[-1](condition_img)
                    #             # print(f'condition_img_d32:{condition_img_d32[0]}')
                    #             # print(f'condition_img_d64:{condition_img_d64[0]}')

                    #     emb_out = [condition_img_d8, condition_img_d16, condition_img_d32, condition_img_d64]
                    #     for i in range(len(emb_out)):
                    #         emb = emb_out[i]
                    #         _,c,h,w = emb.shape
                    #         emb_out[i] = rearrange(emb, "b c h w -> b (h w) c", b=b, h=h, w=w).contiguous()  # 形状为b,h*w,d

                    # else:
                    #     # timestamp, 形状为b,d
                    #     emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for i in range(len(emb_out)):
                emb = emb_out[i]
                # 由于不同的text embedder的输出维度不同，因此可以根据维度来确定输出是最终层还是中间层
                # out_key = self.OUTPUT_DIM2KEYS[emb.dim()] if embedder.input_key != 'condition_image' else 'condition_image'
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    # 对第一维使用伯努利分布进行随机的掩码，表示随机把一些condition变成0，进行无条件生成，注意第一维应该是b
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                # 满足下面的条件就把emb变成全0值，验证的时候uc满足该条件，进行无条件生成
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                emb_out[i] = emb
            output[embedder.input_key] = emb_out
        # # output里面有5个键值对，每个值都是一个列表，其中highres_condition_image有4个不同大小的图像
        # crossattn = []
        # for emb in output['highres_condition_image']:
        #     # print(f'after expand_dims_like highres:{emb[0]}')
        #     b,hw,ch = emb.shape
        #     emb = emb.view((b*hw, ch))
        #     meso_cond = output['meso_condition_image'][0]
        #     b,t,cm = meso_cond.shape
        #     meso_cond = meso_cond.repeat(hw,1,1)  # bhw,t,cm
        #     emb = emb.unsqueeze(1).repeat(1,t,1)  # bhw,t,ch
        #     crossattn.append(torch.cat([meso_cond, emb], 2))
        # output['crossattn'] = crossattn
            
        # 需要处理的条件在DiT中只有静态变量和气象变量，没有其他，output的值都是只有一个元素的列表
        output['crossattn'] = output['highres_condition_image'][0]
        output['vector'] = output['meso_condition_image'][0]

        # output['vector'] = torch.cat([output['original_size_as_tuple'][0], output['crop_coords_top_left'][0], output['target_size_as_tuple'][0]], 1)
        # output为一个字典，其中crossattn键对应一个列表，包括4个不同大小的condition(气象变量和环境变量),形状为bhw,t,c, vector键对应图像大小信息,形状为b,c
        # 其他的原始数据键(包括meso_condition_image这些condition数据)和它们通过conditioner之后的特征对应
                # if out_key in output:
                #     # 同类型的condition拼接起来，vector形状为b,d，因此在第二维度拼接，
                #     # crossattn形状为b,l,d，因此在第三维度拼接
                #     if out_key == "crossattn":
                #         # 把meso condition image形状扩展为b*h*w,l,d，从而能够和highres condition image拼接
                #         if output[out_key].shape[0] > emb.shape[0]:
                #             emb = emb.repeat(output[out_key].shape[0]//emb.shape[0], 1, 1)
                #         else:
                #             output[out_key] = output[out_key].repeat(emb.shape[0]//output[out_key].shape[0], 1, 1)

                #     # # print(f"{out_key} shape:{output[out_key].shape}")
                #     output[out_key] = torch.cat(
                #         (output[out_key], emb), self.KEY2CATDIM[out_key]
                #     )
                # else:
                #     output[out_key] = emb
        
        # 输出为字典，key为vector, crossattn, 值为对应的tensor
        # 所有condition image的序列键为crossattn，形状为b*h*w,l,d
        # 图片大小信息(裁剪信息）vector拼接在了一起，形状为b,d
        return output

    def get_unconditional_conditioning(
        self, batch_c, batch_uc=None, force_uc_zero_embeddings=None
    ):
        # 输出的结果中c和uc都是condition变量，即对batch运行forward函数的结果, 起码在val里面，这个函数就是单独对batch_c运行两次forward，得到两个相同的结果
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class InceptionV3(nn.Module):
    """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
    port with an additional squeeze at the end"""

    def __init__(self, normalize_input=False, **kwargs):
        super().__init__()
        from pytorch_fid import inception

        kwargs["resize_input"] = True
        self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

    def forward(self, inp):
        # inp = kornia.geometry.resize(inp, (299, 299),
        #                              interpolation='bicubic',
        #                              align_corners=False,
        #                              antialias=True)
        # inp = inp.clamp(min=-1, max=1)

        outp = self.model(inp)

        if len(outp) == 1:
            return outp[0].squeeze()

        return outp


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):
    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-xxl", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    # @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(
        self, version="google/byt5-base", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

# TODO ZSJ 这个类负责把图片变为token，具体看看怎么控制它的输出维度，和text通道维度保持一致
# 使用同一个CLIP，图片和文本的dim就会相同，同时图像的embedding是一维向量，因此文本需要通过
# 平均池化或者最大池化，或者用CLS方法把所有信息聚合到一维向量中
class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEmbModel):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="cuda",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length
        )
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
        wrap_video=False,
        kernel_size=1,
        remap_output=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video

    def forward(self, x):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> b t c h w")
            x = rearrange(x, "b t c h w -> (b t) c h w")

        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)
            x = rearrange(x, "b t c h w -> b c t h w")
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class LowScaleEncoder(nn.Module):
    def __init__(
        self,
        model_config,
        linear_start,
        linear_end,
        timesteps=1000,
        max_noise_level=250,
        output_size=64,
        scale_factor=1.0,
    ):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def forward(self, x):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(
            0, self.max_noise_level, (x.shape[0],), device=x.device
        ).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")
        # z = z.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)

class Identify_Mapping(nn.Module):
    """
    用来直接把气象变量直接输入模型
    """
    def __init__(self, param1):
        super().__init__()

    def forward(self, x):
        return x

class Climate_Extractor(nn.Module):
    """
    用来提取一维气象变量的特征，输入的特征形状为B,T,C
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel//8, kernel_size=1),
            nn.BatchNorm1d(out_channel//8),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel//8, out_channels=out_channel//8, kernel_size=1),
            nn.BatchNorm1d(out_channel//8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel//4, out_channels=out_channel//4, kernel_size=1),
            nn.BatchNorm1d(out_channel//4),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=1)
        )
    
    def forward(self, x):
        # shape of x: B,C,T
        x = self.conv_in(x)
        # x = self.conv1(x) + x
        # x = self.conv2(x) + x
        x = torch.cat([self.conv1(x), x], dim=1)
        x = torch.cat([self.conv2(x), x], dim=1)
        out = self.conv3(x)

        return out



class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):
    def __init__(
        self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) -> Tuple[Dict, torch.Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        return log, z



class AdaLNZeroEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes+1, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"): #   discard
        uc_class = (
            self.n_classes
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc

'''
logic: according to input_key to get data and generate condition
'''
class AdaLNZeroConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    labels = batch[embedder.input_key]
                    drop_ids = torch.rand(labels.shape[0], device=labels.device) < embedder.ucg_rate
                    labels = torch.where(drop_ids, embedder.n_classes, labels)
                    emb_out = embedder(labels)
                    # emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    labels = batch[embedder.input_keys[0]]
                    drop_ids = torch.rand(labels.shape[0], device=labels.device) < embedder.ucg_rate
                    labels = torch.where(drop_ids, self.n_classes, labels)
                    emb_out = embedder(labels)
                    # emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                # if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    # emb = (
                    #     expand_dims_like(
                    #         torch.bernoulli(
                    #             (1.0 - embedder.ucg_rate)
                    #             * torch.ones(emb.shape[0], device=emb.device)
                    #         ),
                    #         emb,
                    #     )
                    #     * emb
                    # )
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb
        return output
    
    def get_unconditional_conditioning(
        self, batch_c, batch_uc=None, force_uc_zero_embeddings=None
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        if batch_uc == None:
            y_null = dict()
            y_null['cls'] = torch.tensor([1000] * batch_c['cls'].shape[0], device=batch_c['cls'].device)
            uc = self(y_null)
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc