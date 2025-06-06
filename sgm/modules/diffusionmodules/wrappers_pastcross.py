import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs
        )

class DiTWrapper(IdentityWrapper):
    def forward(
        self, without_noise_input: torch.Tensor, with_noise_input: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        with_noise_input = torch.cat((with_noise_input, c.get("concat", torch.Tensor([]).type_as(with_noise_input))), dim=1)
        return self.diffusion_model(
            without_noise_input,
            with_noise_input,
            t=t,
            # context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs
        )

class VideoWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        # x = noised_input = 参考图像和加了噪声的预测图像序列, 形状为b,t*c,h,w
        # t = c_noise = sigma = sigma = sqrt((1-A)/A),A为噪声累计影响变量，为一维向量
        # c = cond = 字典，text在两个embedder的中间层crossatten拼在了一起，形状为b,l,d
        # text在第二个embedder的最终输出vector和图片大小信息vector拼接在了一起，形状为b,d
        # 两个image embedder的输出拼接在了一起，形状为b,t,l,d
        
        # 输出仍然为x
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs
        )