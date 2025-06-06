"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config
from einops import rearrange

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=x.device
        )
        uc = default(uc, cond)
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc, past_images, noise_image_num):
        # print("#"*10)
        # print(*self.guider.prepare_inputs(x, sigma, cond, uc))
        # print(denoiser)
        #TODO denoiser change the input into fp32!
        # 输出是两个x在维度0拼接，两个sigma在维度0拼接，condition变量中两个crossattn在维度0拼接，一个部分为0，一个全为0，其他不变，两个past_images在维度0拼接。x的形状从b,t*c,h,w变为b,t,c,h,w
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc, past_images, noise_image_num))
        # print(denoised.dtype)
        denoised = self.guider(denoised, sigma)
        # print(denoised.dtype)
        # print("#"*10)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, past_images=None, noise_image_num=None):
        sigma_hat = sigma * (gamma + 1.0)  # sigma_hat=sigma
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, past_images, noise_image_num)  # 输出去噪图像
        # # 另一个选择，把噪声和过去均值图像和未来图像都拼接起来
        # x = rearrange(x, "b (t c) h w -> b t c h w", t=noise_image_num).contiguous()
        # x = x[:,:,:4,...]
        # x = rearrange(x, "b t c h w -> b (t c) h w").contiguous()

        denoised = rearrange(denoised, "b t c h w -> b (t c) h w")
        d = to_d(x, sigma_hat, denoised)  # d = (x - denoised) / append_dims(sigma, x.ndim)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        # 此处假设模型预测噪声，其中x为生成过程中的初始图像（本来是全噪声图像，在这里是噪声图像和过去均值图像的加权和), 
        # 实际上模型的输出结果是噪声和过去均值图像的一种杂糅，现在的关键是对模型的输出，使用过去均值图像进行操作，使得它能够最接近下一个级别噪声的输入图像
        # denoised_k 为去噪器在第k步的时候的结果，为去噪图像
        # 目前的x_n=sqrt(s_n**2+1)*(w*past_img+(1-w)*noise),x_n表示最开始的图像
        # 目前的euler_step = x_k = x_k+1 + dt * d = x_k+1 + (sk - s_k+1)*(x_k+1 - denoised_k)/s_k = s_k/s_k+1 * (x_k+1 - denoised_k) + denoised_k
        # = x_k+1 +(sk+1 - s_k)*(x_k+1-(x_k+1 - s_k*predict_noise_k))/s_k = x_k+1 - (sk+1 - s_k)*predict_noise
        # 目前的denoise输出为 denoise_k = (x_k+1 - s_k*predict_noise_k), x_k+1 - x_k = (sk+1 - s_k)*predict_noise = (sk+1 - s_k)*(x_k+1 - denoised_k)/s_k 
        # 训练的输入的x_n = x0 + s_n*(w*past_img+(1-w)*noise)
        # 训练的输入的x_k = x0 + s_k*(w*past_img+(1-w)*noise)
        # 训练的denoiser的输出也是denoise_k = x_k = x_k+1 - s_k*predict_noise_k = x0 + s_k+1*(w*past_img+(1-w)*noise) - s_k+1*predict_noise_k
        # 其中计算损失使得predict_noise_k 不断趋近于w*past_img+(1-w)*noise
        # 因此验证的生成时，首先输入x_n，得到的denoise_n-1 = 
        # 训练的输入的 x_k+1 - x_k = (sk+1 - s_k)*(w*past_img+(1-w)*noise)
        # 因此期望的情况与真实的情况为diff = w*past_img+(1-w)*noise - predict_noise，而predict_noise一般来说等于(1-w)*noise，
        # 因此每次在合适的位置加入w*past_img即可，即predict_noise_new = predict_noise + w*past_img
        # 现实情况中denoise_k往往不可拆分（尤其是模型直接预测图片），对denoise_k进行修改，denoise_k_old = (x_k+1 - s_k*predict_noise_k_old)
        # denoise_k_new = (x_k+1 - s_k*predict_noise_k_new) = (x_k+1 - s_k*predict_noise_k_old) - s_k*w*past_img = denoise_k_old - s_k*w*past_img
        euler_step = self.euler_step(x, d, dt)  
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, past_images=None, noise_image_num=None,
                past_mean=None, past_weigth=None):
        # 这里的x指的是随机噪声，形状为b,t*c,h,w；num_step为去噪次数；s_in为形状为b的全1值；sigmas为形状为b的噪声级别；
        # cond, uc都是condition变量，num_sigmas为sigmas的长度，为b
        # past_images指的是过去的图像，它处于latent space中
        # past_mean是过去的均值图像，形状为b,t*c,h,w
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )  # x = x*sqrt(sigmas[0]**2+1)
        # print(f'sigma: {sigmas}')
        # 尝试把sigmas乘上z噪声的权重系数看看
        sigmas = sigmas * (1-past_weigth)
        # get_sigma_gen(num_sigmas) = range(num_sigmas - 1)
        for i in self.get_sigma_gen(num_sigmas):
            # gamma=0
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            # gamma=0
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                past_images,
                noise_image_num
            )
            # # 每一步去噪的时候，都加上噪声系数*过去图像权重*过去图像，从而和训练保持一致
            # x = x + past_weigth*past_mean

            # # 另一个选择，把噪声和过去均值图像和未来图像都拼接起来
            # x = rearrange(x, "b (t c) h w -> b t c h w", t=noise_image_num).contiguous()  # b,t,c,h,w
            # x = torch.cat([x, past_mean.repeat(1,x.shape[1],1,1,1)], dim=2)
            # x = rearrange(x, "b t c h w -> b (t c) h w").contiguous()  # b,t*c,h,w


        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x
