import torch.nn as nn
import torch

from ...util import append_dims, instantiate_from_config
from einops import rearrange


class Denoiser(nn.Module):
    def __init__(self, weighting_config, scaling_config):
        super().__init__()

        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network, without_noise_input, with_noise_input, sigma, cond):
        # 根据参考图像预测未来图像的任务范式，修改一下去噪过程
        # without_noise_input和with_noise_input的形状为(b,t,c,h,w)
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, with_noise_input.ndim)  # 五维,和最后要一起计算的predict_noise形状相匹配
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # c_skip=sigma形状的全1值，c_out=-sigma, c_in=1/(sigma**2+1)**0.5, c_noise=sigma
        # c_in计算结果为sqrt(A),其中A为噪声累计影响，加噪后的图像的计算方式为x'=sqrt(1-A)*x+sqrt(A)*N(0,1)
        # input*c_in的计算结果是sqrt(1-A)*x+sqrt(A)*N(0,1)，其中x为原图像正式加噪后的图像
        # network的计算结果正是要不断逼近加入的高斯噪声
        # 因此一切其实很清晰，只是各种计算看起来很复杂，尤其是network里面不知道为什么要加入c_noise这个sigma参数(是为了告诉网络噪声级别从而针对性去噪)
        # 下面是原来的代码
        # return network(input * c_in, c_noise, cond) * c_out + input * c_skip
        # c_in = append_dims(c_in, with_noise_input.ndim)
        with_noise_input = with_noise_input * c_in
        # b,t1,c,h,w = without_noise_input.shape
        # 把过去和未来图像分别输进去，不拼在一起
        noised_input = torch.cat([without_noise_input, with_noise_input], dim=1)  # (b,t,c,h,w)
        # b,t,c,h,w = noised_input.shape
        # noised_input = noised_input.view(b,t*c,h,w).contiguous()  # 四维

        # 没有问题，下面的cond应该要拆开成vector和crossattn，但是没有拆，因为在model_wrapper里面拆开了
        # print(f'dtype: noised_input:{noised_input.dtype}, c_noise:{c_noise.dtype}')
        # print(f'content: c_skip:{c_skip}, c_out:{c_out}, c_in: {c_in}, c_noise:{c_noise}')
        # predict_noise = network(without_noise_input, with_noise_input, c_noise, cond)  # 注意这里模型输出预测图像的预测噪声， b,t2,c,h,w
        predict_noise = network(noised_input, c_noise, cond)  # 注意这里模型输出预测图像的预测噪声， b,t2,c,h,w
        # print(f'noise_input:{noised_input[0][0][0]}')
        # print(f'predict_noise:{predict_noise[0][0][0]}')
        # print(f'c_out:{c_out}')
        # print(f'c_skip:{c_skip}')
        # predict_noise = predict_noise.view(b,t,c,h,w).contiguous()
        # predict_noise = predict_noise[:, t1:, ...]  # 只保留预测图像的预测噪声，b,t2,c,h,w
        # 输出的图像只有预测的部分，没有过去的部分

        # predict_noise = rearrange(predict_noise, "b (t c) h w -> b t c h w", t=noised_num).contiguous()  # b,t2,c,h,w

        # predict_noise = predict_noise.view(b,(t-t1)*c,h,w).contiguous()  # b,t2*c,h,w
        # with_noise_input = with_noise_input.view(b,(t-t1)*c,h,w).contiguous()  # b,t2*c,h,w
        # print(f'predict_noise:{predict_noise.dtype}, c_out:{c_out.dtype}, with_noise:{with_noise_input.dtype}, c_skip:{c_skip.dtype}')

        # return predict_noise * c_out + with_noise_input * c_skip  # 这里的计算结果是去噪后的预测图像本身, b,t2,c,h,w

        return predict_noise  # 这里的计算结果是去噪后的预测图像本身, b,t2,c,h,w，即网络输出的结果就是去噪后的图片


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        weighting_config,
        scaling_config,
        num_idx,
        discretization_config,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        # 之所以要sigma到idx再到sigma，是因为之前的sigma是通过loss函数的噪声级别数量设置的，
        # 而去噪过程中可能设置另外一个噪声级别数目，因此要找到和之前的噪声级别最接近的现在的噪声级别
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise





