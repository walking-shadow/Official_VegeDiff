from functools import partial

import torch

from ...util import default, instantiate_from_config


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)  # 输出就是scale
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
                },
            )
        )

    def __call__(self, x, sigma):
        # X为去噪之后的结果，sigma为和噪声级别有关的值
        x_u, x_c = x.chunk(2)
        # 值就是sigma
        scale_value = self.scale_schedule(sigma)
        # 结果是x_u + scale_value*(x_c-x_u),调控条件控制部分的比重
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc, past_images, noise_image_num):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                if isinstance(c[k], torch.Tensor):
                    c_out[k] = torch.cat((uc[k], c[k]), 0)
                elif isinstance(c[k], list):
                    condition_list = []
                    for i in range(len(c[k])):
                        condition_list.append(torch.cat((uc[k][i], c[k][i]), 0))
                    c_out[k] = condition_list
            # else:
            #     assert c[k] == uc[k]
            #     c_out[k] = c[k]
        b,tc,h,w = x.shape
        x = x.view(b,noise_image_num,-1,h,w).contiguous()
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out, torch.cat([past_images] * 2)


class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
