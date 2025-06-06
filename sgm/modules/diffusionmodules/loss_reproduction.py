from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from taming.modules.losses.lpips import LPIPS
from .metric import EarthnetX_Metric
import numpy as np

from ...util import append_dims, instantiate_from_config
from einops import rearrange



class StandardDiffusionLoss(nn.Module):
    def __init__(
        self, param1
    ):
        super().__init__()

    def __call__(self, network, batch, input_future ):
        
        # for k in batch.keys():
        #     batch[k] = batch[k].to(dtype=torch.bfloat16)

        model_output = network(
            batch
        ).to(dtype=torch.bfloat16)  # b,t2,c,h,w, 范围为（0,1）
        model_output = model_output*2 - 1  # b,t2,c,h,w, 范围为（-1,1）
        
        # 在latent space里面计算损失
        loss, true_latent_rmse = self.get_latent_space_loss(model_output, input_future)


        # 选取生成的样本和重建的未来样本
        model_output_sample = model_output[0,0,...]  # c,h,w


        model_output_sample = torch.clamp(model_output_sample, -1.0, 1.0)


                
        def denormalize(array):
            return (array+1.0)/2
        model_output_sample = denormalize(model_output_sample).detach().to(dtype=torch.float).cpu().numpy()
        
        return loss, true_latent_rmse, model_output_sample

    
    def get_latent_space_loss(self, pred, target):
        # the shape of pred, target and w should be b,t,c,h,w
        # the first shape of all inputs are b
        assert pred.numel() == target.numel(), f'size of pred not equal to target: pred:{pred.numel()}, target:{target.numel()}'     

        mse = torch.mean(
            ((pred - target) ** 2).reshape(target.shape[0], -1), 1
            )  # b
        mse_loss = torch.mean(mse)
        true_rmse = torch.sqrt(torch.mean((pred - target) ** 2))

        angle_loss = self.latent_vector_angle_difference(pred, target)
        loss = mse_loss + angle_loss
        
        return loss, true_rmse


    def latent_vector_angle_difference(self, pred, target):
        # 输入图像形状应该为b,t,c,h,w
        pred_flat = rearrange(pred, "b t c h w -> (b t h w) c").contiguous()
        target_flat = rearrange(target, "b t c h w -> (b t h w) c").contiguous()

        theta = torch.nn.functional.cosine_similarity(pred_flat, target_flat)  # 取值范围为-1到1
        theta = 1 - theta  # 取值范围为0到2
        theta = torch.mean(theta)
        
        return theta

