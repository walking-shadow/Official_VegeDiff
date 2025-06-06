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
        self,
        sigma_sampler_config,
        noise_image_num,
        past_weight,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.noise_image_num = noise_image_num
        self.past_weight = past_weight

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.metric_rgbn = EarthnetX_Metric()
        self.metric_ndvi = EarthnetX_Metric()
        self.metric_upper_limit = EarthnetX_Metric()



    def __call__(self, network, denoiser, conditioner, input, batch, input_future, x_past_mean, decode_first_stage):
        # TODO zsj 这个函数是运行模型和计算损失的，各个功能的更改可以看这里
        # input和input_future的形状为(b,t,c,h,w)，其中input_future为未来的原始图像
        # x_past_mean的形状为(b,c,h,w)
        # 默认使用L2损失，即计算预测图像和真实图像的差的平方

        # cond为字典，key为vector或者crossattn, 值为对应的tensor
        # 所有condition image的序列键为crossattn，形状为b,l,d
        # 图片大小信息(裁剪信息）vector拼接在了一起，形状为b,d
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        # sigma是一个和噪声累计影响变量A相关的值，计算方式为sigma = sqrt((1-A)/A)
        # 生成batch个的不同级别的sigma值，每个sigma和高斯噪声相乘可以得到对应级别的高斯噪声，
        # 与原始图像相加并在之后的去噪过程中乘以对应的缩放系数，就可以达到对应级别的加噪效果
        # 代表加了级别次数次的高斯噪声，比如级别为90，表示连续90次加了高斯噪声得到的结果，可以通过特定的公式一步计算得到
        # 意味着每个时间序列加的噪声强度是一样的，去预测这一个特定级别的噪声
        # 噪声级别越大，sigmas越大，范围大概是从0到16
        sigmas = self.sigma_sampler(input.shape[0]).to(input.device, dtype=torch.bfloat16)
        input = input.to(input.device, dtype=torch.bfloat16)
        # print(f'sigma:{sigmas}')
        # 设置高斯噪声
        without_noise_input, with_noise_input = input[:,:-self.noise_image_num, ...], input[:,-self.noise_image_num:, ...]
        latent_space_future = with_noise_input.clone()  # b,t,c,h,w

        # 设置未来要加噪的部分是未来图像和过去图像均值在latent space的差值,过去的图像也设置为残差
        # 新版本，把过去均值图像以一定比例加入到未来图像噪声中，在过去均值图像基础上预测未来
        x_past_mean = x_past_mean.unsqueeze(1).repeat(1,self.noise_image_num,1,1,1)
        # with_noise_input = with_noise_input - x_past_mean  # b,t2,c,h,w
        # without_noise_input = without_noise_input - x_past_mean[:,:without_noise_input.shape[1],...]  # b,t1,c,h,w

        # 给噪声加入过去均值图像
        noise = torch.randn_like(with_noise_input).to(input.device, dtype=torch.bfloat16)
        noise = self.past_weight*x_past_mean + (1 - self.past_weight)*noise

        # # 另一个选择，把噪声和过去均值图像和未来图像都拼接起来
        # noise = torch.randn_like(with_noise_input).to(input.device, dtype=torch.bfloat16)
        # x_past_mean = x_past_mean.unsqueeze(1)
        # without_noise_input = torch.cat([without_noise_input, x_past_mean.repeat(1,without_noise_input.shape[1],1,1,1)], dim=2)  # b,t,c,h,w




        # offset_noise_level= 0.0
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        # 加入特定噪声过程,只给要预测的图像加噪声，参考图像不加噪声
        with_noise_input_added = with_noise_input + noise * append_dims(sigmas, with_noise_input.ndim)

        # # 另一个选择，把噪声和过去均值图像和未来图像都拼接起来
        # with_noise_input_added = torch.cat([with_noise_input_added, x_past_mean.repeat(1,with_noise_input_added.shape[1],1,1,1)], dim=2)  # b,t,c,h,w

        # TODO zsj 这里是通过Unet去噪的部分，需要根据earthnet2021的特殊数据、自己新增的模块等进行修改
        # model_output为去噪后的结果,只包含未来的图像
        model_output = denoiser(
            network, without_noise_input, with_noise_input_added, sigmas, cond, **additional_model_inputs
        ).to(dtype=torch.bfloat16)  # b,t2,c,h,w
        # model_output = rearrange(model_output.to(dtype=torch.bfloat16), "b (t c) h w -> b t c h w", t=self.noise_image_num).contiguous()  # TODO ZSJ 直接暴力设置成bf16，如果accelerate.mixed_precision修改了也要跟着修改
        # print(f'model_output: {model_output[0][0][0]}')

        # # 未来预测图像等于预测的差值加上过去均值图像
        # # 新版本，模型直接预测图像而不是预测残差了
        # # bm,tm,cm,hm,wm = x_past_mean.shape
        # # x_past_mean = x_past_mean.view(bm,tm*cm,hm,wm)
        # model_output = model_output + x_past_mean  # b,t2,c,h,w 

        noise_added_output = with_noise_input_added
        w = append_dims(denoiser.w(sigmas), input.ndim)


        # print(f'model_output:{model_output}')
        
        # 在latent space里面计算损失
        loss, true_latent_rmse = self.get_latent_space_loss(model_output, latent_space_future, w)
        loss_mean, true_latent_rmse_mean = self.get_latent_space_loss(x_past_mean, latent_space_future, w)

        # loss_latent_space, true_latent_rmse, min_latent_rmse, max_latent_rmse = self.get_latent_space_loss(model_output, latent_space_future, w)
        # mean_loss_latent_space, mean_true_latent_rmse, mean_min_latent_rmse, mean_max_latent_rmse = self.get_latent_space_loss(x_past_mean, latent_space_future, w)
        # # 进行测试，看看单纯无变化是否可以生成
        # loss_latent_space = self.get_latent_space_loss(model_output, x_past_mean, w)

        # 在损失函数内部就计算好了均值
        # loss = torch.mean(loss_latent_space)

        # # 在decoder外面计算损失
        # model_output_decode = decode_first_stage(rearrange(model_output, "b t c h w -> (b t) c h w").contiguous())
        # model_output_decode = rearrange(model_output_decode, "(b t) c h w -> b t c h w", t=self.noise_image_num).contiguous()

        # model_output_decode = torch.clamp(model_output_decode, -1.0, 1.0)
        # input_future = torch.clamp(input_future, -1.0, 1.0)
        # mask = batch["mask"][:, -self.noise_image_num:, ...].repeat(1,1,input_future.shape[2],1,1)  # b,t,c,h,w  # b,t,4,h,w
        # loss_decode_space, true_decode_rmse = self.get_loss(model_output_decode, input_future, w, mask)
        # true_latent_rmse = torch.sqrt(torch.mean((model_output-latent_space_future)**2))  # 记录latent space的rmse，方便和在latent space计算损失的实验对比
        # loss = torch.mean(loss_decode_space)
        # print(f'true_decode_rmse:{true_decode_rmse.item()}')

        # 选取生成的样本和重建的未来样本
        model_output_sample = model_output[:2,0,...]  # b,c,h,w
        latent_space_future_sample = latent_space_future[:2,0,...]  # b,c,h,w
        noise_added_output_sample = noise_added_output[:2,0,...]
        with torch.no_grad():
            model_output_sample = decode_first_stage(model_output_sample)[0]  # c,h,w
            latent_space_future_sample = decode_first_stage(latent_space_future_sample)[0]  # c,h,w
            noise_added_output_sample = decode_first_stage(noise_added_output_sample)[0]  # c,h,w


        model_output_sample = torch.clamp(model_output_sample, -1.0, 1.0)
        latent_space_future_sample = torch.clamp(latent_space_future_sample, -1.0, 1.0)
        noise_added_output_sample = torch.clamp(noise_added_output_sample, -1.0, 1.0)

                
        def denormalize(array):
            return (array+1.0)/2
        model_output_sample = denormalize(model_output_sample).detach().to(dtype=torch.float).cpu().numpy()
        latent_space_future_sample = denormalize(latent_space_future_sample).detach().to(dtype=torch.float).cpu().numpy()
        noise_added_output_sample = denormalize(noise_added_output_sample).detach().to(dtype=torch.float).cpu().numpy()
        
        return loss, true_latent_rmse, true_latent_rmse_mean, model_output_sample, latent_space_future_sample, noise_added_output_sample
        # return loss, true_latent_rmse, min_latent_rmse, max_latent_rmse, model_output_sample, latent_space_future_sample, noise_added_output_sample, w, mean_true_latent_rmse, mean_min_latent_rmse, mean_max_latent_rmse


        # ---------------- 下面就是计算指标和记录样本部分
        # ####################################
        
        # # 设置mask，使得低质量的部分不计算损失
        # mask = batch["mask"][:, -self.noise_image_num:, ...]  # b,t,1,h,w

        # # 解码成最终输出
        # b_mo,t2c_mo,h_mo,w_mo = model_output.shape
        # b_if,t_if,c_if,h_if,w_if = input_future.shape
        # # input_future = input_future.view(b_mo,self.noise_image_num*c_if,h_if,w_if).contiguous()  # b,t*c,h,w
        # # model_output_decode = model_output.view(b_mo*self.noise_image_num,-1,h_mo,w_mo).contiguous()  # b*t,c,h,w
        # model_output_decode = rearrange(model_output, "b (t c) h w -> (b t) c h w", t=self.noise_image_num).contiguous()
        # with torch.no_grad():
        #     model_output_decode = decode_first_stage(model_output_decode).view(b_mo,self.noise_image_num,c_if,h_if,w_if).contiguous()  # b,t,c,h,w
        #     latent_space_future = decode_first_stage(latent_space_future).view(b_mo,self.noise_image_num,c_if,h_if,w_if).contiguous()  # b,t,c,h,w
        # mask_ndvi = mask.clone().squeeze(2)  # b,t,h,w
        # mask_origin = mask.repeat(1,1,c_if,1,1)  # b,t,c,h,w
        # # mask_origin = mask.view(b_mo,self.noise_image_num*c_if,h_if,w_if).contiguous()  # b,t*c,h,w
        # model_output_decode = torch.clamp(model_output_decode, -1.0, 1.0)
        # latent_space_future = torch.clamp(latent_space_future, -1.0, 1.0)
                
        # def denormalize(array):
        #     return (array+1.0)/2
        # model_output_decode = denormalize(model_output_decode)
        # latent_space_future = denormalize(latent_space_future)
        
        # model_output_decode_ndvi = (model_output_decode[:,:,-1,:,:] - model_output_decode[:,:,-2,:,:])/(model_output_decode[:,:,-1,:,:] + model_output_decode[:,:,-2,:,:] + 1e-8)  # b,t,h,w
        # # model_output_decode_ndvi = model_output_decode_ndvi.view(b_mo,self.noise_image_num*c_if,h_if,w_if).contiguous()  # b,t*c,h,w
        # latent_space_future_ndvi = (latent_space_future[:,:,-1,:,:] - latent_space_future[:,:,-2,:,:])/(latent_space_future[:,:,-1,:,:] + latent_space_future[:,:,-2,:,:] + 1e-8) # b,t,h,w

        # metric_rgbn = self.metric_rgbn.update(model_output_decode, input_future, mask_origin)
        # metric_ndvi = self.metric_ndvi.update(model_output_decode_ndvi, latent_space_future_ndvi, mask_ndvi)
        
        # ################################3

        # loss_rgbn = self.get_loss(model_output_decode, input_future, w, mask_origin)
        # loss_ndvi = self.get_loss(model_output_decode_ndvi, input_future_ndvi, w, mask_ndvi)

        # print(f'model_output_decode:{model_output_decode[0][0][0]}')

        # 不要space loss，放置模型过度拟合非掩码部分，导致特征通过解码器的时候出现问题
        # # 设置下采样到和latent space中遥感图像同样大小的mask，并计算latent space内的损失
        # b_o,tc_o,h_o,w_o = model_output.shape
        # mask_space = mask.view(b_o,tc_o,h_m,w_m).contiguous()
        # # # print(f'model_output:{model_output.shape}, mask_space:{mask_space.shape}')
        # mask_space = torch.nn.functional.interpolate(mask_space, scale_factor=h_o/h_m)  # b,t*c,h,w
        # future_image = future_image.view(b_o,tc_o,h_o,w_o).contiguous()

        # loss_space = self.get_loss(model_output, future_image, w, mask_space, latent_space=True)

        # loss = loss_origin + loss_space


        # ##########################################
        # # # 改变形状以计算指标
        # # #################################
        # # TODO ZSJ 不计算的时候可以注释掉，计算一下原始图像和通过Autoencoder的重建图像，计算metric时的指标如何，结果是stable diffusion能够达到的上限
        # past_origin_image = batch["imgs"][:, :-self.noise_image_num, ...].contiguous().detach()
        # past_origin_mask =  batch["mask"][:, :-self.noise_image_num, ...].contiguous().detach()  # b,t,1,h,w
        # # without_noise_input = without_noise_input + x_past_mean.view(bm,tm,cm,hm,wm)[:,:without_noise_input.shape[1],...]  # b,t1,c,h,w
        # bp,tp,cp,hp,wp = without_noise_input.shape
        # bo,to,co,ho,wo = past_origin_image.shape
        # past_origin_mask = past_origin_mask.repeat(1,1,co,1,1)
        # with torch.no_grad():
        #     without_noise_input_decode = decode_first_stage(without_noise_input.contiguous().view(bp*tp,-1,hp,wp))
        #     without_noise_input_decode = torch.clamp(without_noise_input_decode, -1.0, 1.0)
        #     without_noise_input_decode = without_noise_input_decode.view(bo,to,-1,ho,wo).contiguous()  #b,t,c,h,w
        # # 计算Autoencoder能够达到的指标上限
        # without_noise_input_decode = denormalize(without_noise_input_decode)
        # past_origin_image = denormalize(past_origin_image)
        # metric_upper_limit = self.metric_upper_limit.update(without_noise_input_decode, past_origin_image, past_origin_mask)
        # # ###################################

        # # past_origin_image1 = past_origin_image.to(dtype=torch.float32).detach().cpu().numpy()
        # # past_origin_image2 = past_origin_image.to(dtype=torch.float32).detach().cpu().numpy()
        # # print(f'max past_origin_image:{np.abs(past_origin_image1-past_origin_image2).max()}')
        # b_m,t_m,c_m,h_m,w_m = mask_origin.shape

        # without_noise_input = without_noise_input[:2,1,...].contiguous().detach()
        # with_noise_input_sample = with_noise_input[:2,1,...].contiguous().detach()  # b,c,h,w

        # with torch.no_grad():
        #     without_noise_input_decode = decode_first_stage(without_noise_input)  # 2,c,h,w
        #     ## 归到-1到1之间
        #     without_noise_input_decode = torch.clamp(without_noise_input_decode, -1.0, 1.0)
        #     without_noise_input_decode = denormalize(without_noise_input_decode)

        #     with_noise_input_sample = decode_first_stage(with_noise_input_sample)  # 2,c,h,w
        #     ## 归到-1到1之间
        #     with_noise_input_sample = torch.clamp(with_noise_input_sample, -1.0, 1.0)
        #     with_noise_input_sample = denormalize(with_noise_input_sample)

        # model_output_decode = model_output_decode.view(b_m,t_m,c_m,h_m,w_m).contiguous().detach()
        # latent_space_future = latent_space_future.view(b_m,t_m,c_m,h_m,w_m).contiguous().detach()
        # mask_origin = mask_origin.view(b_m,t_m,c_m,h_m,w_m).contiguous().detach()

        # model_output_decode_sample = model_output_decode[0][0].to(dtype=torch.float).cpu().numpy()
        # latent_space_future_sample = latent_space_future[0][0].to(dtype=torch.float).cpu().numpy()
        # mask_sample = mask_origin[0][0][0].to(dtype=torch.float).cpu().numpy()
        # without_noise_input_decode_sample = without_noise_input_decode[0].to(dtype=torch.float).cpu().numpy()
        # with_noise_input_sample = with_noise_input_sample[0].to(dtype=torch.float).cpu().numpy()

        # print(f'model_output_decode_sample_min:{model_output_decode_sample.min()}')
        # print(f'model_output_decode_sample_max:{model_output_decode_sample.max()}')
        # print(f'input_future_sample_min:{latent_space_future_sample.min()}')
        # print(f'input_future_sample_max:{latent_space_future_sample.max()}')


        # return loss, metric_rgbn, metric_ndvi, model_output_decode_sample, latent_space_future_sample, mask_sample, without_noise_input_decode_sample, with_noise_input_sample, metric_upper_limit
        # ############################################33

        # return loss, metric_rgbn, metric_ndvi, loss_rgbn, loss_ndvi, model_output_decode_sample, input_future_sample, mask_sample, without_noise_input_decode_sample

        # return loss, metric, loss_space, loss_origin, model_output_decode_sample, input_future_sample, mask_sample

    def get_loss(self, pred, target, w, mask):
        # the shape of pred, target, mask and w should be b,t,c,h,w
        assert pred.numel() == mask.numel(), 'size of pred not equal to mask'
        assert target.numel() == mask.numel(), 'size of target not equal to mask'
        
        # 仅考虑掩膜图像值为1的部分
        prediction = pred * mask
        target = target * mask
        
        # 计算均方根误差
        num_valid = torch.sum(mask.reshape(target.shape[0], -1), dim=-1)
        mse = torch.sum((w*((prediction-target)**2)).reshape(target.shape[0], -1), dim=-1)/(num_valid + 1e-8)
        true_rmse = torch.sqrt(torch.sum(((prediction-target)**2))/(torch.sum(mask) + 1e-8))
        # mse = torch.mean(mse)

        # 默认是l2, mse
        return mse, true_rmse
    
    def get_latent_space_loss(self, pred, target, w):
        # the shape of pred, target and w should be b,t,c,h,w
        # the first shape of all inputs are b
        assert pred.numel() == target.numel(), f'size of pred not equal to target: pred:{pred.numel()}, target:{target.numel()}'     

        mse = torch.mean(
            (w * (pred - target) ** 2).reshape(target.shape[0], -1), 1
            )  # b
        mse_loss = torch.mean(mse)
        true_rmse = torch.sqrt(torch.mean((pred - target) ** 2))

        angle_loss = self.latent_vector_angle_difference(pred, target)
        loss = mse_loss + angle_loss
        
        # # 计算均方根误差
        # mse = torch.mean(w*((prediction-target)**2), dim=1)  # b
        # true_mse = torch.mean(w*((prediction-target)**2), dim=1)  # b
        # min_latent_rmse = torch.sqrt(torch.min(true_mse))
        # max_latent_rmse = torch.sqrt(torch.max(true_mse))
        # # mse = torch.mean(mse)
        # true_rmse = torch.sqrt(torch.mean((prediction-target)**2))
        # # mean_square = (prediction-target)**2
        # # if torch.max(w)>1000:
        # #     print(mean_square)
        # #     print(w)
        # #     print(torch.mean(w*((prediction-target)**2), dim=1))

        # 默认是l2, mse
        # return mse, true_rmse, min_latent_rmse, max_latent_rmse
        return loss, true_rmse


    def latent_vector_angle_difference(self, pred, target):
        # 输入图像形状应该为b,t,c,h,w
        pred_flat = rearrange(pred, "b t c h w -> (b t h w) c").contiguous()
        target_flat = rearrange(target, "b t c h w -> (b t h w) c").contiguous()

        theta = torch.nn.functional.cosine_similarity(pred_flat, target_flat)  # 取值范围为-1到1
        theta = 1 - theta  # 取值范围为0到2
        theta = torch.mean(theta)
        
        return theta

