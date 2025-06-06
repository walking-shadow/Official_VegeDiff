import os
import torch
import argparse
import datetime
import time
import torchvision
import wandb
import logging
import math
import shutil
import accelerate
import torch
import torch.utils.checkpoint
import transformers
import diffusers
import imageio
import numpy as np
import torch.nn.functional as F
from PIL import Image

from torch.cuda import amp
from omegaconf import OmegaConf
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from safetensors import safe_open
from sgm.util import (
    instantiate_from_config,
    default,
    get_obj_from_str,
)
from sgm.modules.diffusionmodules.metric import EarthnetX_Metric
from einops import rearrange



logger = get_logger(__name__, log_level="INFO")

# For Omegaconf Tuple
def resolve_tuple(*args):
    return tuple(args)
OmegaConf.register_new_resolver("tuple", resolve_tuple)

def parse_args():
    parser = argparse.ArgumentParser(description="Argument.")
    parser.add_argument(
        "--project_name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="if setting, the logdir will be like: project_name",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="workdir",
        help="workdir",
    )
    parser.add_argument( # if resume, you change it none. i will load from the resumedir
        "--cfgdir",
        nargs="*",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

# Prepare log
# TODO: Check this
def log_videos_local_and_wandb(
    batch, 
    model,
    conditioner,
    first_stage_model,
    sampler,
    denoiser,
    input_key,
    scale_factor,
    noise_image_num,
    metrics_rgbn,
    metrics_ndvi,
    metrics_arvi,
    # upper_limit_rgbn,
    # upper_limit_ndvi,
    # mean_metric_rgbn,
    # mean_metric_ndvi,
    # decode_mean_metric_rgbn,
    # decode_mean_metric_ndvi,
    past_weigth,
):
    device = batch[input_key].device
    weight_dtype = batch[input_key].dtype
    def encode_first_stage(first_stage_model, input_tensor, scale_factor):
        with torch.no_grad():
            z = first_stage_model.encode(input_tensor)
            if isinstance(z, dict):
                z=z.latent_dist.sample()
            z = scale_factor * z
        return z

    def decode_first_stage(first_stage_model, input_tensor, scale_factor):
        with torch.no_grad():
            z = 1.0 / scale_factor * input_tensor
            out = first_stage_model.decode(z)
            if isinstance(out, dict):
                out=out.sample
        return out
    
    def validate_and_log(
        batch,
        sample = True,
        ucg_keys = None,
        noise_image_num = noise_image_num,
        past_weigth = past_weigth,
    ):
        def sample_func(
            cond,
            uc = None,
            batch_size = 16,
            shape = None,
            past_images = None,
            past_mean = None,
            noise_image_num = None,
            past_weigth = past_weigth,
        ):
            # 这里控制randn的时间只有20，past_images是前面的10帧图像，控制形状为B,T,C,H,W。同时注意在后面把randn转成B,T,C,H,W的形状
            randn = torch.randn(batch_size, *shape).to(device, dtype=weight_dtype)

            # 把噪声和过去均值图像加权混合
            past_mean = past_mean.repeat(1,noise_image_num,1,1,1)  # b,t,c,h,w
            past_mean = rearrange(past_mean, "b t c h w -> b (t c) h w").contiguous()  # b,t*c,h,w
            randn = past_weigth*past_mean + (1-past_weigth)*randn  # b,t*c,h,w

            # # 另一个选择，把噪声和过去均值图像和未来图像都拼接起来
            # randn = rearrange(randn, "b (t c) h w -> b t c h w", t=noise_image_num).contiguous()  # b,t,c,h,w
            # randn = torch.cat([randn, past_mean.repeat(1,randn.shape[1],1,1,1)], dim=2)
            # randn = rearrange(randn, "b t c h w -> b (t c) h w").contiguous()  # b,t*c,h,w
            # past_images = torch.cat([past_images, past_mean.repeat(1,past_images.shape[1],1,1,1)], dim=2)


            denoiser_func = lambda input, sigma, c, past_images: denoiser(
                model, past_images, input, sigma, c
            )
            samples = sampler(denoiser_func, randn, cond, uc=uc, past_images=past_images, noise_image_num=noise_image_num,
                              past_mean=past_mean, past_weigth=past_weigth)
            # 输出形状为b,t*c,h,w

            # # 另一个选择，把噪声和过去均值图像和未来图像都拼接起来
            # samples = rearrange(samples, "b (t c) h w -> b t c h w", t=noise_image_num).contiguous()
            # samples = samples[:,:,:4,...]
            # samples = rearrange(samples, "b t c h w -> b (t c) h w").contiguous()

            return samples
        
        # conditioner_input_keys = [e.input_key for e in conditioner.module.embedders]
        conditioner_input_keys = [e.input_key for e in conditioner.embedders]

        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        
        # get input
        x = batch[input_key]

        # get c and uc, notion here doesn't use the uc(uc=c)
        # c和uc都是condition变量，即对batch运行conditioner forward函数的结果
        # 它们都是一个字典，其中crossattn键对应一个列表，包括4个不同大小的condition(气象变量和环境变量),形状为bhw,t,c, vector键对应图像大小信息,形状为b,c
        
        # c, uc = conditioner.module.get_unconditional_conditioning(
        #     batch,
        #     force_uc_zero_embeddings=ucg_keys
        #     if len(conditioner.module.embedders) > 0
        #     else [],
        # )

        c, uc = conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(conditioner.embedders) > 0
            else [],
        )

        # c = conditioner(batch)
        # import copy
        # uc = copy.deepcopy(c)

        # log reconstructions
        log = dict()
        N = x.shape[0]  # batchsize
        x = x.to(device, dtype=weight_dtype)[:N]
        
        log["inputs_sample"] = x[0,-1,...]  # 取最后一张图像记录

        x_= x
        B, T, C, H, W = x_.shape
        assert B==N
        x_ = x_.view(B*T, C, H, W)
        z = encode_first_stage(first_stage_model=first_stage_model, input_tensor=x_, scale_factor=scale_factor).to(device, dtype=weight_dtype)
        _, C_, H_, W_ = z.shape
        log["reconstructions"] = torch.clamp(decode_first_stage(first_stage_model=first_stage_model, input_tensor=z, scale_factor=scale_factor).view(B, T, C, H, W), -1.0, 1.0)[:,-noise_image_num:,...]  # 只取出后面预测的重建部分
        log["reconstructions_sample"] = log["reconstructions"][0,-1,...]  # 取最后一张图像记录
        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k].to(device, dtype=weight_dtype), (c, uc))
            elif isinstance(c[k], list):
                for i in range(len(c[k])):
                    c[k][i] = c[k][i].to(device, dtype=weight_dtype)
                    uc[k][i] = uc[k][i].to(device, dtype=weight_dtype)
        
        # sample
        past_images = z.view(B, -1, C_, H_, W_)[:,:-noise_image_num,...]
        future_images = z.view(B, -1, C_, H_, W_)[:,-noise_image_num:,...]  # b,t,c,h,w
        # 得到过去均值图像和过去残差图像
        past_mean = torch.mean(past_images, dim=1, keepdim=True)  # b,1,c,h,w
        # log["past_mean"] = past_mean.repeat(1,noise_image_num,1,1,1)  # 记录形状为b,t,c,h,w的baseline结果

        # past_res = past_images - past_mean.repeat(1,past_images.shape[1],1,1,1)
        # past_images = past_res
        if sample:
            # sample 形状为b,t*c,h,w
            samples = sample_func(c, shape=torch.Size([noise_image_num*C_,H_,W_]), uc=uc, batch_size=N, past_images=past_images, past_mean=past_mean, noise_image_num=noise_image_num)
            #TODO Check denoiser dtype

            # 得到的samples是未来图像的残差图像，加上过去均值图像得到未来预测图像
            # 新版本，直接输出图像，不预测残差
            samples = rearrange(samples, "b (t c) h w -> (b t) c h w", t=noise_image_num).contiguous().to(device, dtype=weight_dtype)  # b,t2,c,h,w
            sample_latent = rearrange(samples, "(b t) c h w -> b t c h w", t=noise_image_num).contiguous()
            # samples = rearrange(samples, "b (t c) h w -> b t c h w", t=noise_image_num).contiguous()  # b,t2,c,h,w
            # samples = samples + past_mean.repeat(1,noise_image_num, 1,1,1)
            # samples = rearrange(samples, "b t c h w -> (b t) c h w").contiguous().to(device, dtype=weight_dtype)  # b*t2,c,h,w
            # # samples=samples.view(B*noise_image_num, C_, H_, W_).to(dtype=weight_dtype)

            samples = decode_first_stage(first_stage_model=first_stage_model, input_tensor=samples, scale_factor=scale_factor).view(B, noise_image_num, C, H, W)
            samples = torch.clamp(samples, -1.0, 1.0)
            log["last_samples"] = samples[0,-1,...]  # 取最后一张图像记录
            log["first_samples"] = samples[0,0,...]  # 取最后一张图像记录

        past_mean_latent_rmse = torch.sqrt(torch.mean((past_mean.repeat(1,noise_image_num,1,1,1) - future_images) ** 2))
        predict_latent_rmse = torch.sqrt(torch.mean((sample_latent - future_images) ** 2))
        print(f'past_mean_latent_rmse:{past_mean_latent_rmse}, predict_latent_rmse:{predict_latent_rmse}')
        return log, samples
    
    model.eval()
    conditioner.eval()
    with torch.no_grad():
        # validate_log是个字典里面有inputs,reconstructions和samples这三个键，分别对应全时序输入、全时序重建、生成的未来序列（时间数量为noise_image_num）
        validate_log, samples = validate_and_log(batch)
        pred = samples  # b,t,c,h,w
        target = batch['imgs'][:,-noise_image_num:,...]
        mask = batch['mask'][:,-noise_image_num:,...]
        b,t,c,h,w = pred.shape

        fromer_rmse = torch.sqrt(torch.mean((pred[:,:10,...] - target[:,:10,...])**2))
        latter_rmse = torch.sqrt(torch.mean((pred[:,10:,...] - target[:,10:,...])**2))
        print(f'fromer_rmse:{fromer_rmse}, latter_rmse:{latter_rmse}')

        mask_ndvi = mask.clone().squeeze(2)  # b,t,1,h,w
        mask_origin = mask.repeat(1,1,c,1,1)  # b,t,c,h,w
        # mask_origin = mask.view(b_mo,self.noise_image_num*c_if,h_if,w_if).contiguous()  # b,t*c,h,w
        pred = torch.clamp(pred, -1.0, 1.0)
                
        def denormalize(array):
            return (array+1.0)/2
        pred = denormalize(pred)
        target = denormalize(target)
        
        pred_blue, pred_green, pred_red, pred_nir = pred[:,:,0,:,:], pred[:,:,1,:,:], pred[:,:,2,:,:], pred[:,:,3,:,:]
        target_blue, target_green, target_red, target_nir = target[:,:,0,:,:], target[:,:,1,:,:], target[:,:,2,:,:], target[:,:,3,:,:]
        pred_ndvi = (pred_nir - pred_red)/(pred_nir + pred_red + 1e-8)  # b,t,h,w
        # model_output_decode_ndvi = model_output_decode_ndvi.view(b_mo,self.noise_image_num*c_if,h_if,w_if).contiguous()  # b,t*c,h,w
        target_ndvi = (target_nir - target_red)/(target_nir + target_red + 1e-8) # b,t,h,w

        pred_arvi = (pred_nir - (2 * pred_red) + pred_blue) / (pred_nir + (2 * pred_red) + pred_blue + 1e-8)
        target_arvi = (target_nir - (2 * target_red) + target_blue) / (target_nir + (2 * target_red) + target_blue + 1e-8)

        btach_metric_rgbn = metrics_rgbn.update(pred, target, mask_origin)
        btach_metric_ndvi = metrics_ndvi.update(pred_ndvi, target_ndvi, mask_ndvi)
        btach_metric_arvi = metrics_arvi.update(pred_arvi, target_arvi, mask_ndvi)

        # reconstructions = validate_log['reconstructions']
        # reconstructions = denormalize(reconstructions)
        # reconstructions_ndvi = (reconstructions[:,:,-1,:,:] - reconstructions[:,:,-2,:,:])/(reconstructions[:,:,-1,:,:] + reconstructions[:,:,-2,:,:] + 1e-8)  # b,t,h,w

        # batch_upper_limit_rgbn = upper_limit_rgbn.update(reconstructions, target, mask_origin)
        # batch_upper_limit_ndvi = upper_limit_ndvi.update(reconstructions_ndvi, target_ndvi, mask_ndvi)

        # past_mean = batch['imgs'][:,:-noise_image_num,...]
        # past_mean = torch.mean(past_mean, dim=1, keepdim=True)  # b,1,c,h,w

        # past_mean_encode = past_mean.squeeze(1)  # b,c,h,w
        # past_mean_encode = encode_first_stage(first_stage_model=first_stage_model, input_tensor=past_mean_encode, scale_factor=scale_factor).to(device, dtype=weight_dtype)  # b,c,h,w
        # past_mean_decode = torch.clamp(decode_first_stage(first_stage_model=first_stage_model, input_tensor=past_mean_encode, scale_factor=scale_factor), -1.0, 1.0)  # b,c,h,w
        
        # past_mean = past_mean.repeat(1,noise_image_num,1,1,1)  # b,t,c,h,w
        # past_mean = denormalize(past_mean)
        # past_mean_ndvi = (past_mean[:,:,-1,:,:] - past_mean[:,:,-2,:,:])/(past_mean[:,:,-1,:,:] + past_mean[:,:,-2,:,:] + 1e-8)  # b,t,h,w

        # past_mean_decode = past_mean_decode.unsqueeze(1).repeat(1,noise_image_num,1,1,1)  # b,t,c,h,w
        # past_mean_decode = denormalize(past_mean_decode)
        # past_mean_decode_ndvi = (past_mean_decode[:,:,-1,:,:] - past_mean_decode[:,:,-2,:,:])/(past_mean_decode[:,:,-1,:,:] + past_mean_decode[:,:,-2,:,:] + 1e-8)  # b,t,h,w

        # # print(f'info: {past_mean.shape}, {past_mean_ndvi.shape}, {mask_origin.shape}, {mask_ndvi.shape}')
        # batch_mean_rgbn = mean_metric_rgbn.update(past_mean, target, mask_origin)
        # batch_mean_ndvi = mean_metric_ndvi.update(past_mean_ndvi, target_ndvi, mask_ndvi)

        # batch_decode_mean_rgbn = decode_mean_metric_rgbn.update(past_mean_decode, target, mask_origin)
        # batch_decode_mean_ndvi = decode_mean_metric_ndvi.update(past_mean_decode_ndvi, target_ndvi, mask_ndvi)


    model.train()
    conditioner.train()
    # return btach_metric_rgbn, btach_metric_ndvi, batch_upper_limit_rgbn, batch_upper_limit_ndvi, batch_mean_rgbn, batch_mean_ndvi, batch_decode_mean_rgbn, batch_decode_mean_ndvi, validate_log
    return btach_metric_rgbn, btach_metric_ndvi, btach_metric_arvi, validate_log


def main():
    args = parse_args()
    
    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = None
    workdir = None
    workdirnow = None
    cfgdir = None
    ckptdir = None
    logging_dir = None
    videodir = None
    
    if args.project_name:
        project_name = args.project_name
        if os.path.exists(os.path.join(args.workdir, project_name)): #open resume
            workdir=os.path.join(args.workdir, project_name)
        else: # new a workdir
            workdir = os.path.join(args.workdir, project_name)
            # if accelerator.is_main_process:
            os.makedirs(workdir, exist_ok=True)
        workdirnow = workdir

        cfgdir = os.path.join(workdirnow, "configs")
        ckptdir = os.path.join(workdirnow, "checkpoints")
        logging_dir = os.path.join(workdirnow, "logs")
        videodir = os.path.join(workdirnow, "videos")

        # if accelerator.is_main_process:
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(ckptdir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(videodir, exist_ok=True)
    if args.cfgdir:
        load_cfgdir = args.cfgdir
    
    # setup config
    configs_list = load_cfgdir # read config from a config dir
    configs = [OmegaConf.load(cfg) for cfg in configs_list]
    config = OmegaConf.merge(*configs)
    config_accelerate = config.accelerate
    config_diffusion = config.diffusion
    config_data = config.data
    
    accelerator_project_config = ProjectConfiguration(project_dir=workdirnow, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=config_accelerate.gradient_accumulation_steps,
        mixed_precision=config_accelerate.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        File_handler = logging.FileHandler(os.path.join(logging_dir, project_name+"_"+datenow+".log"), encoding="utf-8")
        File_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        File_handler.setLevel(logging.INFO)
        logger.logger.addHandler(File_handler)
        
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        learning_rate = (
            config_accelerate.learning_rate * 
            config_accelerate.gradient_accumulation_steps * 
            config_data.params.train_batch_size * 
            accelerator.num_processes / config_accelerate.learning_rate_base_batch_size
        )
        print('adjust learning rate as scale_lr is True')
    else:
        learning_rate = config_accelerate.learning_rate


    # Setup model
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config_accelerate.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config_accelerate.mixed_precision = accelerator.mixed_precision
    
    # 初始化VideoUNetModel
    model = instantiate_from_config(config_diffusion.network_config)
    # 初始化VideoWrapper作为后续的预测噪声模型
    model = get_obj_from_str(config_diffusion.network_wrapper)(
        model, compile_model=config_diffusion.compile_model
    )
    for name, param in model.named_parameters():
        param.requires_grad = True
        logger.info(f"model trainable: {name}")
    
    # def make_print_grad(name):
    #     def print_grad(grad):
    #         print(name, torch.mean(grad))
    #     return print_grad

    # for name, param in model.named_parameters():
    #     param.register_hook(make_print_grad(name))




    # # Load SDXL
    # # ZSJ TODO 不需要stable diffusion的预训练模型了，改成只观察它们的训练参数有哪些
    # def init_from_ckpt( # init SD-xL from safetensors, TODO:Fix bugs about conditioner
    #     checkpoint_dir, ignore_keys=None, verbose=True,
    # ) -> None: 
    #     model_state_dict=safe_open(checkpoint_dir, framework="pt", device="cpu")
    #     model_new_ckpt=dict()
    #     for i in model_state_dict.keys():
    #         if "model.diffusion_model." in i:
    #             model_new_ckpt[i.replace("model.diffusion_model.", "diffusion_model.")] = model_state_dict.get_tensor(i)
    #     sd=model_new_ckpt
    #     keys = list(sd.keys())
    #     for k in keys:
    #         if ignore_keys:
    #             for ik in ignore_keys:
    #                 if re.match(ik, k):
    #                     logger.info("Deleting key {} from state_dict.".format(k))
    #                     del sd[k]
    #     missing, unexpected = model.load_state_dict(sd, strict=False)
    #     if verbose:
    #         logger.info(
    #             f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
    #         )
    #         if len(missing) > 0:
    #             logger.info(f"Missing Keys: {missing}")
    #         if len(unexpected) > 0:
    #             logger.info(f"Unexpected Keys: {unexpected}")
    #     if verbose:
    #         logger.info("")
    #         logger.info("Unfrozen module", config_diffusion.trainable_modules)
    #         logger.info("Unfrozen parameters:")
    #     for name, param in model.named_parameters():
    #         param.requires_grad = False
    #         # if verbose:
    #         #     logger.info(f"****{name}")
    #         for trainable_module_name in config_diffusion.trainable_modules:
    #             if trainable_module_name in name:
    #                 param.requires_grad = True
    #                 if verbose:
    #                     logger.info(f"model trainable: {name}")
    #                 break
    #     if verbose:
    #         logger.info("")
    # init_from_ckpt(checkpoint_dir=config_diffusion.pretrained_image_ckpt)
    
    
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        conditioner = instantiate_from_config(
            config_diffusion.conditioner_config
        )
        # TODO ZSJ把这里换成自己训练的autoencoder,并加载预训练参数
        ae_config = OmegaConf.load(config_diffusion.first_stage_config.config_path)
        first_stage_model = instantiate_from_config(ae_config.model)
        model_ckpt = torch.load(config_diffusion.first_stage_config.pretrained_path)['state_dict']
        for key in list(model_ckpt.keys()):
            if 'loss.perceptual_loss' in key:
                del(model_ckpt[key])
        first_stage_model.load_state_dict(model_ckpt)

        # first_stage_model = AutoencoderKL.from_pretrained(
        #     config_diffusion.first_stage_config.from_pretrained
        # )

    # Put model to gpu and cast to correct dtype
    conditioner = conditioner.to(accelerator.device, dtype=weight_dtype)
    first_stage_model = first_stage_model.to(accelerator.device, dtype=weight_dtype)
    for param in first_stage_model.parameters():
        param.requires_grad = False
    # # TODO ZSJ 设置conditioner里面的resnet的参数可以训练（默认好像就可以训练）
    for name, param in conditioner.named_parameters():
        param.requires_grad = True
        logger.info(f"conditioner trainable: {name}")

    # for embedder in conditioner.embedders:
    #     if embedder.is_trainable:
    #         for name, param in embedder.named_parameters():
    #             param.requires_grad = True
    #             logger.info(f"conditioner trainable: {name}")
    
    # # 为验证的部分准备的,因为后面conditioner通过accelerator之后，它的类就变了
    # conditioner_input_keys = [e.input_key for e in conditioner.embedders]

    # for param in conditioner.parameters():
    #     param.requires_grad = False

    # Setup Dataloader
    get_train_dataloader = instantiate_from_config(config_data)
    train_len = get_train_dataloader.train_len()
    train_dataloader = get_train_dataloader.train_dataloader()
    test_dataloader = get_train_dataloader.test_dataloader()
    # test_dataloader = get_train_dataloader.train_dataloader()


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config_accelerate.gradient_accumulation_steps)
    if config_accelerate.max_train_steps is None: # not assign max steps
        config_accelerate.max_train_steps = config_accelerate.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Setup optimizer and lr_scheduler
    params = list(model.parameters())
    for embedder in conditioner.embedders:
        params = params + list(embedder.parameters())
    optimizer_config = default(
        config_accelerate.optimizer, {"target": "torch.optim.AdamW"}
    )
    def instantiate_optimizer_from_config(params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )
    optimizer = instantiate_optimizer_from_config(params, learning_rate, optimizer_config)
    #TODO
    lr_scheduler = get_scheduler(
        config_accelerate.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config_accelerate.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config_accelerate.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare Accelerate
    model, conditioner, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, conditioner, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    model = model.to(dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config_accelerate.gradient_accumulation_steps)
    if overrode_max_train_steps: # 
        config_accelerate.max_train_steps = config_accelerate.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config_accelerate.num_train_epochs = math.ceil(config_accelerate.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    #TODO
    if accelerator.is_main_process:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), workdirnow)
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=dict(config), 
            init_kwargs={"wandb": {"group": datenow, 'mode': 'offline'}}
        )

    # Train!
    total_batch_size = config_data.params.train_batch_size * accelerator.num_processes * config_accelerate.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_len}")
    logger.info(f"  Num Epochs = {config_accelerate.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config_data.params.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Learning rate = {learning_rate}")
    logger.info(f"  Gradient Accumulation steps = {config_accelerate.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config_accelerate.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # # normal read w/o safety check
        # if args.resume_from_checkpoint != "latest":
        #     path = os.path.basename(args.resume_from_checkpoint)
        # else:
        #     # Get the most recent checkpoint
        #     dirs = os.listdir(ckptdir)
        #     dirs = [d for d in dirs if d.startswith("checkpoint")]
        #     dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        #     path = dirs[-1] if len(dirs) > 0 else None

        # if path is None:
        #     logger.info(
        #         f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        #     )
        #     args.resume_from_checkpoint = None
        # else:
        #     logger.info(f"Resuming from checkpoint {path}")
        #     accelerator.load_state(os.path.join(ckptdir, path))
        #     global_step = int(path.split("-")[1]) # gs not calculate the gradient_accumulation_steps
        #     resume_global_step = global_step * config_accelerate.gradient_accumulation_steps
        #     first_epoch = global_step // num_update_steps_per_epoch
        #     resume_step = resume_global_step % (num_update_steps_per_epoch * config_accelerate.gradient_accumulation_steps)

        # normal read with safety check
        resume_step=0
        resume_global_step=0
        error_times=0
        while(True):
            if error_times >= 100:
                raise
            try:
                if args.resume_from_checkpoint != "latest":
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = os.listdir(ckptdir)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None

                if path is None:
                    logger.info(
                        f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                    )
                    args.resume_from_checkpoint = None
                else:
                    # global_step, config_accelerate.max_train_steps: no *acc
                    # resume_global_step, resume_step: has *acc
                    logger.info(f"Resuming from checkpoint {path}")
                    accelerator.load_state(os.path.join(ckptdir, path))
                    global_step = int(path.split("-")[1]) # gs not calculate the gradient_accumulation_steps
                    resume_global_step = global_step * config_accelerate.gradient_accumulation_steps
                    first_epoch = global_step // num_update_steps_per_epoch
                    resume_step = resume_global_step % (num_update_steps_per_epoch * config_accelerate.gradient_accumulation_steps)
                break
            except (RuntimeError, Exception) as err:
                error_times+=1
                if accelerator.is_local_main_process:
                    logger.warning(err)
                    logger.warning(f"Failed to resume from checkpoint {path}, causing {err}")
                    shutil.rmtree(os.path.join(ckptdir, path))
                    load_flag=True
                else:
                    # import time
                    time.sleep(2)
    
    # save config
    OmegaConf.save(config=config, f=os.path.join(cfgdir, "config.yaml"))
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config_accelerate.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Optim Steps")

    # Prepare diffusion
    loss_fn = instantiate_from_config(config_diffusion.loss_fn_config)
    denoiser = instantiate_from_config(config_diffusion.denoiser_config)
    sampler = instantiate_from_config(config_diffusion.sampler_config)
    
    loss_fn = loss_fn.to(accelerator.device, dtype=weight_dtype)
    denoiser = denoiser.to(accelerator.device, dtype=weight_dtype)
    # sampler.to(accelerator.device, dtype=weight_dtype)
    
    def encode_first_stage(input_tensor):
        with torch.no_grad():
            z = first_stage_model.encode(input_tensor)
            if isinstance(z, dict):
                z=z.latent_dist.sample()
            z = config_diffusion.scale_factor * z
        return z

    def decode_first_stage(input_tensor):
        # 我在最后的输出部分计算损失，需要这里记录一些值
        # with torch.no_grad():
        z = 1.0 / config_diffusion.scale_factor * input_tensor
        out = first_stage_model.decode(z)
        if isinstance(out, dict):
            out=out.sample
        return out

    # Training Loop
    # try:
    print('Start training loop')
    for epoch in range(first_epoch, config_accelerate.num_train_epochs):
        model.train()
        conditioner.train()
        train_loss = 0.0
        # train_loss_space = 0.0
        # train_loss_rgbn = 0.0
        # train_loss_ndvi = 0.0
        if args.resume_from_checkpoint and epoch == first_epoch and resume_step > 0:
            train_dataloader = skip_first_batches(train_dataloader, resume_step)
            progress_bar.update(resume_step//config_accelerate.gradient_accumulation_steps)
            resume_step = 0
        start_read_time = time.time()
        
        # Start Train! 
        # TODO zsj 由于dataloader修改了，把训练时数据格式也对应修改一下，同时加入val_dataloader的部分
        for step, batch in enumerate(train_dataloader):
            end_read_time = time.time()
            # # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % config_accelerate.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            # if accelerator.is_main_process:
            #     with open(os.path.join(logging_dir, "dataloader.txt"), "a+") as f:
            #         print("step: "+str(step)+" idx: "+str(batch['idx'][0])+"\n")
            #         f.write("step: "+str(step)+" idx: "+str(batch['idx'][0])+"\n")

            for batch_key in batch.keys():
                if torch.is_tensor(batch[batch_key]):
                    batch[batch_key] = batch[batch_key].to(dtype=weight_dtype)
                # elif batch_key == 'condition_image':
                #     for i in range(len(batch[batch_key])):
                #         batch[batch_key][i] = batch[batch_key][i].to(dtype=weight_dtype)

            x = batch[config_diffusion.input_key]  # 遥感图像
            with accelerator.accumulate(model):
                start_encode_time = time.time()
                # video inference
                B, T, C, H, W = x.shape
                x_future = x[:,-config_diffusion.network_config.params.noise_image_num:,...]  # 获取

                x_past = x[:,:-config_diffusion.network_config.params.noise_image_num,...]
                x_past_mean = torch.mean(x_past, dim=1).contiguous() # b,c,h,w
                x_past_mean_sample = x_past_mean[0] # c,h,w

                x = x.view(B * T, C, H, W).contiguous()
                x = encode_first_stage(x)
                x_past_mean = encode_first_stage(x_past_mean)

                # print(f'encode_x:{x[0][0][0]}')
                x = x.to(dtype=weight_dtype)
                x_past_mean = x_past_mean.to(dtype=weight_dtype)
                _, C, H, W = x.shape
                # x = x.view(B, T, C, H, W).contiguous().view(B, T * C, H, W).contiguous()
                # 修改x形状为（b,t,c,h,w)
                x = x.view(B, T, C, H, W).contiguous()
                assert x.dtype==weight_dtype
                start_calc_loss_time = time.time()
                # calc loss
                # TODO zsj 模型的计算都在这里，根据数据格式对应修改一下
                # x是加噪和去噪的图像，batch是dataloader的原始数据，model是videounetmodel
                # x形状为（b,t,c,h,w)
                # loss, metric, loss_space, loss_rgbn, prediction_sample, future_sample, mask_sample = loss_fn(model, denoiser, conditioner, x, batch, x_future, decode_first_stage)  # b
                # loss, metric_rgbn, metric_ndvi, loss_rgbn, loss_ndvi, prediction_sample, future_sample, mask_sample, without_noise_input_decode_sample = \
                # loss, metric_rgbn, metric_ndvi, prediction_sample, future_sample, mask_sample, without_noise_input_decode_sample, with_noise_input_sample, metric_upper_limit = \
                # loss, true_latent_rmse, min_latent_rmse, max_latent_rmse, prediction_sample, future_sample, noise_added_output_sample, w, mean_true_latent_rmse, mean_min_latent_rmse, mean_max_latent_rmse = \
                # loss, true_latent_rmse, prediction_sample, future_sample, noise_added_output_sample = \
                loss, true_latent_rmse, true_latent_rmse_mean, prediction_sample, future_sample, noise_added_output_sample = \
                loss_fn(model, denoiser, conditioner, x, batch, x_future, x_past_mean, decode_first_stage)  # b
                # loss_space = loss_space.mean()
                # loss_rgbn = loss_rgbn.mean()
                # loss = loss.mean()
                # TODO ZSJ 添加计算metrics的代码
                def denormalize(array):
                    return (array+1.0)/2
                
                x_past_mean_sample = denormalize(x_past_mean_sample).detach().to(dtype=torch.float).cpu().numpy()
                x_past_mean_sample_rgb, x_past_mean_sample_nir = \
                (x_past_mean_sample[:3,:,:]*255).astype(np.uint8), (x_past_mean_sample[3,:,:]*255).astype(np.uint8)

                prediction_sample = prediction_sample
                prediction_sample_rgb, prediction_sample_nir = \
                (prediction_sample[:3,:,:]*255).astype(np.uint8), (prediction_sample[3,:,:]*255).astype(np.uint8)

                future_sample = future_sample
                future_sample_rgb, future_sample_nir = \
                (future_sample[:3,:,:]*255).astype(np.uint8), (future_sample[3,:,:]*255).astype(np.uint8)

                noise_added_output_sample = noise_added_output_sample
                noise_added_output_sample_rgb, noise_added_output_sample_nir = \
                (noise_added_output_sample[:3,:,:]*255).astype(np.uint8), (noise_added_output_sample[3,:,:]*255).astype(np.uint8)

                # mask_sample = (mask_sample*255).astype(np.uint8)

                # without_noise_input_decode_sample = without_noise_input_decode_sample
                # without_noise_input_decode_sample_rgb, without_noise_input_decode_sample_nir = \
                # (without_noise_input_decode_sample[:3,:,:]*255).astype(np.uint8), (without_noise_input_decode_sample[3,:,:]*255).astype(np.uint8)

                # with_noise_input_sample = with_noise_input_sample
                # with_noise_input_sample_rgb, with_noise_input_sample_nir = \
                # (with_noise_input_sample[:3,:,:]*255).astype(np.uint8), (with_noise_input_sample[3,:,:]*255).astype(np.uint8)

                x_past_mean_sample_rgb = Image.fromarray(x_past_mean_sample_rgb.transpose(1, 2, 0))
                x_past_mean_sample_nir = Image.fromarray(x_past_mean_sample_nir)
                prediction_sample_rgb = Image.fromarray(prediction_sample_rgb.transpose(1, 2, 0))
                prediction_sample_nir = Image.fromarray(prediction_sample_nir)
                future_sample_rgb = Image.fromarray(future_sample_rgb.transpose(1, 2, 0))
                future_sample_nir = Image.fromarray(future_sample_nir)
                noise_added_output_sample_rgb = Image.fromarray(noise_added_output_sample_rgb.transpose(1, 2, 0))
                noise_added_output_sample_nir = Image.fromarray(noise_added_output_sample_nir)
                # mask_sample = Image.fromarray(mask_sample)
                # without_noise_input_decode_sample_rgb = Image.fromarray(without_noise_input_decode_sample_rgb.transpose(1, 2, 0))
                # without_noise_input_decode_sample_nir = Image.fromarray(without_noise_input_decode_sample_nir)
                # with_noise_input_sample_rgb = Image.fromarray(with_noise_input_sample_rgb.transpose(1, 2, 0))
                # with_noise_input_sample_nir = Image.fromarray(with_noise_input_sample_nir)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config_data.params.train_batch_size)).mean()
                train_loss += avg_loss.item() / config_accelerate.gradient_accumulation_steps
                # avg_loss_space = accelerator.gather(loss_space.repeat(config_data.params.train_batch_size)).mean()
                # train_loss_space += avg_loss_space.item() / config_accelerate.gradient_accumulation_steps
                # avg_loss_rgbn = accelerator.gather(loss_rgbn.repeat(config_data.params.train_batch_size)).mean()
                # train_loss_rgbn += avg_loss_rgbn.item() / config_accelerate.gradient_accumulation_steps
                # avg_loss_ndvi = accelerator.gather(loss_ndvi.repeat(config_data.params.train_batch_size)).mean()
                # train_loss_ndvi += avg_loss_ndvi.item() / config_accelerate.gradient_accumulation_steps
                # Backpropagate
                start_bp_loss_time = time.time()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config_accelerate.max_grad_norm)
                    accelerator.clip_grad_norm_(conditioner.parameters(), config_accelerate.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            end_bp_loss_time = time.time()
            # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
            if accelerator.sync_gradients: 
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                 "true_latent_rmse": true_latent_rmse.item(),
                                 "true_latent_rmse_mean": true_latent_rmse_mean.item(),
                                #  "min_latent_rmse": min_latent_rmse.item(),
                                #  "max_latent_rmse": max_latent_rmse.item(),
                                #  "loss_space":train_loss_space,
                                #  "loss_rgbn":train_loss_rgbn,
                                #  "loss_ndvi":train_loss_ndvi,
                                 }, step=global_step)
                if global_step % config_diffusion.img_log_step_inteval == 0:
                    accelerator.log({
                                    "x_past_mean_sample_rgb":wandb.Image(x_past_mean_sample_rgb),
                                    "x_past_mean_sample_nir":wandb.Image(x_past_mean_sample_nir),
                                    "prediction_sample_rgb":wandb.Image(prediction_sample_rgb),
                                    "prediction_sample_nir":wandb.Image(prediction_sample_nir),
                                    "future_sample_rgb":wandb.Image(future_sample_rgb),
                                    "future_sample_nir":wandb.Image(future_sample_nir),
                                    "noise_added_output_sample_rgb":wandb.Image(noise_added_output_sample_rgb),
                                    "noise_added_output_sample_nir":wandb.Image(noise_added_output_sample_nir),
                                    # "mask_sample": wandb.Image(mask_sample),
                                    # "without_noise_input_decode_sample_rgb": wandb.Image(without_noise_input_decode_sample_rgb),
                                    # "without_noise_input_decode_sample_nir": wandb.Image(without_noise_input_decode_sample_nir),
                                    # "with_noise_input_sample_rgb": wandb.Image(with_noise_input_sample_rgb),
                                    # "with_noise_input_sample_nir": wandb.Image(with_noise_input_sample_nir),
                                    }, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                # # 记录结果指标
                # for k,v in metric_rgbn.items():
                #     accelerator.log({f'{k}_rgbn': v}, step=global_step)
                # for k,v in metric_ndvi.items():
                #     accelerator.log({f'{k}_ndvi': v}, step=global_step)
                train_loss = 0.0
                # train_loss_rgbn = 0.0
                # train_loss_ndvi = 0.0
                if global_step % config_accelerate.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config_accelerate.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(ckptdir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config_accelerate.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config_accelerate.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(ckptdir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(ckptdir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    accelerator.wait_for_everyone()
                if global_step in config_accelerate.checkpointing_steps_list:
                    if accelerator.is_main_process:
                        save_path = os.path.join(ckptdir, f"save-checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    accelerator.wait_for_everyone()

                # if global_step % config_accelerate.validate_steps == 0:
                #     if accelerator.is_main_process:
                #         # log_videos_local_and_wandb(
                #         #     batch, 
                #         #     current_step=global_step, 
                #         #     current_epoch=epoch
                #         # )
                #         log_videos_local_and_wandb(
                #             batch=batch, 
                #             current_step=global_step, 
                #             current_epoch=epoch,
                #             model=model,
                #             conditioner=conditioner,
                #             first_stage_model=first_stage_model,
                #             sampler=sampler,
                #             denoiser=denoiser,
                #             workdirnow=workdirnow,
                #             input_key=config_diffusion.input_key,
                #             scale_factor=config_diffusion.scale_factor,
                #             split="train",
                #         )
                #     accelerator.wait_for_everyone()
                
            read_data_time = round((end_read_time - start_read_time),2)
            start_read_time = time.time()
            process_data_time = round((start_read_time - end_read_time),2)
            before_encode_time = round((start_encode_time - end_read_time),2)
            encode_time = round((start_calc_loss_time - start_encode_time),2)
            loss_time = round((start_bp_loss_time - start_calc_loss_time),2)
            bp_time = round((end_bp_loss_time - start_bp_loss_time),2)
            logs = {"step_loss": loss.detach().item(), 
                    "true_latent_rmse": true_latent_rmse.item(),
                    "true_latent_rmse_mean": true_latent_rmse_mean.item(),
                    # "min_latent_rmse": min_latent_rmse.item(),
                    # "max_latent_rmse": max_latent_rmse.item(),
                    # "w_min": torch.min(w).item(),
                    # "w_max": torch.max(w).item(),
                    # "mean_true_latent_rmse": mean_true_latent_rmse.item(),
                    # "mean_min_latent_rmse": mean_min_latent_rmse.item(),
                    # "mean_max_latent_rmse": mean_max_latent_rmse.item(),
                    # "loss_space": loss_space.detach().item(),
                    # "loss_rgbn":loss_rgbn.detach().item(),
                    # "loss_ndvi":loss_ndvi.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0], 
                    "read_data_time":read_data_time,
                    "process_data_time":process_data_time,
                    "before_encode_time":before_encode_time,
                    "encode_time":encode_time,
                    "loss_time":loss_time,
                    "bp_time":bp_time}
            # # 记录训练的metrics
            # for k,v in metric_rgbn.items():
            #     logs[f'{k}_rgbn'] = v
            # for k,v in metric_ndvi.items():
            #     logs[f'{k}_ndvi'] = v
    
            # # TODO ZSJ 记录结果指标的上限
            # for k,v in metric_upper_limit.items():
            #     logs[f'{k}_upper_limit'] = v

            progress_bar.set_postfix(**logs)
            if global_step % config_accelerate.logging_steps == 0:
                if accelerator.is_main_process:
                    logger.info("step="+str((resume_global_step//config_accelerate.gradient_accumulation_steps)+step)+" / total_step="+str(config_accelerate.max_train_steps)+", step_loss="+str(logs["step_loss"])+', lr='+str(logs["lr"]))

        if accelerator.is_main_process:
            logger.info(f"Current epoch is {first_epoch}, step is {global_step}.")
            if config_accelerate.checkpointing_epochs is True:
                save_path = os.path.join(ckptdir, f"epoch-{first_epoch}-step-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
        accelerator.wait_for_everyone()
        
        if global_step >= config_accelerate.max_train_steps:
            break

        # epoch_metric_rgbn = loss_fn.metric_rgbn.calculate_epoch_metrics()
        # # epoch_metric_rgbn['r_squared'] /= epoch_metric_rgbn['num_valid']
        # # epoch_metric_rgbn['rmse'] /= epoch_metric_rgbn['num_valid']
        # # epoch_metric_rgbn['nse'] /= epoch_metric_rgbn['num_valid']
        # # epoch_metric_rgbn['bias'] /= epoch_metric_rgbn['num_valid']
        # # epoch_metric_rgbn['rmse_5'] /= epoch_metric_rgbn['num_valid_5']

        # epoch_metric_ndvi = loss_fn.metric_ndvi.calculate_epoch_metrics()
        # # epoch_metric_ndvi['r_squared'] /= epoch_metric_ndvi['num_valid']
        # # epoch_metric_ndvi['rmse'] /= epoch_metric_ndvi['num_valid']
        # # epoch_metric_ndvi['nse'] /= epoch_metric_ndvi['num_valid']
        # # epoch_metric_ndvi['bias'] /= epoch_metric_ndvi['num_valid']
        # # epoch_metric_ndvi['rmse_5'] /= epoch_metric_ndvi['num_valid_5']

        # for k,v in epoch_metric_rgbn.items():
        #     accelerator.log({f'epoch_{k}_rgbn': v}, step=global_step)
        # for k,v in epoch_metric_ndvi.items():
        #     accelerator.log({f'epoch_{k}_ndvi': v}, step=global_step)
        # accelerator.log(epoch_metric_rgbn, step=global_step)
        # accelerator.log(epoch_metric_ndvi, step=global_step)

        # loss_fn.metric_rgbn.reset()
        # loss_fn.metric_ndvi.reset()

        # Validation !!!
        print('Start Validation Process')
        validate_metric_rgbn = EarthnetX_Metric(rgbn=True).to(accelerator.device, dtype=weight_dtype)
        validate_metric_ndvi = EarthnetX_Metric(rgbn=False).to(accelerator.device, dtype=weight_dtype)
        validate_metric_arvi = EarthnetX_Metric(rgbn=False).to(accelerator.device, dtype=weight_dtype)

        # upper_limit_rgbn = EarthnetX_Metric()
        # upper_limit_ndvi = EarthnetX_Metric()
        # mean_metric_rgbn = EarthnetX_Metric()
        # mean_metric_ndvi = EarthnetX_Metric()
        # decode_mean_metric_rgbn = EarthnetX_Metric()
        # decode_mean_metric_ndvi = EarthnetX_Metric()
        print('metric set!')

        if epoch % config_accelerate.validate_epoch_interval == 0:
            print('start valid epoch')
            for batch_num, batch in enumerate(test_dataloader):
                print(f'batch_num: {batch_num}')
                if batch_num >= config_accelerate.validate_batch_num:
                    break
                print('Start Validation Batch')
                # if accelerator.is_main_process:
                for batch_key in batch.keys():
                    if torch.is_tensor(batch[batch_key]):
                        batch[batch_key] = batch[batch_key].to(dtype=weight_dtype)
                # log_videos_local_and_wandb(
                #     batch, 
                #     current_step=global_step, 
                #     current_epoch=epoch
                # )
                # btach_metric_rgbn, btach_metric_ndvi, batch_upper_limit_rgbn, batch_upper_limit_ndvi, batch_mean_rgbn, batch_mean_ndvi, batch_decode_mean_rgbn, batch_decode_mean_ndvi, \
                btach_metric_rgbn, btach_metric_ndvi, btach_metric_arvi, validate_log = log_videos_local_and_wandb(
                    batch=batch,
                    model=model,
                    conditioner=conditioner,
                    # conditioner_input_keys = conditioner_input_keys,
                    first_stage_model=first_stage_model,
                    sampler=sampler,
                    denoiser=denoiser,
                    input_key=config_diffusion.input_key,
                    scale_factor=config_diffusion.scale_factor,
                    noise_image_num=config_diffusion.network_config.params.noise_image_num,
                    past_weigth=config_diffusion.loss_fn_config.params.past_weight,
                    metrics_rgbn=validate_metric_rgbn,
                    metrics_ndvi=validate_metric_ndvi,
                    metrics_arvi=validate_metric_arvi,
                    # upper_limit_rgbn=upper_limit_rgbn,
                    # upper_limit_ndvi=upper_limit_ndvi,
                    # mean_metric_rgbn=mean_metric_rgbn,
                    # mean_metric_ndvi=mean_metric_ndvi,
                    # decode_mean_metric_rgbn=decode_mean_metric_rgbn,
                    # decode_mean_metric_ndvi=decode_mean_metric_ndvi,
                )
                for k,v in btach_metric_rgbn.items():
                    accelerator.log({f'btach_metric_rgbn{k}': v}, step=global_step)
                    print(f'btach_metric_rgbn{k}: {v}')
                for k,v in btach_metric_ndvi.items():
                    accelerator.log({f'btach_metric_ndvi{k}': v}, step=global_step)
                    print(f'btach_metric_ndvi{k}: {v}')
                for k,v in btach_metric_arvi.items():
                    accelerator.log({f'btach_metric_arvi{k}': v}, step=global_step)
                    print(f'btach_metric_arvi{k}: {v}')
                # for k,v in batch_upper_limit_rgbn.items():
                #     accelerator.log({f'batch_upper_limit_rgbn{k}': v}, step=global_step)
                #     print(f'batch_upper_limit_rgbn{k}: {v}')
                # for k,v in batch_upper_limit_ndvi.items():
                #     accelerator.log({f'batch_upper_limit_ndvi{k}': v}, step=global_step)
                #     print(f'batch_upper_limit_ndvi{k}: {v}')
                # for k,v in batch_mean_rgbn.items():
                #     accelerator.log({f'batch_mean_rgbn{k}': v}, step=global_step)
                #     print(f'batch_mean_rgbn{k}: {v}')
                # for k,v in batch_mean_ndvi.items():
                #     accelerator.log({f'batch_mean_ndvi{k}': v}, step=global_step)
                #     print(f'batch_mean_ndvi{k}: {v}')
                # for k,v in batch_decode_mean_rgbn.items():
                #     accelerator.log({f'batch_decode_mean_rgbn{k}': v}, step=global_step)
                #     print(f'batch_decode_mean_rgbn{k}: {v}')
                # for k,v in batch_decode_mean_ndvi.items():
                #     accelerator.log({f'batch_decode_mean_ndvi{k}': v}, step=global_step)
                #     print(f'batch_decode_mean_ndvi{k}: {v}')

                def denormalize(array):
                    return (array+1.0)/2
                
                input_sample = denormalize(validate_log['inputs_sample']).detach().to(dtype=torch.float).cpu().numpy()  # c,h,w
                reconstructions_sample = denormalize(validate_log['reconstructions_sample']).detach().to(dtype=torch.float).cpu().numpy()  # c,h,w
                last_prediction_sample = denormalize(validate_log['last_samples']).detach().to(dtype=torch.float).cpu().numpy()  # c,h,w
                first_prediction_sample = denormalize(validate_log['first_samples']).detach().to(dtype=torch.float).cpu().numpy()  # c,h,w


                input_sample_ndvi = ((input_sample[-1,:,:] - input_sample[-2,:,:])/(input_sample[-1,:,:] + input_sample[-2,:,:] + 1e-8)*255).astype(np.uint8)  # h,w
                reconstructions_sample_ndvi = ((reconstructions_sample[-1,:,:] - reconstructions_sample[-2,:,:])/(reconstructions_sample[-1,:,:] + reconstructions_sample[-2,:,:] + 1e-8)*255).astype(np.uint8)  # h,w
                last_prediction_sample_ndvi = ((last_prediction_sample[-1,:,:] - last_prediction_sample[-2,:,:])/(last_prediction_sample[-1,:,:] + last_prediction_sample[-2,:,:] + 1e-8)*255).astype(np.uint8)  # h,w
                first_prediction_sample_ndvi = ((first_prediction_sample[-1,:,:] - first_prediction_sample[-2,:,:])/(first_prediction_sample[-1,:,:] + first_prediction_sample[-2,:,:] + 1e-8)*255).astype(np.uint8)  # h,w

                input_sample_rgb, input_sample_nir = \
                (input_sample[:3,:,:]*255).astype(np.uint8), (input_sample[3,:,:]*255).astype(np.uint8)
                reconstructions_sample_rgb, reconstructions_sample_nir = \
                (reconstructions_sample[:3,:,:]*255).astype(np.uint8), (reconstructions_sample[3,:,:]*255).astype(np.uint8)
                last_prediction_sample_rgb, last_prediction_sample_nir = \
                (last_prediction_sample[:3,:,:]*255).astype(np.uint8), (last_prediction_sample[3,:,:]*255).astype(np.uint8)
                first_prediction_sample_rgb, first_prediction_sample_nir = \
                (first_prediction_sample[:3,:,:]*255).astype(np.uint8), (first_prediction_sample[3,:,:]*255).astype(np.uint8)

                input_sample_rgb = Image.fromarray(input_sample_rgb.transpose(1, 2, 0))
                input_sample_nir = Image.fromarray(input_sample_nir)
                reconstructions_sample_rgb = Image.fromarray(reconstructions_sample_rgb.transpose(1, 2, 0))
                reconstructions_sample_nir = Image.fromarray(reconstructions_sample_nir)
                last_prediction_sample_rgb = Image.fromarray(last_prediction_sample_rgb.transpose(1, 2, 0))
                last_prediction_sample_nir = Image.fromarray(last_prediction_sample_nir)
                first_prediction_sample_rgb = Image.fromarray(first_prediction_sample_rgb.transpose(1, 2, 0))
                first_prediction_sample_nir = Image.fromarray(first_prediction_sample_nir)

                input_sample_ndvi = Image.fromarray(input_sample_ndvi)
                reconstructions_sample_ndvi = Image.fromarray(reconstructions_sample_ndvi)
                last_prediction_sample_ndvi = Image.fromarray(last_prediction_sample_ndvi)
                first_prediction_sample_ndvi = Image.fromarray(first_prediction_sample_ndvi)


                accelerator.log({
                    "validate_input_sample_rgb":wandb.Image(input_sample_rgb),
                    "validate_input_sample_nir":wandb.Image(input_sample_nir),
                    "validate_reconstructions_sample_rgb":wandb.Image(reconstructions_sample_rgb),
                    "validate_reconstructions_sample_nir":wandb.Image(reconstructions_sample_nir),
                    "validate_last_prediction_sample_rgb": wandb.Image(last_prediction_sample_rgb),
                    "validate_last_prediction_sample_nir": wandb.Image(last_prediction_sample_nir),
                    "validate_first_prediction_sample_rgb": wandb.Image(first_prediction_sample_rgb),
                    "validate_first_prediction_sample_nir": wandb.Image(first_prediction_sample_nir),
                    "validate_input_sample_ndvi": wandb.Image(input_sample_ndvi),
                    "validate_reconstructions_sample_ndvi": wandb.Image(reconstructions_sample_ndvi),
                    "validate_last_prediction_sample_ndvi": wandb.Image(last_prediction_sample_ndvi),
                    "validate_first_prediction_sample_ndvi": wandb.Image(first_prediction_sample_ndvi),
                    }, step=global_step)
                accelerator.wait_for_everyone()

            epoch_validate_metric_rgbn = validate_metric_rgbn.calculate_epoch_metrics()
            epoch_validate_metric_ndvi = validate_metric_ndvi.calculate_epoch_metrics()
            epoch_validate_metric_arvi = validate_metric_arvi.calculate_epoch_metrics()
            # epoch_upper_limit_rgbn=upper_limit_rgbn.calculate_epoch_metrics()
            # epoch_upper_limit_ndvi=upper_limit_ndvi.calculate_epoch_metrics()
            # epoch_mean_metric_rgbn=mean_metric_rgbn.calculate_epoch_metrics()
            # epoch_mean_metric_ndvi=mean_metric_ndvi.calculate_epoch_metrics()
            # epoch_decode_mean_metric_rgbn=decode_mean_metric_rgbn.calculate_epoch_metrics()
            # epoch_decode_mean_metric_ndvi=decode_mean_metric_ndvi.calculate_epoch_metrics()

            for k,v in epoch_validate_metric_rgbn.items():
                accelerator.log({f'epoch_validate_metric_rgbn{k}': v}, step=global_step)
                print(f'epoch_validate_metric_rgbn{k}: {v}')
            for k,v in epoch_validate_metric_ndvi.items():
                accelerator.log({f'epoch_validate_metric_ndvi{k}': v}, step=global_step)
                print(f'epoch_validate_metric_ndvi{k}: {v}')
            for k,v in epoch_validate_metric_arvi.items():
                accelerator.log({f'epoch_validate_metric_arvi{k}': v}, step=global_step)
                print(f'epoch_validate_metric_arvi{k}: {v}')

            # for k,v in epoch_upper_limit_rgbn.items():
            #     accelerator.log({f'epoch_upper_limit_rgbn{k}': v}, step=global_step)
            #     print(f'epoch_upper_limit_rgbn{k}: {v}')
            # for k,v in epoch_upper_limit_ndvi.items():
            #     accelerator.log({f'epoch_upper_limit_ndvi{k}': v}, step=global_step)
            #     print(f'epoch_upper_limit_ndvi{k}: {v}')
            # for k,v in epoch_mean_metric_rgbn.items():
            #     accelerator.log({f'epoch_mean_metric_rgbn{k}': v}, step=global_step)
            #     print(f'epoch_mean_metric_rgbn{k}: {v}')
            # for k,v in epoch_mean_metric_ndvi.items():
            #     accelerator.log({f'epoch_mean_metric_ndvi{k}': v}, step=global_step)
            #     print(f'epoch_mean_metric_ndvi{k}: {v}')
            # for k,v in epoch_decode_mean_metric_rgbn.items():
            #     accelerator.log({f'epoch_decode_mean_metric_rgbn{k}': v}, step=global_step)
            #     print(f'epoch_decode_mean_metric_rgbn{k}: {v}')
            # for k,v in epoch_decode_mean_metric_ndvi.items():
            #     accelerator.log({f'epoch_decode_mean_metric_ndvi{k}': v}, step=global_step)
            #     print(f'epoch_decode_mean_metric_ndvi{k}: {v}')

            validate_metric_rgbn.reset()
            validate_metric_ndvi.reset()
            validate_metric_arvi.reset()
            # upper_limit_rgbn.reset()
            # upper_limit_ndvi.reset()
            # mean_metric_rgbn.reset()
            # mean_metric_ndvi.reset()
            # decode_mean_metric_rgbn.reset()
            # decode_mean_metric_ndvi.reset()
    
    # except (RuntimeError, Exception, BaseException) as err:
    #     logger.warning(err)
    # finally:
    #     if accelerator.is_main_process:
    #         logger.info("############################")
    #         logger.info("### Summoning Checkpoint ###")
    #         logger.info("############################")
    #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    #         if config_accelerate.checkpoints_total_limit is not None:
    #             checkpoints = os.listdir(ckptdir)
    #             checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    #             # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    #             if len(checkpoints) >= config_accelerate.checkpoints_total_limit:
    #                 num_to_remove = len(checkpoints) - config_accelerate.checkpoints_total_limit + 1
    #                 removing_checkpoints = checkpoints[0:num_to_remove]

    #                 logger.info(
    #                     f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
    #                 )
    #                 logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

    #                 for removing_checkpoint in removing_checkpoints:
    #                     removing_checkpoint = os.path.join(ckptdir, removing_checkpoint)
    #                     shutil.rmtree(removing_checkpoint)

    #         save_path = os.path.join(ckptdir, f"checkpoint-{global_step}")
    #         accelerator.save_state(save_path)
    #         logger.info(f"Saved state to {save_path}")
    #         wandb.finish()
    accelerator.end_training()

if __name__ == "__main__":
    main()