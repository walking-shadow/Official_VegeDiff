import argparse
import datetime
import glob
import inspect
import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
import cv2
import imageio

from einops import rearrange
from PIL import Image
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from inspect import Parameter
from PIL import Image
from typing import Union
from sgm.utils.mmlogger import MMLogger as logger
from sgm.utils.mmlogger import print_log
from sgm.utils.checkpoint_util import get_checkpoint_name
from sgm.util import (
    exists,
    instantiate_from_config,
    isheatmap,
)

# General Callback
class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        datenow,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
    ):
        super().__init__()
        self.resume = resume
        self.datenow = datenow
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.infologger = None

    # save checkpoint
    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        if trainer.global_rank == 0:
            print_log("!!!!!Summoning checkpoint!!!!!")
            ckpt_path = os.path.join(self.ckptdir, "latest.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # makedirs and save config
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # if "callbacks" in self.lightning_config:
            #     if (
            #         "trainsteps_checkpoint"
            #         in self.lightning_config["callbacks"]
            #     ):
            #         os.makedirs(
            #             os.path.join(self.ckptdir, "trainstep_checkpoints"),
            #             exist_ok=True,
            #         )
            
            print_log("Project config:", logger=self.infologger)
            print_log(OmegaConf.to_yaml(self.config), logger=self.infologger)
            print_log("", logger=self.infologger)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.datenow)),
            )
            print_log("Lightning config", logger=self.infologger)
            print_log(OmegaConf.to_yaml(self.lightning_config), logger=self.infologger)
            print_log("", logger=self.infologger)
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.datenow)),
            )
            print_log("Start Training!", logger=self.infologger)

# Image Logger Callback
# What you should is to code "def log_images" function in a new trainer
class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency, # log images every "batch_frequency" (change)
        max_images, # max images in each log (change)
        clamp=True, # whether to clamp the imgs into [-1,1]
        increase_log_steps=True, # whether to add 1,2,4,8... to log (change)
        rescale=True, # whether to change [-1, 1] into [0, 1]
        disabled=False, # whether to log
        log_on_batch_idx=False, # log according to batch_idx in every epoch or global_step
        log_first_step=False, # whether to log the first step, during batch_end
        log_images_kwargs=None, # other params for log_images in each trainer
        log_before_first_step=False, # whether to log the fisrt step, during batch_start
        enable_autocast=True, # whether to use autocast (change)
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:
                # transform imgs to rgb space
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                # log imgs in local file system
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)
                # log imgs in wandb
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[
                            img,
                        ],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # batch_idx refresh every epoch
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        # check_freq and max_images > 0
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and
            # batch_idx > 5 and
            self.max_images > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
        
            # setup autocast
            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            # select top max_images images from output
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)
            # log images and save it in local system
            self.log_local(
                pl_module._trainer.workdirnow,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        # check (every batch_freq or log_steps) and (check_idx > 0)
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step): 
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                # print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # check after every batch end : log_first_step=True or freq
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step): 
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # check before every batch start : log_before_first_step=True and first step
        if self.log_before_first_step and pl_module.global_step == 0: 
            print(f"{self.__class__.__name__}: logging before training")
            self.log_img(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
         # TODO Clean
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


def numpy_array_to_video(numpy_array, video_out_path, fps=8): #TCHW -> THWC
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1)) #THWC
    with imageio.get_writer(video_out_path, fps=fps) as video:
        for image in numpy_array:
            video.append_data(image)

# Image Logger Callback
# What you should is to code "def log_images" function in a new trainer
class VideoLogger(Callback):
    def __init__(
        self,
        batch_frequency, # log images every "batch_frequency" (change)
        max_videos, # max images in each log (change)
        clamp=True, # whether to clamp the imgs into [-1,1]
        increase_log_steps=True, # whether to add 1,2,4,8... to log (change)
        rescale=True, # whether to change [-1, 1] into [0, 1]
        disabled=False, # whether to log
        log_on_batch_idx=False, # log according to batch_idx in every epoch or global_step
        log_first_step=False, # whether to log the first step, during batch_end
        log_videos_kwargs=None, # other params for log_images in each trainer
        log_before_first_step=False, # whether to log the fisrt step, during batch_start
        enable_autocast=True, # whether to use autocast (change)
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_videos_kwargs = log_videos_kwargs if log_videos_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        videos,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "videos", split)
        for k in videos:
            # transform imgs to rgb space
            B, T, C, H, W = videos[k].shape
            grid = videos[k].view(B, T*C, H, W)
            grid = torchvision.utils.make_grid(grid, nrow=4)
            grid = grid.view(T, C, grid.shape[-2], grid.shape[-1])
            
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)#TCHW
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # grid = rearrange(grid, "(b t) c h w -> b t c h w")
            error_times=0
            while(True):
                try:
                    numpy_array_to_video(grid, path, fps=8)
                    break
                except Exception as e:
                    error_times+=1
                    print("error times:", error_times)
            wandb.log({f"{split}/{k}_videos": wandb.Video(grid, fps=30, format="mp4")}, step=pl_module.global_step)

    @rank_zero_only
    def log_video(self, pl_module, batch, batch_idx, split="train"):
        # batch_idx refresh every epoch
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        # check_freq and max_images > 0
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_videos")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_videos)
            and
            # batch_idx > 5 and
            self.max_videos > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
        
            # setup autocast
            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                videos = pl_module.log_videos(
                    batch, split=split, **self.log_videos_kwargs
                )

            # select top max_images images from output
            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if self.clamp and not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)
            
            # log images and save it in local system
            self.log_local(
                pl_module._trainer.workdirnow,
                split,
                videos,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        # check (every batch_freq or log_steps) and (check_idx > 0)
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step): 
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                # print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # check after every batch end : log_first_step=True or freq
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step): 
            self.log_video(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # check before every batch start : log_before_first_step=True and first step
        if self.log_before_first_step and pl_module.global_step == 0: 
            print(f"{self.__class__.__name__}: logging before training")
            self.log_video(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_video(pl_module, batch, batch_idx, split="val")
        #  # TODO Clean
        # if hasattr(pl_module, "calibrate_grad_norm"):
        #     if (
        #         pl_module.calibrate_grad_norm and batch_idx % 25 == 0
        #     ) and batch_idx > 0:
        #         self.log_gradients(trainer, pl_module, batch_idx=batch_idx)