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
from natsort import natsorted

# def get_checkpoint_name(logdir, logger):
#     ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
#     ckpt = natsorted(glob.glob(ckpt))
#     ckpt = ckpt[0]
#     logger.info(f'Available "last" checkpoints: {ckpt}')
#     # if len(ckpt) > 1: # ßfind the latest ckpt by modification time
#     #     logger.info("Got most recent checkpoint")
#     #     ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
#     #     logger.info(f"Most recent ckpt is {ckpt}")
#     #     with open(os.path.join(logdir, "redundancy", "most_recent_ckpt.txt"), "w") as f:
#     #         f.write(ckpt + "\n")
#     #     try:
#     #         version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
#     #     except Exception as e:
#     #         logger.info("version confusion but not bad")
#     #         logger.info(e)
#     #         version = 1
#     #     # version = last_version + 1
#     # else:
#     #     # in this case, we only have one "last.ckpt"
#     #     ckpt = ckpt[0]
#     #     version = 1
#     # melk_ckpt_name = f"last-v{version}.ckpt"
#     # logger.info(f"Current melk ckpt name (used to save): {melk_ckpt_name}")
#     return ckpt, None

def get_checkpoint_name(workdirnow):
    ckptdir = os.path.join(workdirnow, "checkpoints", "last.ckpt")
    return ckptdir