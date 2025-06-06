import os
import wandb

from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def init_wandb(save_dir, args, config, group_name, name_str, mode):
    os.makedirs(save_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = save_dir
    if not args.wandb:
        wandb.init(project=args.project_name, mode="offline", group=group_name)
    else:
        wandb.init(
            project=args.project_name, # project-name
            config=config, # config
            settings=wandb.Settings(code_dir="./sgm"), # settings
            group=group_name, # group-name
            name=name_str, # now-name
            mode=mode
        )