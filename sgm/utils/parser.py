import argparse
import torch
import inspect

from pytorch_lightning.trainer import Trainer
from packaging import version
from inspect import Parameter

def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param] != Parameter.empty
    }
    return default_args


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

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
    parser.add_argument(
        "--resumedir",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="if setting, will resume from this project or ckpt",
    )
    parser.add_argument( # if resume, you change it none. i will load from the resumedir
        "--cfgdir",
        nargs="*",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether use wandb",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--test",
        type=str2bool,
        default=False,
        nargs="?",
        help="test",
    )
    parser.add_argument(
        "--autoresume",
        type=str2bool,
        default=True,
        nargs="?",
        help="must setting up autoresume",
    )
    # setting up your name
    # parser.add_argument(
    #     "-n",
    #     "--name",
    #     type=str,
    #     const=True,
    #     default="",
    #     nargs="?",
    #     help="if setting, the logdir will be like: datetime_name",
    # )
    # parser.add_argument(
    #     "--no_date",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)",
    # )
    # parser.add_argument(
    #     "--no_base_name",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,  # TODO: later default to True
    #     help="setup name from base but no base name, so only like: datetime",
    # )
    

    
    # setting up your resume
    # parser.add_argument(
    #     "-r",
    #     "--resume",
    #     type=str,
    #     const=True,
    #     default="",
    #     nargs="?",
    #     help="resume from logdir or checkpoint in logdir: 1. from known ckpt (dir/checkpoints/xxxxxx.ckpt) 2. from known dir (dir)",
    # )
    # # setting up your configs
    # parser.add_argument(
    #     "-b",
    #     "--base",
    #     nargs="*",
    #     metavar="base_config.yaml",
    #     help="paths to base configs. Loaded from left-to-right. "
    #     "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    #     default=list(),
    # )
    
    # setting up train and test
    # parser.add_argument(
    #     "-t",
    #     "--train",
    #     type=str2bool,
    #     const=True,
    #     default=True,
    #     nargs="?",
    #     help="train",
    # )
    # parser.add_argument(
    #     "--test",
    #     type=str2bool,
    #     const=True,
    #     default=False,
    #     nargs="?",
    #     help="test",
    # )
    
    # setting up your wandb
    # parser.add_argument(
    #     "--wandb",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="open wandb",
    # )
    # parser.add_argument(
    #     "--projectname",
    #     type=str,
    #     default="stablediffusion",
    # )
    
    # setting up your trainer device info
    # parser.add_argument(
    #     "--devices",
    #     type=int,
    #     default=1,
    # )
    # parser.add_argument(
    #     "--num_nodes",
    #     type=int,
    #     default=1,
    # )
    # parser.add_argument(
    #     "--accelerator",
    #     type=str,
    #     default="gpu",
    # )

    #TODO
    # parser.add_argument(
    #     "-p", "--project", help="name of new or path to existing project"
    # )
    # parser.add_argument(
    #     "-d",
    #     "--debug",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="enable post-mortem debugging",
    # )
    # parser.add_argument(
    #     "-s",
    #     "--seed",
    #     type=int,
    #     default=23,
    #     help="seed for seed_everything",
    # )
    # parser.add_argument(
    #     "-f",
    #     "--postfix",
    #     type=str,
    #     default="",
    #     help="post-postfix for default name",
    # )
    # parser.add_argument(
    #     "-l",
    #     "--logdir",
    #     type=str,
    #     default="logs",
    #     help="directory for logging dat shit",
    # )
    # parser.add_argument(
    #     "--scale_lr",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="scale base-lr by ngpu * batch_size * n_accumulate",
    # )
    # parser.add_argument(
    #     "--legacy_naming",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="name run based on config file name if true, else by whole path",
    # )
    # parser.add_argument(
    #     "--enable_tf32",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",
    # )
    # parser.add_argument(
    #     "--startup",
    #     type=str,
    #     default=None,
    #     help="Startuptime from distributed script",
    # )

    #TODO
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    default_args = default_trainer_args()
    for key in default_args:
        try:
            parser.add_argument("--" + key, default=default_args[key])
        except Exception as e:
            continue
    return parser