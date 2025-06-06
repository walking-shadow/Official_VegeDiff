from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch

from diffusers.models import AutoencoderKL
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from safetensors import safe_open

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)


# TODO: 1. Latent inference 2. Text Latent inference 3. Change lit to accelerate


class VideoDiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        pretrained_image_ckpt: Union[None, str] = None,
        trainable_modules=None
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.trainable_modules = trainable_modules
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        if pretrained_image_ckpt is not None:
            self.init_from_ckpt(pretrained_image_ckpt, verbose=True)
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        

    def init_from_ckpt( # init SD-xL from safetensors
        self, checkpoint_dir, ignore_keys=None, verbose=False,
    ) -> None: 
        model_state_dict=safe_open(checkpoint_dir, framework="pt", device="cpu")
        model_new_ckpt=dict()
        for i in model_state_dict.keys():
            if "model.diffusion_model." in i:
                model_new_ckpt[i.replace("model.diffusion_model.", "diffusion_model.")] = model_state_dict.get_tensor(i)
        sd=model_new_ckpt
        keys = list(sd.keys())
        for k in keys:
            if ignore_keys:
                for ik in ignore_keys:
                    if re.match(ik, k):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if verbose:
            print(
                f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")
        if verbose:
            print("")
            print("Unfrozen module", self.trainable_modules)
            print("Unfrozen parameters:")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if verbose:
                print("****", name)
            for trainable_module_name in self.trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    if verbose:
                        print(name)
                    break
        if verbose:
            print("")
            # if "temporal_transformer_block" in name:
            #     # with open("frozen.txt", "a+") as f:
            #     #     f.write(name+"\n")
            #     param.requires_grad = True
            # else:
            #     # with open("nofrozen.txt", "a+") as f:
            #     #     f.write(name+"\n")
            #     param.requires_grad = False

    def _init_first_stage(self, config):
        if "from_pretrained" in config.keys():
            self.first_stage_model = AutoencoderKL.from_pretrained(config.from_pretrained)
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config).eval()
            model.train = disabled_train
            for param in model.parameters():
                param.requires_grad = False
            self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            out = self.first_stage_model.decode(z)
            if isinstance(out, dict):
                out=out.sample
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x)
            if isinstance(z, dict):
                z=z.latent_dist.sample()
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        
        # for video
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W).contiguous()
        x = self.encode_first_stage(x)
        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W).contiguous().view(B, T * C, H, W).contiguous()
        
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,  
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None: # log learning rate
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        # configure training params
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        # configure training optimizers
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        # configure training learning rate scheduler
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    # TODO: Discard
    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    # log video: -> B,T,C,H,W
    @torch.no_grad()
    def log_videos(
        self,
        batch: Dict,
        N: int = 4,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        
        # get input
        
        x = self.get_input(batch)

        # get c and uc, notion here doesn't use the uc(uc=c)
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        # log reconstructions
        log = dict()
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        
        log["inputs"] = x
        x_= x
        B, T, C, H, W = x_.shape
        assert B==N
        x_ = x_.view(B*T, C, H, W)
        z = self.encode_first_stage(x_)
        _, C_, H_, W_ = z.shape
        log["reconstructions"] = self.decode_first_stage(z).view(B, T, C, H, W)
        
        # log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
        
        # sample
        sampling_kwargs = {}
        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=torch.Size([T*C_,H_,W_]), uc=uc, batch_size=N, **sampling_kwargs
                )
                samples=samples.view(B*T, C_, H_, W_)
            samples = self.decode_first_stage(samples).view(B, T, C, H, W)
            log["samples"] = samples
        return log
