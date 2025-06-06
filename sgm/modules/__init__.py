from .encoders.modules import GeneralConditioner, AdaLNZeroConditioner

UNCONDITIONAL_CONFIG = {
    "target": "sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
