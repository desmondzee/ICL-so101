"""Run upstream `wan_va.train` unchanged, with WandB logging switched on.

The released robotwin_train preset hard-codes enable_wandb=False, and train.py only offers --disable-wandb, so this
sets the flag and then runs the module as torchrun would. It also replaces the hard-coded run name 'test_lln' with
$WANDB_NAME, and appends every logged step to $SO101_LOSS_LOG (JSON lines) so the Modal driver can read the losses
back. Nothing else changes: same config, same arguments, same training loop.

Optional, for short runs (both off unless set): $SO101_WARMUP_STEPS overrides the preset's warmup_steps (200), and
$SO101_COSINE_TOTAL_STEPS replaces upstream's warmup-then-constant schedule with linear warmup then cosine decay to 0
over that many steps. Upstream's schedule has no decay at all.

    python -m torch.distributed.run ... -m so101_wandb_train --config-name robotwin_train --datasets oxe:1.0 ...
"""

import json
import os
import runpy

import wandb

from wan_va.configs import VA_CONFIGS

import math

import wan_va.utils

_config = VA_CONFIGS[os.environ.get("CONFIG_NAME", "robotwin_train")]
_config.enable_wandb = True
if os.environ.get("SO101_WARMUP_STEPS"):
    _config.warmup_steps = int(os.environ["SO101_WARMUP_STEPS"])
if os.environ.get("SO101_COSINE_TOTAL_STEPS"):
    _total = int(os.environ["SO101_COSINE_TOTAL_STEPS"])

    def warmup_cosine_lambda(current_step, warmup_steps=1000):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = min(1.0, (current_step - warmup_steps) / max(1, _total - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    wan_va.utils.warmup_constant_lambda = warmup_cosine_lambda  # train.py imports it from wan_va.utils at run time

_init = wandb.init


def init(*args, **kwargs):
    kwargs["name"] = os.environ.get("WANDB_NAME") or kwargs.get("name")
    run = _init(*args, **kwargs)

    def log(values, *args, **kwargs):  # wandb.init rebinds wandb.log to the run's, so wrap that one
        if os.environ.get("SO101_LOSS_LOG"):
            with open(os.environ["SO101_LOSS_LOG"], "a") as f:
                f.write(json.dumps({"step": kwargs.get("step"), **values}) + "\n")
        return run.log(values, *args, **kwargs)

    wandb.log = log
    return run


wandb.init = init
runpy.run_module("wan_va.train", run_name="__main__", alter_sys=True)
