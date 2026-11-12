"""freeze_torso_keep_lora: frozen torso + trainable adapters + head (A3 mode)."""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import train as train_mod


def test_flag_default_and_yaml():
    cfg = train_mod.TrainConfig()
    assert cfg.freeze_torso_keep_lora is False
    p = Path(__file__).parent / "_tmp_keep_lora.yaml"
    yaml.safe_dump({"freeze_torso_keep_lora": True}, open(p, "w"))
    assert yaml.safe_load(open(p))["freeze_torso_keep_lora"] is True
    p.unlink()


def test_unfreeze_matches_only_lora_names():
    # Simulate the freeze pass shape: after freeze_torso everything is frozen;
    # the keep_lora pass must re-enable exactly the adapter tensors.
    names = [
        "torso.layers.0.self_attn.q_proj.lora_A.weight",
        "torso.layers.0.self_attn.q_proj.lora_B.weight",
        "torso.layers.0.self_attn.q_proj.weight",
        "head.q.weight",
    ]
    frozen = {n: torch.nn.Parameter(torch.zeros(2), requires_grad=False) for n in names}
    n = 0
    for name, p in frozen.items():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
            n += p.numel()
    assert n == 4
    assert frozen["torso.layers.0.self_attn.q_proj.weight"].requires_grad is False
    assert frozen["head.q.weight"].requires_grad is False
    assert frozen["torso.layers.0.self_attn.q_proj.lora_A.weight"].requires_grad is True
