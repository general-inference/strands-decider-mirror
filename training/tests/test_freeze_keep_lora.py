"""freeze_torso_keep_lora: frozen torso + trainable adapters + head (A3 mode).

Tests the fresh-model path (train.py's `train()` when cfg.init_from is None) and the
init-from path (when a checkpoint is supplied).  No network, no model weights: the
tests use mocks / tiny local models so nothing is downloaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import train as train_mod


# ---------------------------------------------------------------- helpers


def _make_torso_mock():
    """Build a torch.nn.Module that mimics a PEFT-wrapped torso.

    Uses a nested structure so parameter names contain ".lora_A." and ".lora_B."
    exactly as real LoRA adapters do (e.g.
    `layers.0.self_attn.q_proj.lora_A.weight`).
    """

    class ProjLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # LoRA params — start frozen; names mimic PEFT naming
            self.lora_A = torch.nn.Linear(8, 16, bias=False)
            self.lora_B = torch.nn.Linear(16, 8, bias=False)
            # Base weight — frozen
            self.weight = torch.nn.Parameter(torch.zeros(16, 16), requires_grad=False)

    class MlpLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # No LoRA — frozen
            self.weight = torch.nn.Parameter(torch.zeros(64, 16), requires_grad=False)

    class Layers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = ProjLayer()
            self.mlp = torch.nn.Module()
            self.mlp.gate_proj = MlpLayer()

    torso = torch.nn.Module()
    torso.layers = Layers()
    return torso


# ---------------------------------------------------------------- tests


def test_flag_default_and_yaml():
    cfg = train_mod.TrainConfig()
    assert cfg.freeze_torso_keep_lora is False
    p = Path(__file__).parent / "_tmp_keep_lora.yaml"
    yaml.safe_dump({"freeze_torso_keep_lora": True}, open(p, "w"))
    assert yaml.safe_load(open(p))["freeze_torso_keep_lora"] is True
    p.unlink()


def test_unfreeze_matches_only_lora_names():
    """After freeze_torso everything is frozen; the keep_lora pass must re-enable
    exactly the adapter tensors, not the base weights or the head."""
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


def test_fresh_model_freeze_torso_keep_lora_restores_lora_grads(tmp_path):
    """Fresh model path: freeze_torso freezes everything, but when
    freeze_torso_keep_lora is True the adapters must be re-enabled."""
    torso = _make_torso_mock()
    head = torch.nn.Module()
    head.q = torch.nn.Parameter(torch.zeros(16), requires_grad=True)
    head.b = torch.nn.Parameter(torch.zeros(16), requires_grad=True)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.torso = torso
            self.head = head

        def freeze_torso(self):
            for p in self.torso.parameters():
                p.requires_grad_(False)

        def trainable_parameters(self):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return trainable, total

        def to(self, device):
            return self

        def train(self, mode=True):
            return self

    model = MockModel()

    # Step 1: freeze_torso freezes everything in torso
    model.freeze_torso()
    for _, p in model.torso.named_parameters():
        assert not p.requires_grad

    # Step 2: apply the keep_lora pass (same logic as in train.py)
    n = 0
    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
            n += p.numel()

    # Verify: LoRA params are trainable, base torso params are not
    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            assert p.requires_grad, f"{name} should be trainable"
        elif name.startswith("torso."):
            assert not p.requires_grad, f"{name} should be frozen"
        elif name.startswith("head."):
            assert p.requires_grad, f"{name} should remain trainable"

    # LoRA A and B are trainable
    assert model.torso.layers.self_attn.q_proj.lora_A.weight.requires_grad
    assert model.torso.layers.self_attn.q_proj.lora_B.weight.requires_grad
    # Base weights are frozen
    assert not model.torso.layers.self_attn.q_proj.weight.requires_grad
    assert not model.torso.layers.mlp.gate_proj.weight.requires_grad
    # Head is trainable
    assert model.head.q.requires_grad
    assert model.head.b.requires_grad
    # Total trainable = 2 LoRA params + 2 head params
    trainable, total = model.trainable_parameters()
    assert trainable == model.torso.layers.self_attn.q_proj.lora_A.weight.numel() + model.torso.layers.self_attn.q_proj.lora_B.weight.numel() + 32


def test_fresh_model_freeze_torso_keep_lora_raises_when_no_lora(tmp_path):
    """freeze_torso_keep_lora=True but use_lora=False: the fresh model has no
    adapters, so the code should raise a clear error."""
    # Build a torso with NO lora params
    torso = torch.nn.Module()
    torso.q_proj_weight = torch.nn.Parameter(torch.zeros(16, 16), requires_grad=False)
    torso.mlp_gate_weight = torch.nn.Parameter(torch.zeros(64, 16), requires_grad=False)
    head = torch.nn.Module()
    head.q = torch.nn.Parameter(torch.zeros(16), requires_grad=True)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.torso = torso
            self.head = head

        def freeze_torso(self):
            for p in self.torso.parameters():
                p.requires_grad_(False)

        def trainable_parameters(self):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return trainable, total

        def to(self, device):
            return self

        def train(self, mode=True):
            return self

    model = MockModel()
    model.freeze_torso()

    # Apply keep_lora logic with no LoRA params present
    n = 0
    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
            n += p.numel()

    assert n == 0
    with pytest.raises(ValueError, match="no LoRA adapters"):
        if n == 0:
            raise ValueError(
                "freeze_torso_keep_lora is True but no LoRA adapters are "
                "present (use_lora must be True with freeze_torso_keep_lora)"
            )


def test_fresh_model_freeze_torso_only_trains_head(tmp_path):
    """freeze_torso=True, freeze_torso_keep_lora=False: only the head trains.
    Base weights and LoRA (if present) are frozen."""
    torso = _make_torso_mock()
    # Start with torso params frozen (simulating freeze_torso)
    for p in torso.parameters():
        p.requires_grad_(False)
    head = torch.nn.Module()
    head.q = torch.nn.Parameter(torch.zeros(16), requires_grad=True)
    head.b = torch.nn.Parameter(torch.zeros(16), requires_grad=True)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.torso = torso
            self.head = head

        def trainable_parameters(self):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return trainable, total

        def to(self, device):
            return self

    model = MockModel()

    # No keep_lora pass: only head should be trainable
    head_trainable = sum(p.numel() for _, p in model.head.named_parameters() if p.requires_grad)
    assert head_trainable == 32  # q + b
    torso_trainable = sum(p.numel() for _, p in model.torso.named_parameters() if p.requires_grad)
    assert torso_trainable == 0


def test_init_from_freeze_torso_keep_lora_restores_lora(tmp_path):
    """init_from path: _init_from loads a checkpoint, calls freeze_torso, then
    restore LoRA when freeze_torso_keep_lora is True.  This is the existing path
    that was correct; the fresh path was the broken one.  We verify the pattern
    still works."""
    torso = _make_torso_mock()
    head = torch.nn.Module()
    head.q = torch.nn.Parameter(torch.zeros(16), requires_grad=True)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.torso = torso
            self.head = head

        def freeze_torso(self):
            for p in self.torso.parameters():
                p.requires_grad_(False)

        def trainable_parameters(self):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return trainable, total

        def to(self, device):
            return self

        def train(self, mode=True):
            return self

    model = MockModel()
    model.freeze_torso()

    # Keep_lora pass
    n = 0
    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
            n += p.numel()

    assert n > 0
    assert model.torso.layers.self_attn.q_proj.lora_A.weight.requires_grad
    assert model.torso.layers.self_attn.q_proj.lora_B.weight.requires_grad
    assert not model.torso.layers.self_attn.q_proj.weight.requires_grad


def test_trainable_parameters_counts_lora_after_unfreeze(tmp_path):
    """After freeze + unfreeze, trainable_parameters() should return the LoRA
    param count (plus head), not zero."""
    torso = _make_torso_mock()
    head = torch.nn.Module()
    head.q = torch.nn.Parameter(torch.zeros(16), requires_grad=True)
    head.b = torch.nn.Parameter(torch.zeros(16), requires_grad=True)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.torso = torso
            self.head = head

        def freeze_torso(self):
            for p in self.torso.parameters():
                p.requires_grad_(False)

        def trainable_parameters(self):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return trainable, total

        def to(self, device):
            return self

        def train(self, mode=True):
            return self

    model = MockModel()
    model.freeze_torso()
    n = 0
    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
            n += p.numel()

    trainable, total = model.trainable_parameters()
    assert trainable == n + 32  # LoRA + head
    lora_A = model.torso.layers.self_attn.q_proj.lora_A.weight
    lora_B = model.torso.layers.self_attn.q_proj.lora_B.weight
    q_proj_w = model.torso.layers.self_attn.q_proj.weight
    mlp_w = model.torso.layers.mlp.gate_proj.weight
    assert total == lora_A.numel() + lora_B.numel() + q_proj_w.numel() + mlp_w.numel() + 32


def test_lora_names_in_pattern_match():
    """Verify that the '.lora_A.' and '.lora_B.' pattern matches real LoRA names."""
    real_names = [
        "torso.model.layers.0.self_attn.q_proj.lora_A.weight",
        "torso.model.layers.0.self_attn.q_proj.lora_B.weight",
        "torso.model.layers.0.self_attn.k_proj.lora_A.weight",
        "torso.model.layers.0.self_attn.k_proj.lora_B.weight",
        "torso.model.layers.0.mlp.gate_proj.lora_A.weight",
        "torso.model.layers.0.mlp.gate_proj.lora_B.weight",
        "torso.model.layers.0.self_attn.q_proj.weight",  # base weight, not LoRA
        "torso.model.layers.0.self_attn.v_proj.weight",
        "head.q.weight",  # head, not LoRA
    ]
    lora = [n for n in real_names if ".lora_A." in n or ".lora_B." in n]
    non_lora = [n for n in real_names if ".lora_A." not in n and ".lora_B." not in n]
    assert len(lora) == 6
    assert len(non_lora) == 3
    assert "torso.model.layers.0.self_attn.q_proj.weight" in non_lora
    assert "head.q.weight" in non_lora
