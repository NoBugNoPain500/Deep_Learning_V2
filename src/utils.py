"""Reproducibility, checkpoint, and model utilities."""
from __future__ import annotations
import random, numpy as np, torch

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def rng_state() -> dict:
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state: dict) -> None:
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def count_parameters(model) -> int: return sum(p.numel() for p in model.parameters())
