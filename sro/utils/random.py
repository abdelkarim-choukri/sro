from __future__ import annotations
import os, random
import numpy as np

try:
    import torch
except Exception:
    torch = None
# this chunk is about making  runs reproducible (same results every time) 
# by seeding all RNGs (random number generators) and 
# forcing PyTorch/cuDNN (NVIDIA’s deep-learning kernels) into deterministic behavior
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
