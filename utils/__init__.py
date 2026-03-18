import random
import os
import numpy as np
import torch

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    # To avoid RuntimeError with deterministic algorithms on newer CUDA versions
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # For newer versions of PyTorch
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations may not support deterministic mode
            # We can disable it or just warn
            pass
