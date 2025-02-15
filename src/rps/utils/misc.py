import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_run_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(log_dir) if d.startswith('run')]
    existing_runs.sort()
    for i in range(1, len(existing_runs) + 2):
        run_dir = f"run_{i}"
        if run_dir not in existing_runs:
            run_dir = os.path.join(log_dir, run_dir)
            break
    
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Selected device: {device}")
    return device