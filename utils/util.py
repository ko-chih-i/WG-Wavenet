import torch
import numpy as np
from hparams import hparams as hps

def mode(obj, model=False):
    # 强制使用 CPU
    d = torch.device('cpu')  # 强制将设备设置为 CPU
    return obj.to(d, non_blocking=False if model else hps.pin_mem)

def to_arr(var):
    # 确保张量转换为 NumPy 数组时在 CPU 上
    return var.cpu().detach().numpy().astype(np.float32)
