"""Regenerate data/cifar_subset.npz from the HF CIFAR-10 mirror.

Run this in its own process; importing `datasets` alongside a torch.nn.Conv2d
construction triggers a Triton stack overflow on this machine.
"""

import numpy as np
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

def prep(split:str, n:int):
  x = np.stack([np.asarray(ds[split][i]["img"], np.float32) for i in range(n)])
  x = (x.transpose(0, 3, 1, 2) / 255.0 - 0.5) / 0.25
  y = np.array([ds[split][i]["label"] for i in range(n)], np.int64)
  return x.astype(np.float32), y

xt, yt = prep("train", 2048)
xs, ys = prep("test", 512)
np.savez("data/cifar_subset.npz", xtr=xt, ytr=yt, xte=xs, yte=ys)
print("saved", xt.shape, xs.shape)
