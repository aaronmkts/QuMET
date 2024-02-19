import numpy as np
import os
import pickle
import torch

import colorlog
import torch
import subprocess

from torch import Tensor

import logging

import itertools

use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device("cuda:0" if use_cuda else "cpu")

logger = logging.getLogger(__name__)


def is_tensor(x):
    return torch.is_tensor(x)


def to_numpy(x):
    if use_cuda:
        x = x.cpu()
    return x.detach().numpy()


def to_numpy_if_tensor(x):
    if is_tensor(x):
        return to_numpy(x)
    return x


def to_tensor(x):
    return torch.from_numpy(x).to(device)


def to_tensor_if_numpy(x):
    if isinstance(x, np.ndarray):
        return to_tensor(x)
    return x


def copy_weights(src_weight: Tensor, tgt_weight: Tensor):
    with torch.no_grad():
        tgt_weight.copy_(src_weight)


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def execute_cli(cmd, log_output: bool = True, log_file=None, cwd="."):
    if log_output:
        logger.debug("{} (cwd = {})".format(subprocess.list2cmdline(cmd), cwd))
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=cwd
        ) as result:
            if log_file:
                f = open(log_file, "w")
            if result.stdout or result.stderr:
                logger.info("")
            if result.stdout:
                for line in result.stdout:
                    if log_file:
                        f.write(line)
                    line = line.rstrip("\n")
                    # logger.trace(line)
            if result.stderr:
                for line in result.stderr:
                    if log_file:
                        f.write(line)
                    line = line.rstrip("\n")
                    # logger.trace(line)
            if log_file:
                f.close()
    else:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=cwd)
    return result.returncode

def parse_accelerator(accelerator: str):
    if accelerator == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif accelerator == "gpu":
        device = torch.device("cuda:0")
    elif accelerator == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported accelerator {accelerator}")
    return device