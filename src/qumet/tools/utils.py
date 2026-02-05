"""General utility functions for QuMET.

This module provides utilities for tensor/numpy conversions, device management,
checkpoint handling, and CLI execution.
"""

import logging
import os
import subprocess

import numpy as np
import torch
from torch import Tensor

use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device("cuda:0" if use_cuda else "cpu")

logger = logging.getLogger(__name__)


def is_tensor(x):
    """Check if input is a PyTorch tensor.
    
    Args:
        x: Input to check.
    
    Returns:
        bool: True if input is a tensor.
    """
    return torch.is_tensor(x)


def to_numpy(x):
    """Convert a PyTorch tensor to numpy array.
    
    Args:
        x: PyTorch tensor to convert.
    
    Returns:
        np.ndarray: Numpy array on CPU.
    """
    if use_cuda:
        x = x.cpu()
    return x.detach().numpy()


def to_numpy_if_tensor(x):
    """Convert to numpy if input is a tensor, otherwise return as-is.
    
    Args:
        x: Input to potentially convert.
    
    Returns:
        np.ndarray or original type: Numpy array if input was tensor.
    """
    if is_tensor(x):
        return to_numpy(x)
    return x


def to_tensor(x):
    """Convert a numpy array to PyTorch tensor on configured device.
    
    Args:
        x: Numpy array to convert.
    
    Returns:
        torch.Tensor: Tensor on the configured device.
    """
    return torch.from_numpy(x).to(device)


def to_tensor_if_numpy(x):
    """Convert to tensor if input is numpy array, otherwise return as-is.
    
    Args:
        x: Input to potentially convert.
    
    Returns:
        torch.Tensor or original type: Tensor if input was numpy array.
    """
    if isinstance(x, np.ndarray):
        return to_tensor(x)
    return x


def copy_weights(src_weight: Tensor, tgt_weight: Tensor):
    """Copy weights from source tensor to target tensor.
    
    Args:
        src_weight: Source weight tensor.
        tgt_weight: Target weight tensor to copy into.
    """
    with torch.no_grad():
        tgt_weight.copy_(src_weight)


def get_checkpoint_file(checkpoint_dir):
    """Find a checkpoint file in the specified directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoint files.
    
    Returns:
        str or None: Filename of first .ckpt file found, or None.
    """
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def execute_cli(cmd, log_output: bool = True, log_file=None, cwd="."):
    """Execute a CLI command with optional logging.
    
    Args:
        cmd: Command to execute as a list of strings.
        log_output: Whether to log command output. Defaults to True.
        log_file: Optional file path to write output to.
        cwd: Working directory for command execution. Defaults to '.'.
    
    Returns:
        int: Command return code.
    """
    if log_output:
        logger.debug(f"{subprocess.list2cmdline(cmd)} (cwd = {cwd})")
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
    """Parse accelerator string to PyTorch device.
    
    Args:
        accelerator: Accelerator type ('auto', 'gpu', or 'cpu').
    
    Returns:
        torch.device: PyTorch device object.
    
    Raises:
        RuntimeError: If accelerator type is not supported.
    """
    if accelerator == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif accelerator == "gpu":
        device = torch.device("cuda:0")
    elif accelerator == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported accelerator {accelerator}")
    return device
