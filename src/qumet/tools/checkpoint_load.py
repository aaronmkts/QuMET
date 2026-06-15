"""Checkpoint loading utilities for QuMET models.

This module provides functions to load PyTorch and PyTorch Lightning checkpoints
into model instances, handling state dictionary mapping and conversions.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def load_lightning_ckpt_to_unwrapped_model(checkpoint: str, model: torch.nn.Module):
    """Load a PyTorch Lightning checkpoint into an unwrapped PyTorch model.

    Handles the mapping of Lightning's wrapped state dict keys to the base model.

    Args:
        checkpoint: Path to the Lightning checkpoint file.
        model: PyTorch model to load the checkpoint into.

    Returns:
        torch.nn.Module: Model with loaded checkpoint weights.
    """
    src_state_dict = torch.load(checkpoint)["state_dict"]
    tgt_state_dict = model.state_dict()
    new_tgt_state_dict = {}
    for k, v in src_state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        if possible_tgt_k in tgt_state_dict:
            new_tgt_state_dict[possible_tgt_k] = v
    model.load_state_dict(state_dict=new_tgt_state_dict)
    return model


def load_unwrapped_ckpt(checkpoint: str, model: torch.nn.Module):
    """Load a PyTorch state dict checkpoint into a model.

    Args:
        checkpoint: Path to the PyTorch checkpoint file.
        model: PyTorch model to load the checkpoint into.

    Returns:
        torch.nn.Module: Model with loaded checkpoint weights.
    """
    state_dict = torch.load(checkpoint)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict=state_dict)
    return model


def load_model(
    load_name: str, load_type: str = "pl", model: torch.nn.Module = None
) -> torch.nn.Module:
    """Load a PyTorch or Lightning checkpoint into a model.

    Args:
        load_name: Path to the checkpoint file.
        load_type: Checkpoint type, either 'pt' (PyTorch) or 'pl' (Lightning).
            Defaults to 'pl'.
        model: Model instance to load the checkpoint into.

    Returns:
        torch.nn.Module: Model with loaded checkpoint weights.

    Raises:
        ValueError: If load_type is not 'pt' or 'pl'.
    """

    if load_type not in ["pt", "pl"]:
        raise ValueError(f"Unknown extension for 'load_type': {load_type}")

    if load_type == "pt":
        model = load_unwrapped_ckpt(checkpoint=load_name, model=model)
        logger.info(f"Loaded pytorch checkpoint from {load_name}")

    elif load_type == "pl":
        if not load_name.endswith(".ckpt"):
            logger.warning(
                f"Lightning checkpoint should end with '.ckpt', but got {load_name}"
            )
        model = load_lightning_ckpt_to_unwrapped_model(
            checkpoint=load_name, model=model
        )
        logger.info(f"Loaded pytorch lightning checkpoint from {load_name}")
    return model
