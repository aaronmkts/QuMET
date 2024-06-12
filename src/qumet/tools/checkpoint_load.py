import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_lightning_ckpt_to_unwrapped_model(checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch Lightning checkpoint to a PyTorch model.
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
    """
    Load a PyTorch state dict or checkpoint containing state dict to a PyTorch model.
    """
    state_dict = torch.load(checkpoint)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict=state_dict)
    return model


def load_model(
    load_name: str, load_type: str = "pl", model: torch.nn.Module = None
) -> torch.nn.Module:
    """Load a pytorch/lightning checkpoint to a model.

    Args:
        load_name (str): path to the checkpoint
        load_type (str, optional): checkpoint type, must be one of ['pt', 'pl'],
        representing pytorch/lightning. Defaults to "auto" inferred from the extension.
        model (torch.nn.Module, optional): Model candidate to load checkpoint.

    Raises:
        ValueError: Unknown extension for 'load_type'.

    Returns:
        nn.Module: the model with the checkpoint loaded
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
