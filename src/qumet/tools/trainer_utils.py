"""Training utilities for QuMET models.

This module provides optimizer configuration and initialization utilities
for training quantum machine learning models.
"""

import torch


def get_optimizer(
    model: torch.nn.Module,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
):
    """Get a configured optimizer for model training.

    Sets up optimizer with parameter groups that apply different weight decay
    to different parameter types (excluding bias and LayerNorm weights).

    Args:
        model: Model whose parameters will be optimized.
        optimizer: Optimizer name ('adam', 'adamw', or 'sgd').
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization (not applied to bias/LayerNorm).

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        ValueError: If optimizer name is not supported.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    match optimizer:
        case "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=learning_rate
            )
        case "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
        case "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    return optimizer
