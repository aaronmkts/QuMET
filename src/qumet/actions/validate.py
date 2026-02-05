"""Validation action for QuMET models.

This module provides the validation function for evaluating trained quantum
machine learning models using PyTorch Lightning.
"""

import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment

from qumet.plt_wrapper import get_model_wrapper
from qumet.tools.checkpoint_load import load_model

logger = logging.getLogger(__name__)


def validate(
    model,
    model_info,
    data_module,
    dataset_info,
    task,
    optimizer,
    learning_rate,
    plt_trainer_args,
    auto_requeue,
    save_path,
    visualizer,
    load_name,
    load_type,
):
    """Validate a trained quantum machine learning model.
    
    Sets up PyTorch Lightning trainer and runs validation on the model using
    a checkpoint file or pretrained weights.
    
    Args:
        model: Model instance to validate.
        model_info: Model metadata and configuration.
        data_module: DataModule for loading validation data.
        dataset_info: Dataset metadata and configuration.
        task: Task type (e.g., 'discrete_generation').
        optimizer: Optimizer name ('adam', etc.).
        learning_rate: Learning rate (used for optimizer setup).
        plt_trainer_args: Arguments for PyTorch Lightning Trainer.
        auto_requeue: Whether to enable SLURM auto-requeue.
        save_path: Directory path for outputs.
        visualizer: Logger for experiment tracking (WandB/TensorBoard).
        load_name: Path to checkpoint file to load.
        load_type: Type of checkpoint ('pt' or 'pl').
    """
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt_trainer_args["callbacks"] = []
        plt_trainer_args["logger"] = visualizer

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    wrapper_cls = get_model_wrapper(model_info, task)

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
    plt_model = wrapper_cls(
        model,
        info=dataset_info,
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**plt_trainer_args)

    if data_module.dataset_info.validation_split_available:
        trainer.validate(plt_model, datamodule=data_module)
    else:
        logger.warning(
            f"Validation split not available for dataset {data_module.info.name}"
        )
