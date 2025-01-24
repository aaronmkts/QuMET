import logging
import os
from pathlib import Path
from qumet.plt_wrapper import get_model_wrapper
from qumet.plt_wrapper.vaeqgan_wrapper import Encoder
from qumet.tools.checkpoint_load import load_model
from qumet.tools.progress_bar import progress_bar
from qumet.tools.callbacks import select_callbacks
import lightning.pytorch as pl
from lightning.pytorch.callbacks  import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.distributed.fsdp import FullyShardedDataParallel
from lightning.pytorch.strategies import DDPStrategy

import torch


logger = logging.getLogger(__name__)


def train(
    model,
    model_info,
    data_module,
    dataset_info,
    task,
    optimizer,
    learning_rate,
    weight_decay,
    plt_trainer_args,
    auto_requeue,
    save_path,
    visualizer,
    load_name,
    load_type,
    metrics,
    metric_init_args,
    add_vae
):
    if save_path is not None:
        # if save_path is None, the model will not be saved
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="metrics/val_mse_reduction",
            mode="min",
            filename="best",
            dirpath=save_path,
            save_last=True,
        )
     
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        
        
        callbacks = select_callbacks(model_info, dataset_info, task, metrics, metric_init_args)
        callbacks.append(checkpoint_callback)
        callbacks.append(lr_monitor_callback)
       
        plt_trainer_args["callbacks"] = [cb for cb in callbacks if cb is not None]
        plt_trainer_args["logger"] = visualizer

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins
    

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")
    
    wrapper_cls = get_model_wrapper(model_info, task, add_vae)

    if add_vae:
        encoder = Encoder()

        pl_model = wrapper_cls(
            base_model=model,
            encoder=encoder,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=plt_trainer_args["max_epochs"],
            optimizer=optimizer,
        )
        
    else:
        pl_model = wrapper_cls(
            model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=plt_trainer_args["max_epochs"],
            optimizer=optimizer,
        )
    
    trainer = pl.Trainer(**plt_trainer_args, deterministic= True, num_sanity_val_steps=0)
    #trainer.validate(model=pl_model, datamodule=data_module)
    trainer.fit(
        pl_model,
        datamodule=data_module,
    )

    # Save the trained model along with relevant metadata in the training_ckpts folder.
    # NOTE: This is important if the model was previously transformed with architectural
    # changes. The state dictionary that's saved by PyTorch Lightning wouldn't work.
    if save_path is not None and load_name is not None:
        transformed_ckpt = Path(save_path) / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
