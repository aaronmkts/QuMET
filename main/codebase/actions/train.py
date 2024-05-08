import logging
import os
from pathlib import Path
from codebase.plt_wrapper import get_model_wrapper
from codebase.tools.checkpoint_load import load_model
from codebase.tools.progress_bar import progress_bar
import lightning.pytorch as pl
from lightning.pytorch.callbacks  import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.distributed.fsdp import FullyShardedDataParallel
from lightning.pytorch.strategies import DDPStrategy


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
):
    if save_path is not None:
        # if save_path is None, the model will not be saved
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_kl_epoch",
            mode="min",
            filename="best",
            dirpath=save_path,
            save_last=True,
        )
        
        # tb_logger = TensorBoardLogger(save_dir=save_path, name="logs")
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")

        plt_trainer_args["callbacks"] = [
            checkpoint_callback,
            lr_monitor_callback,
            #progress_bar()
        ]
        plt_trainer_args["logger"] = visualizer

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    # Check optimizer
    # if plt_trainer_args["strategy"] in ["deepspeed_stage_3"]:
    #     assert optimizer in [
    #         "FusedAdam",
    #         "fused_adam",
    #     ], "optimizer should be 'fused_adam' given --strategy={}".format(
    #         plt_trainer_args["strategy"]
    #     )
    # elif plt_trainer_args["strategy"] in ["fsdp_custom"]:
    #     plt_trainer_args["strategy"] = CustomFSDPStrategy()

    wrapper_cls = get_model_wrapper(model_info, task)

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")

    pl_model = wrapper_cls(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=plt_trainer_args["max_epochs"],
        optimizer=optimizer,
    )
    
    trainer = pl.Trainer(**plt_trainer_args, deterministic= True)
    
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
