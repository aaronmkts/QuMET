import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src"
    )
)

import torch.nn as nn
from qumet.actions.train import train
from qumet.dataset import QuMETDataModule, get_dataset_info
from qumet.models import get_model, get_model_info

import pytorch_lightning as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import time
from pathlib import Path


def main():
    def _setup_visualizer(visualiser, save_path):
        visualizer = None
        match visualiser:
            case "tensorboard":
                visualizer = TensorBoardLogger(
                    save_dir=save_path.joinpath("tensorboard")
                )
            case _:
                raise ValueError(f"unsupported reporting tool {visualiser}")
        return visualizer

    def _setup_folders(task, dataset_name):
        ROOT = Path(__file__).parent.parent.parent.absolute()
        # No project name is given; so we construct one structured as follows:
        # {MODEL-NAME}_{TASK-TYPE}_{DATASET-NAME}_{TIMESTAMP}
        # NOTE: We set the attribute in args so that any subsequent routine has
        # access to the name of the project. :)
        project = "{}_{}_{}".format(
            task,
            dataset_name,
            time.strftime("%Y-%m-%d"),
        )

        output_dir = Path(os.path.join(ROOT, "qumet_output")) / project
        output_dir_sw = Path(output_dir) / "software"
        output_dir_sw.mkdir(parents=True, exist_ok=True)

        print(f"File path -> {output_dir}")

        return output_dir_sw

    # Model
    model_name = "qcbm"
    model_info = get_model_info(model_name)

    task = "probs_generation" #image_generation discrete_generation
    dataset_name = "bars_and_stripes" #bars_and_stripes

    # Training params
    batch_size = 512
    n_qubits = 9
    num_workers = 1
    optimizer = "adam"
    learning_rate: float = 0.1
    weight_decay: float = 0.005
    output_dir_sw = _setup_folders(task, dataset_name)
    load_name: str = None
    load_type: str = ""
    is_to_auto_requeue = False
    report_to = "tensorboard"
    visualizer = _setup_visualizer(report_to, output_dir_sw)

    data_module = QuMETDataModule(
        model_name=None,
        name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        n_qubits=n_qubits,
        transform=None,
        discretise=False
    )
    dataset_info = get_dataset_info(dataset_name)
    model = get_model(model_name, task, dataset_info)

    # plt_trainer args
    max_epochs: int = 25
    max_steps: int = -1
    num_devices: int = 1
    num_nodes: int = 1
    accelerator: int = "auto"
    strategy = "auto"
    trainer_precision = "32"
    accumulate_grad_batches = 1
    log_every_n_steps = 1

    plt_trainer_args = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "devices": num_devices,
        "num_nodes": num_nodes,
        "accelerator": accelerator,
        "strategy": strategy,
        "precision": trainer_precision,
        "accumulate_grad_batches": accumulate_grad_batches,
        "log_every_n_steps": log_every_n_steps,
    }

    train_params = {
        "model": model,
        "model_info": model_info,
        "data_module": data_module,
        "dataset_info": dataset_info,
        "task": task,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "plt_trainer_args": plt_trainer_args,
        "auto_requeue": is_to_auto_requeue,
        "save_path": os.path.join(output_dir_sw, "training_ckpts"),
        "visualizer": visualizer,
        "load_name": load_name,
        "load_type": load_type,
    }

    train(**train_params)


if __name__ == "__main__":
    main()