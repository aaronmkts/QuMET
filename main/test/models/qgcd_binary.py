import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main"
     )
    )

import torch.nn as nn
from codebase.actions.train_ import train

from codebase.dataset import QuMETDataModule
from codebase.models.qgan.qgcd.configuration_qgcd import QGCDConfig
import toml
from codebase.models import get_model 
from codebase.models.qgan.qgcd_binary.modelling_qgan_binary import Binary_Generator, Binary_Discriminator
from codebase.models.qgan.qgcd_binary.configuration_qgan_binary import QGCD_Binary_Config

def main():
    # model
    config = QGCD_Binary_Config()
    generator = Binary_Generator(config = config)
    discriminator = Binary_Discriminator(config = config)

    # dataset
    task = "generation"
    dataset_name = "2d_ring_gaussian"
    binary = True
    n_qubits = generator.n_qubits

    # training
    batch_size = 256
    optimizer = "adam"
    max_epochs: int = 100
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.01
    weight_decay: float = 0.005
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    save_path: str = "./ckpts/test/testing"
    load_name: str = None
    load_type: str = ""
    evaluate_before_training: bool = False

    data_module = QuMETDataModule(
        name=dataset_name,
        batch_size=batch_size,
        binary=binary,
        n_qubits = n_qubits
    )

    train(
        generator=generator,
        discriminator=discriminator,
        task=task,
        data_module=data_module,
        generator_optimizer=optimizer,
        discriminator_optimizer=optimizer,
        max_epochs=max_epochs,
        max_steps=max_steps,
        generator_learning_rate=learning_rate,
        discriminator_learning_rate = learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        save_path=save_path,
        load_name=load_name,
        load_type=load_type,
        evaluate_before_training = evaluate_before_training
     )


if __name__ == "__main__":
    main()