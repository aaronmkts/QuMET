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
from codebase.models.manual.qgan.configuration_qgan import QganConfig
from codebase.models.manual.qgan.modelling_qgan import Generator, Discriminator
import toml


def main():
    generator = Generator()
    discriminator = Discriminator()
    task = "generation"
    dataset_name = "gaussian"
    
    
    # Reduced for unit test
    batch_size = 50
    optimizer = "adam"
    max_epochs: int = 100
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    save_path: str = "./ckpts/test"
    load_name: str = None
    load_type: str = ""
    evaluate_before_training: bool = True

    data_module = QuMETDataModule(
        model_name= None,
        name=dataset_name,
        batch_size=batch_size,
    )

    train(
        generator=generator,
        discriminator  = discriminator,
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