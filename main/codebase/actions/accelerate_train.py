import copy
from functools import partial
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    ShardingStrategy,
)
import math
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm.auto import tqdm
import argparse
import toml
import sys

from codebase.tools import get_scheduler, get_optimizer


def train(
    model,
    task,
    data_module,
    generator_optimizer,
    discriminator_optimizer,
    max_epochs: int,
    max_steps: int = -1,
    generator_learning_rate: float = 5e-5,
    discriminator_learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: str = "linear",
    num_warmup_steps: int = 0,
    save_path: str = ".",
    load_name: str = None,
    load_type: str = "",
):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        cpu_offload=CPUOffload(offload_params=False),
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={},
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=save_path,
        fsdp_plugin=fsdp_plugin,
    )

    if accelerator.is_main_process:
        print(f"Using accelerator: {accelerator.state}")

    # Prepare model
    model = accelerator.prepare(model)

    # dataset
    if accelerator.is_local_main_process:
        data_module.prepare_data()

    accelerator.wait_for_everyone()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    eval_dataloader = data_module.val_dataloader()
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # optimizers
    optG = get_optimizer(
        accelerator, model, generator_optimizer, generator_learning_rate, weight_decay
    )
    optD = get_optimizer(
        accelerator,
        model,
        discriminator_optimizer,
        discriminator_learning_rate,
        weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps / accelerator.num_processes
    )

    if max_steps == -1:
        max_steps = max_epochs * num_update_steps_per_epoch
    else:
        max_steps = min(max_steps, max_epochs * num_update_steps_per_epoch)

    # lr_scheduler for classical discriminator
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optD,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    experiment_config = {
        "model": type(model).__name__,
        "dataset": type(data_module.train_dataset).__name__,
        "task": task,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "generator_learning_rate": generator_learning_rate,
        "discriminator_learning_rate": discriminator_learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr_scheduler_type": lr_scheduler_type,
        "num_warmup_steps": num_warmup_steps,
        "save_path": save_path,
        "load_name": load_name,
        "load_type": load_type,
    }

    accelerator.init_trackers(save_path.replace("/", "_"), config=experiment_config)

    # training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Instantaneous batch size per device = {data_module.batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {data_module.batch_size * accelerator.num_processes * gradient_accumulation_steps}"
        )
        print(
            f"  Gradient Accumulation steps = {gradient_accumulation_steps}, Total optimization steps = {max_steps}, Num update steps per epoch = {num_update_steps_per_epoch}"
        )

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Training steps")

    train_iterator = iter(train_dataloader)
    criterion = nn.BCELoss()
    for step in range(max_steps):
        epoch = step // num_update_steps_per_epoch
        model = model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        noise
        # Training the discriminator
        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)

        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        loss = loss / gradient_accumulation_steps

        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0 or step == max_steps - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        accelerator.log({"train_loss": loss.detach()}, step=step)

        # complete an epoch

        if (step + 1) % num_update_steps_per_epoch == 0 or step == max_steps - 1:
            # evaluate
            eval_results = evaluate(
                accelerator, model, task, eval_dataloader, len(eval_dataloader.dataset)
            )
            accelerator.log(
                eval_results,
                step=epoch,
            )

        progress_bar.set_postfix(
            {
                "epoch": epoch + 1,
                "train_loss": loss.item(),
                "eval_acc": eval_results["eval_acc"],
            }
        )

        progress_bar.update(1)

    accelerator.wait_for_everyone()
    accelerator.end_training()

    if accelerator.is_local_main_process:
        print("Training completed")
