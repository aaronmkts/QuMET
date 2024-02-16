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
import toml
import sys
from transformers import get_scheduler
from codebase.tools import get_optimizer
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy
import os 
from pennylane.tape import QuantumTape, QuantumScript
from typing import Callable, Sequence, Union
import pennylane as qml
from pennylane.transforms.core import transform

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

def get_probs_list(bit_list, n_qubits):

        def _binary_to_int(bit_list):
            output = 0
            for bit in bit_list:
                output = output * 2 + bit
            return int(output)

        int_list = list(map(_binary_to_int,bit_list))

        count_dict = {integer: 0 for integer in range(2 ** n_qubits)}
        # Potentially replace for loop with: count_dict.update(Counter(int_list))
        for integer in int_list:
            count_dict[integer] += 1

        total_counts = sum(count_dict.values())
        probs_list = np.array([count / total_counts for count in count_dict.values()])

        return probs_list






def train(
    generator,
    discriminator,
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
    evaluate_before_training: bool = True
):
    
    if save_path is not None:
        # if save_path is None, the model will not be saved
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    # Enable CUDA device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    n_qubits = generator.n_qubits

    # dataset
    data_module.prepare_data()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    pred_dataloader = data_module.pred_dataloader() # The entire dataset
    

    # optimizers
    optG = get_optimizer(
            generator, generator_optimizer, generator_learning_rate, weight_decay
    )
    optD = get_optimizer(discriminator, discriminator_optimizer, discriminator_learning_rate, weight_decay)


    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps 
    )

    if max_steps == -1:
        max_steps = max_epochs * num_update_steps_per_epoch
    else:
        max_steps = min(max_steps, max_epochs * num_update_steps_per_epoch)
    
    #lr_scheduler for classical discriminator
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optD,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )

    experiment_config = {
        "model": type(generator).__name__,
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
    ''' 
    accelerator.init_trackers(save_path.replace("/", "_"), config=experiment_config)
    '''
    # training

    print("***** Running training *****")
    print(f"  Instantaneous batch size per device = {data_module.batch_size}")
    print(
            f"  Total train batch size & accumulation) = {data_module.batch_size  * gradient_accumulation_steps}"
        )
    print(
            f"  Gradient Accumulation steps = {gradient_accumulation_steps}, Total optimization steps = {max_steps}, Num update steps per epoch = {num_update_steps_per_epoch}"
        )

    progress_bar = tqdm(range(max_steps))
    progress_bar.set_description(f"Training steps")

    train_iterator = iter(train_dataloader)
    
    pred_probs , _ = next(iter(pred_dataloader))
    
    criterion =  nn.BCELoss()


    real_labels = torch.full((data_module.batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((data_module.batch_size,), 0.0, dtype=torch.float, device=device)

    # Fixed noise allows us to visually track the generated images throughout training

    samples = 1000
    fixed_noise = torch.rand((samples, n_qubits), device=device) * math.pi / 2
   
    writer = SummaryWriter(save_path)

    if evaluate_before_training == True:
        output = generator(fixed_noise)
        probs_list = get_probs_list(output, n_qubits)
        entropy_value = entropy(probs_list, pred_probs.squeeze())
        writer.add_scalar('Relative_Entropy', entropy_value, 0)
        
    for step in range(max_steps):
        epoch = step // num_update_steps_per_epoch

        try:
            probs, batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            probs, batch = next(train_iterator)

        real_data = batch 
        
        # Generate fake-data using noise input
        noise = torch.rand((data_module.batch_size, n_qubits), dtype = torch.float, device=device) * math.pi /2

        fake_data = generator(noise)
        
         # Training the generator
        optG.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = criterion(outD_fake, real_labels) #generator loss
        errG.backward(retain_graph = True)
        optG.step()

        # Training the discriminator
        optD.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)
        

        errD_real = criterion(outD_real, real_labels) #discriminator loss, real
        errD_fake = criterion(outD_fake, fake_labels) #discriminator loss, fake

        errD = (errD_real + errD_fake) /2
        errD.backward()
        optD.step()
        

        lr_scheduler.step()

        writer.add_scalars('train_loss', {'d_loss': errD.item(),
                                          'g_loss': errG.item()},
                                          step)
        


        
        # complete an epoch
        if (step + 1) % num_update_steps_per_epoch == 0 or step == max_steps - 1:
            # evaluate
            with torch.no_grad():
                #During evaluation turn PyTorch autograd off so we arent training with this data
                output = generator(fixed_noise)
                
            probs_list = get_probs_list(output, n_qubits)
            entropy_value = entropy(probs_list, pred_probs.squeeze())
            writer.add_scalar('Relative_Entropy', entropy_value, step)
  
        progress_bar.set_postfix(
            {
                "epoch": epoch + 1,
                "d_loss": errD.item(),
                'g_loss': errG.item(),
            }
        )
        progress_bar.update(1)

    print("Training completed")
    writer.close()

