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
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

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

    # dataset
    data_module.prepare_data()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    test_dataloader = data_module.test_dataloader() # The entire dataset
    

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
    
    test_data = next(iter(test_dataloader))

    def adversarial_loss(input, target, w):
        bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
        weighted_loss = w * bce_loss
        total_loss = -torch.sum(weighted_loss)
        return total_loss


    real_labels = torch.full((data_module.batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((data_module.batch_size,), 0.0, dtype=torch.float, device=device)

    # Fixed noise allows us to visually track the generated images throughout training
    fixed_noise = torch.rand((256, 4), device=device) * math.pi / 2
   
    writer = SummaryWriter(save_path)

    if evaluate_before_training == True:
        fixed_fake_data = generator(fixed_noise).reshape(-1,1)
        writer.add_histogram('Distribution', fixed_fake_data.squeeze(), 0)
        writer.add_histogram('True Distribution', test_data.squeeze().numpy(), 0)

    for step in range(max_steps):
        epoch = step // num_update_steps_per_epoch

        try:
            real_data, coords = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            real_data, coords = next(train_iterator)
        

      

        # Generate fake-data using noise input
        noise = torch.tensor([])
        fake_data = generator(noise).reshape(-1,1)
        # configure samples
        disc_value = discriminator(coords)

        # Training the generator
        optG.zero_grad()
        errG = adversarial_loss(disc_value, real_labels, fake_data) #generator loss
        errG.backward(retain_graph = True)
        optG.step()

        # Training the discriminator
        optD.zero_grad()
        errD_real = adversarial_loss(disc_value, real_labels, real_data) #discriminator loss, real
        errD_fake = adversarial_loss(disc_value, fake_labels, fake_data.detach()) #discriminator loss, fake

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
                fixed_fake_data = generator(noise).reshape(-1,1)
                
                entropy_value = entropy(fixed_fake_data.detach().squeeze().numpy(), real_data.squeeze())
                
            writer.add_scalar('Relative_Entropy', entropy_value, step)
            #writer.add_histogram('Distribution', fixed_fake_data.squeeze(), epoch + 1)
  
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

