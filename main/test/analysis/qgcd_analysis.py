import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main"
     )
    )
import torch 
import torch.nn as nn
from codebase.dataset.manual import TwoDGaussianDataset, TwoDRingGaussianDataset, TwoDGridGaussianDataset
from codebase.dataset.manual import GaussianDataset
from codebase.models.qgan.qgcd_binary.modelling_qgan_binary import Binary_Generator, Binary_Discriminator
from codebase.models.qgan.qgcd_binary.configuration_qgan_binary import QGCD_Binary_Config

def main():
    '''
    config = QGCD_Binary_Config()
    generator = Binary_Generator(config = config)
    discriminator = Binary_Discriminator(config = config)
    input_ = torch.tensor([])
    output = generator(input_)
    probs_ = discriminator(output)

    
    def binary(x):
        bits = 16
        mask = 2**torch.arange(bits-1, -1, -1)
        x = torch.tensor([x])
        bitstring = x.bitwise_and(mask).ne(0).float()
        return bitstring
    bitstring = binary(6)
    print(bitstring)
    ''' 
    dataset = TwoDGridGaussianDataset()
    x, y = dataset.__getitem__(index = 5)
  
if __name__ == "__main__":
    main()
