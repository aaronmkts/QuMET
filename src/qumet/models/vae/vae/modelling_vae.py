import torch
from torch import nn
from logging import getLogger
from typing import Dict

logger = getLogger(__name__)
#Input img -> Hidden dim -> mean, std -> Parameterization trick -> decoder -> Output img

config = { "input_dim": 784, 
        "z_dim": 1024,
        "input_channels": 1,
        "input_height": 28, 
        "input_width": 28}   

class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)
    
class VAE(nn.Module):
    def __init__(self, config: Dict, task: str):
        
        input_channels = config["input_channels"]
        input_height = config["input_height"]
        input_width = config["input_width"]
        input_dim = config["input_dim"]
        z_dim = config["z_dim"]
        self.z_dim = z_dim
        
        super(VAE, self).__init__()
        
        ENC_OUT_DIM = 128

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels*input_height*input_width, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, ENC_OUT_DIM)
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, input_channels*input_height*input_width),
            nn.Sigmoid(),
            Stack(input_channels, input_height, input_width),
        )

        self.hidden2mu = nn.Linear(ENC_OUT_DIM, z_dim)
        self.hidden2log_var = nn.Linear(ENC_OUT_DIM, z_dim)
        
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)

        return mu, log_var
    

    def decode(self, z):
        
        x = self.decoder(z)
        return x #Due to MNIST Normalization [0,1]-> sigmoid
    
    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model

        sigma = torch.exp(0.5*log_var).sqrt() # standard deviation
        z = torch.randn_like(sigma)
        return mu + sigma*z
    
    def forward(self, x):
        
        mu, log_var = self.encode(x)
        std = torch.exp(log_var / 2).sqrt() # standard deviation
        #Sample from distribution
        z = torch.distributions.Normal(mu, std).rsample()
        #Push sample through decoder
        x_hat = self.decode(z)

        return mu, std, z, x_hat
    
def _vae(config, task: str) -> VAE:

    model = VAE(config, task)
    return model


def get_vae(info: Dict) -> VAE:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into QGCD_PROBS_GAN ")
    return _vae(config=config, task=task)
