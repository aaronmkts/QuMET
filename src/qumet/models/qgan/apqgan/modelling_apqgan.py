import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
from typing import Dict
from ...networks.utils import FeatureExtractor


logger = getLogger(__name__)

image_shape = (1, 28, 28) 

config = {
    "encoder": {
        "z_dim": 7,
        "enc_out_dim": 128,
        "image_shape": image_shape,
    },
    "generator": {
        "device": "default.qubit",
        "n_qubits": 7,
        "n_a_qubits": 1,
        "shots": 10000,
        "depth": 6,
        "q_delta": 1,
        "diff_method": "best",
        "n_generators": 16,
        "q_delta": 1,
        'patch_shape': (7, 7),
        'image_shape': image_shape,
    },
    "discriminator": {
        "image_shape": image_shape,
        "return_features": False,
        }, 
}

class Encoder(nn.Module):
    
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        for key, value in config.items():
            setattr(self, key, value)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512), nn.LeakyReLU(), 
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, self.enc_out_dim), nn.LeakyReLU(0.2, inplace=True)
        )

        self.hidden2mu = nn.Linear(self.enc_out_dim, self.z_dim)
        self.hidden2log_var = nn.Linear(self.enc_out_dim, self.z_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        hidden = self.encoder(x)
        mu, log_var = self.hidden2mu(hidden), self.hidden2log_var(hidden)

        return mu, log_var
    
    def reparametrize(self, mu, log_var):
        sigma = torch.exp(log_var / 2)
        z = torch.randn_like(sigma)
        return mu + sigma * z
    

class QuantumGenerator(nn.Module):
    def __init__(self,  config):
        super(QuantumGenerator, self).__init__()
        
        for key, value in config.items():
            setattr(self, key, value)

        self.q_device = qml.device(self.device, wires= self.n_qubits)
        self.params = nn.ParameterList([nn.Parameter(torch.rand(self.depth, self.n_qubits, 3), 
                                                     requires_grad=True) for _ in range(self.n_generators)])
        self.qnode = qml.QNode(self.circuit, self.q_device, interface="torch")

        
    def partial_measure_and_postprocess(self, noise, weights):
        # Non-linear Transform
        probs = self.qnode(noise, weights)
        post_measurement_probs = probs[: (2 ** (self.n_qubits - self.n_a_qubits))]
        post_measurement_probs /= torch.sum(probs)

        # Post-Processing

        post_processed_patch = ((post_measurement_probs / torch.max(post_measurement_probs)) - 0.5) * 2
        return post_processed_patch

    def circuit(self, latent_vector, weights):
        for i in range(self.n_qubits):
            qml.RY(latent_vector[i], wires=i)
        
        for i in range(self.depth):
            for j in range(self.n_qubits):
                qml.Rot(*weights[i][j], wires=j)

            for j in range(self.n_qubits-1):
                qml.CNOT(wires=[j, j+1])
        
        return qml.probs(wires=list(range(self.n_qubits)))
    
    def forward(self, x):
            
            special_shape = bool(self.patch_shape[0]) and bool(self.patch_shape[1])
            patch_size = 2 ** (self.n_qubits - self.n_a_qubits)
            image_pixels = self.image_shape[2] ** 2
            pixels_per_patch = image_pixels // self.n_generators
            if special_shape and self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
                raise ValueError("patch shape and patch size dont match!")
            output_images = torch.Tensor(x.size(0), 0)

            for sub_generator_param in self.params:
                patches = torch.Tensor(0, pixels_per_patch)
                for item in x:
                    sub_generator_out = self.partial_measure_and_postprocess(item, sub_generator_param).float().unsqueeze(0)
                    if pixels_per_patch < patch_size:
                        sub_generator_out = sub_generator_out[:,:pixels_per_patch]
                    patches = torch.cat((patches, sub_generator_out))
                output_images = torch.cat((output_images, patches), 1)

            if special_shape:
                final_out = torch.zeros(x.size(0), *self.image_shape)
                for i,img in enumerate(output_images):
                    for patches_done, j in enumerate(range(0, img.shape[0], pixels_per_patch)):
                        patch = torch.reshape(img[j:j+pixels_per_patch], self.patch_shape)
                        starting_h = ((patches_done * self.patch_shape[1]) // self.image_shape[2]) * self.patch_shape[0]
                        starting_w = (patches_done * self.patch_shape[1]) % self.image_shape[2]
                        final_out[i, 0, starting_h:starting_h+self.patch_shape[0], starting_w:starting_w+self.patch_shape[1]] = patch
            else:
                final_out = output_images.view(output_images.shape[0], *self.image_shape)
            return final_out

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        for key, value in config.items():
            setattr(self, key, value)

        if self.return_features:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = lambda x: x

        self.discriminator = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512), nn.LeakyReLU(0.1),
            nn.Linear(512, 256), self.feature_extractor(nn.LeakyReLU(0.2, inplace=True)),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(input.shape[0], -1)

        if self.return_features:
            self.feature_extractor.clean()
            output = self.discriminator(input)
            features = torch.cat(
                [torch.ravel(x) for x in self.feature_extractor.features]
            )
            return output, features
        else:
            output = self.discriminator(input)
            return output
    
class APQGAN(nn.Module):
    def __init__(self, config, task):
        super(APQGAN, self).__init__()

        encoder_config = config["encoder"]
        generator_config = config["generator"]  
        discriminator_config = config["discriminator"]


        self.encoder = Encoder(encoder_config)
        self.generator = QuantumGenerator(generator_config)
        self.discriminator = Discriminator(discriminator_config)

    def forward(self,z):
        output = self.generator(z)
        return output
    
    def vae_forward(self, imgs):
        mu, log_var = self.encoder(imgs)
        z = self.encoder.reparametrize(mu, log_var)
        recon_images = self.generator(z)
        return mu, log_var, z, recon_images
    
    def discriminate(self, x):
        return self.discriminator(x)
    

# ---------------------------------------
# SSQGAN
# ---------------------------------------


def _apqgan(config, task: str) -> APQGAN:

    model = APQGAN(config, task)
    return model


def get_apqgan(info: Dict) -> APQGAN:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into SSPQGAN")
    return _apqgan(config=config, task=task)