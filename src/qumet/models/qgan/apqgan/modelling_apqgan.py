import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
from typing import Dict
from ...networks.utils import FeatureExtractor
from pennylane.qnn import TorchLayer as TorchConnector

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
        "depth": 12,
        "q_delta": 1,
        "diff_method": "best",
        "n_generators": 14,
        "q_delta": 1,
        'patch_shape': (2, 28),
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
        
        name = "encoder"
        encoder_config = config[name]

        self.image_shape = encoder_config["image_shape"]
        self.enc_out_dim = encoder_config["enc_out_dim"]
        self.z_dim = encoder_config["z_dim"]


        # Convolutional layers to progressively reduce the spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.image_shape[0], out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )


        # Calculate the output dimension after convolutional layers
        conv_out_dim = self._get_conv_out_dim()


        # Fully connected layer to map to the latent space
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, self.enc_out_dim),
            nn.LeakyReLU(0.1)
        )

        self.hidden2mu = nn.Linear(self.enc_out_dim, self.z_dim)
        self.hidden2log_var = nn.Linear(self.enc_out_dim, self.z_dim)

             # Apply LeCun initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # LeCun initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def _get_conv_out_dim(self):
        # Calculate the flattened output size after the final convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.image_shape)
            output = self.encoder(dummy_input)
        return int(np.prod(output.size()))


    def forward(self, x):
        x = x.view(-1, *self.image_shape)
        conv_out = self.encoder(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        hidden = self.fc(conv_out)
        mu, log_var = self.hidden2mu(hidden), self.hidden2log_var(hidden)

        return mu, log_var
    
    def reparametrize(self, mu, log_var):
        sigma = torch.exp(log_var / 2)
        z = torch.randn_like(sigma)
        return mu + sigma * z
    

class QuantumGenerator(nn.Module):
    def __init__(self,  config):
        super(QuantumGenerator, self).__init__()
        name = "generator"
        generator_config = config[name]

        q_delta = config[name]["q_delta"]

        self.n_generators = generator_config["n_generators"]
        self.device = generator_config["device"]
        self.n_qubits = generator_config["n_qubits"]
        self.n_a_qubits = generator_config["n_a_qubits"]
        self.depth = generator_config["depth"]
        self.diff_method = generator_config["diff_method"]
        self.patch_shape = generator_config["patch_shape"]
        self.image_shape = generator_config["image_shape"]
        self.q_device = qml.device(self.device, wires= self.n_qubits)
        self._construct_quantum_layers()
   
    def _construct_quantum_layers(self):
        qnode = qml.QNode(self.circuit, self.q_device, interface="torch", diff_method=self.diff_method)
        weight_shapes = {"weights": (self.depth, self.n_qubits, 3)}
        self.q_layers = nn.ModuleList(
            [TorchConnector(qnode, weight_shapes) for _ in range(self.n_generators)]
        )

    def partial_trace_and_postprocess(self, noise, q_layer):
        probs = q_layer(noise)
        probsgiven0 = probs[:2**(self.n_qubits - self.n_a_qubits)]
        probsgiven0 /= torch.sum(probs)
        
        # Post-Processing
        post_processed_patch = ((probsgiven0 / torch.max(probsgiven0)) - 0.5) * 2
        return post_processed_patch
    
    def circuit(self, inputs, weights):

        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        for i in range(self.depth):
            for j in range(self.n_qubits):
                qml.Rot(*weights[i][j], wires=j)

            for j in range(self.n_qubits-1):
                qml.CNOT(wires=[j, j+1])
                
            qml.CNOT(wires=[self.n_qubits-1, 0])
        
        return qml.probs(wires=list(range(self.n_qubits)))
    
    def forward(self, x):
        special_shape = bool(self.patch_shape[0]) and bool(self.patch_shape[1])
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits )
        image_pixels = self.image_shape[2] ** 2
        pixels_per_patch = image_pixels // self.n_generators
        if special_shape and self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
            raise ValueError("patch shape and patch size dont match!")
        output_images = torch.Tensor(x.size(0), 0)

        for q_layer in self.q_layers:
            patches = torch.Tensor(0, pixels_per_patch)
            for item in x:
                sub_generator_out = self.partial_trace_and_postprocess(item, q_layer).float().unsqueeze(0)
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
        super().__init__()
        name = "discriminator"
        self.image_shape = config[name]["image_shape"]

        self.return_features = config[name]["return_features"]
        if self.return_features:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = lambda x: x


        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

             # Apply LeCun initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                # LeCun initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, input):
        N = input.shape[0]

        if self.return_features:
            self.feature_extractor.clean()
            x = input.view(N, -1)
            output = self.model(x)
            features = self.feature_extractor.features[0]
           # features = torch.cat(
           #     [torch.ravel(x) for x in self.feature_extractor.features]
           # )
            return output, features
        
        else:
            x = input.view(N, -1)
            output = self.model(x)
            return output

   
class APQGAN(nn.Module):
    def __init__(self, config, task):
        super(APQGAN, self).__init__()

        self.encoder = Encoder(config)
        self.generator = QuantumGenerator(config)
        self.discriminator = Discriminator(config)

    def forward(self,z):
        
        imgs = self.generator(z)
    
        return imgs
    
    def vae_forward(self, imgs):
        mu, log_var = self.encoder(imgs)
        z = self.encoder.reparametrize(mu, log_var)
        
        return mu, log_var, z
    
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