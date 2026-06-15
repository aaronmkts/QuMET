import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, z_dim=7, enc_out_dim=128, image_shape=(1, 28, 28)):
        super().__init__()

        self.image_shape = image_shape
        self.z_dim = z_dim
        self.enc_out_dim = enc_out_dim

        # Convolutional layers to progressively reduce the spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.image_shape[0],
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        # Calculate the output dimension after convolutional layers
        conv_out_dim = self._get_conv_out_dim()

        # Fully connected layer to map to the latent space
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, self.enc_out_dim), nn.LeakyReLU(0.1)
        )

        self.hidden2mu = nn.Linear(self.enc_out_dim, self.z_dim)
        self.hidden2log_var = nn.Linear(self.enc_out_dim, self.z_dim)

        # Apply LeCun initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # LeCun initialization
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
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
