"""VAE-GAN hybrid wrapper modules.

This module provides PyTorch Lightning wrappers for training hybrid VAE-GAN
models that combine variational autoencoders with adversarial training.
"""

from .vaeqgan_models import Encoder
from .vaeqgan_module import VAEGANWrapper
