import torch
import torch.nn as nn
from torchmetrics import KLDivergence
import numpy as np
from ..base import WrapperBase



class MMD:

    def __init__(self, scales, space):
        gammas = 1 / (2 * (scales ** 2))
        sq_dists = np.abs(space[:, None] - space[None, :]) ** 2
        self.K = sum(np.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.K = torch.tensor(self.K, dtype=torch.float64)
        self.scales = scales

    def k_expval(self, px, py):
        return torch.matmul(px, torch.matmul(self.K, py))

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)


class QCBMProbsGenModelWrapper(WrapperBase):
    def __init__(
            self,
            model,
            dataset_info,
            learning_rate=1e-2,
            weight_decay=0.0,
            epochs=100,
            optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            optimizer=optimizer,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_qubits = self.model.generator.n_qubits
        self.entropy_val = KLDivergence()

        bandwidth = np.array([0.25])
        space = np.arange(2 ** self.n_qubits)
        self.criterion = MMD(bandwidth, space)
        self.noise = 1e-8
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optG = self.optimizers()
        probs = batch

        # optG = torch.optim.Adam([self.model.generator.weights], lr=0.1)
        self.toggle_optimizer(optG)

        optG.zero_grad()
        px = self.model.forward()
        loss = self.criterion(px, probs)
        loss.backward()
        optG.step()
        kl_div = self.model.generator.kl_divergence(px, probs)
        self.untoggle_optimizer(optG)


        self.log("mmd_loss", loss, on_epoch=True, prog_bar=True)
        self.log('kl_divergence', kl_div, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([self.model.generator.weights], lr=0.1)
        return self.optimizer
