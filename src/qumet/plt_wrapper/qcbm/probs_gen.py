import torch
import torch.nn as nn
from torchmetrics import KLDivergence
import numpy as np
from ..base import WrapperBase, ValidationResult



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
        

    def training_step(self, batch):
     
        py = batch.reshape(-1,)
        
        px = self.model()
        loss = self.criterion(px, py)

        kl_div = self.entropy_val(py.unsqueeze(0), px.unsqueeze(0)).detach()

        self.log("mmd_loss", loss, prog_bar=True)
        self.log('val_kl_epoch', kl_div,  prog_bar=True)

        return loss
    
    def validation_step(self, batch):
        py = batch.reshape(-1,)
        px = self.model()
        


        return ValidationResult(real_distribution= py, fake_distribution= px)

   
    
    def configure_optimizers(self):
        match self.optimizer.lower():
            case "adam":
                b1, b2, = 0.777, 0.999

                optG = torch.optim.Adam(self.model.generator.parameters(), 
                                        lr=self.learning_rate, 
                                        weight_decay=self.weight_decay,
                                        betas=(b1, b2))
        return [optG]
