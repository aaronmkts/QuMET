import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"src"
     )
    )
import torch 
import torch.nn as nn

import itertools
from itertools import product
from qumet.dataset import QuMETDataModule
from qumet.models import get_model, get_model_info
from qumet.dataset import get_dataset, get_dataset_info
import torch.optim as optim

import numpy as np

import torch
import pennylane as qml



def main():
    # Seed the random number generators for reproducibility
    seed = 89
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if MPS device is available
    device = torch.device("cpu")
    print(f'Using device: {device}')

    class MMD:

        def __init__(self, scales, space):
            gammas = 1 / (2 * (scales**2))
            sq_dists = np.abs(space[:, None] - space[None, :]) ** 2
            self.K = sum(np.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
            self.K = torch.tensor(self.K, dtype=torch.float64).to(device)
            self.scales = scales

        def k_expval(self, px, py):
            return torch.matmul(px, torch.matmul(self.K, py))

        def __call__(self, px, py):
            pxy = px - py
            return self.k_expval(pxy, pxy)

    n = 3
    n_qubits = n**2
    def get_bars_and_stripes(n): #correct
        bitstrings = [list(np.binary_repr(i, n))[::-1] for i in range(2**n)]
        bitstrings = np.array(bitstrings, dtype=int)

        stripes = bitstrings.copy()
        stripes = np.repeat(stripes, n, 0)
        stripes = stripes.reshape(2**n, n * n)

        bars = bitstrings.copy()
        bars = bars.reshape(2**n * n, 1)
        bars = np.repeat(bars, n, 1)
        bars = bars.reshape(2**n, n * n)
        return np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))
    
    data = get_bars_and_stripes(n)
    bitstrings = []
    nums = []
    for d in data:
        bitstrings += ["".join(str(int(i)) for i in d)]
        nums += [int(bitstrings[-1], 2)]
    probs = np.zeros(2**n_qubits)
    probs[nums] = 1 / len(data)
    probs = torch.tensor(probs, dtype=torch.float64).to(device)  # Ensure probs is a Float tensor

    bandwidth = np.array([0.25])
    space = np.arange(2**n_qubits)

    mmd = MMD(bandwidth, space)




    class QCBM:

        def __init__(self, circ, mmd, py):
            self.circ = circ
            self.mmd = mmd
            self.py = py.clone().detach()

        def mmd_loss(self, params):
          
            px = self.circ(params)
            return self.mmd(px, self.py), px

        def kl_divergence(self, px):
            # Avoid division by zero and handle log(0) cases
            qcbm_probs = px.clone().detach()
            target_probs = self.py
            kl_div = -torch.sum(target_probs * torch.nan_to_num(torch.log(qcbm_probs / target_probs)))
            return kl_div
    
    dev = qml.device("default.qubit", wires=n_qubits)
    n_layers = 6
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = np.random.random(size=wshape)
    weights = torch.tensor(weights, requires_grad=True, dtype=torch.float64).to(device)

    @qml.qnode(dev, interface='torch', diff_method= 'backprop')
    def circuit(weights):
        qml.StronglyEntanglingLayers(
            weights=weights, ranges=[1] * n_layers, wires=range(n_qubits)
        )
        return qml.probs()

    
    qcbm = QCBM(circuit, mmd, probs)
    b1 , b2 = 0.777, 0.999
    optimizer = optim.Adam([weights], lr=0.1,  betas=(b1, b2))

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        loss, px = qcbm.mmd_loss(weights)
        loss.backward()
        optimizer.step()
        kl_div = qcbm.kl_divergence(px)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, KL Divergence: {kl_div.item()}')



if __name__ == "__main__":
    main()
