"Hybrid classical-quantum generative adversial network configuration"

from typing import Union
import pennylane as qml


class QGCD_Binary_Config:
    def __init__(
        self,
        device = 'default.qubit',
        shots = 1,
        n_qubits= 16,
        input_size = 2,
        n_a_qubits=0,
        depth=4,
        q_delta=1,
        diff_method="parameter-shift",
        batch_ops=False,  # GPU options
        mpi=False,  # Distribution across nodes
        **kwargs,
    ):
        self.input_size = n_qubits
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.depth = depth
        self.q_delta = q_delta
        self.device = device
        self.shots = shots
        self.diff_method = diff_method if (batch_ops and mpi) == False else "best"
        self.batch_ops = batch_ops
        self.mpi = mpi
