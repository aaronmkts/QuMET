from typing import Dict, List, Optional, Union

import pennylane as qml
import numpy as np


"""def get_default_gates(
        qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    gates = []
    for idx, qubit in enumerate(qubits):
        next_qubit = qubits[(idx + 1) % len(qubits)]
        gates += [
            cirq.rz(np.pi / 4.)(qubit),
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
            cirq.H(qubit),
            cirq.CNOT(qubit, next_qubit)
        ]
    return gates"""

def get_default_gates(qubits: List[qml.wires.Wires]) -> List[qml.Operation]:
    gates = []
    n_qubits = len(qubits)
    for idx, qubit in enumerate(qubits):
        next_qubit = qubits[(idx + 1) % n_qubits]
        gates += [
            qml.RZ(np.pi / 4, wires=qubit),
            qml.PauliX(wires=qubit),
            qml.PauliY(wires=qubit),
            qml.PauliZ(wires=qubit),
            qml.Hadamard(wires=qubit),
            qml.CNOT(wires=[qubit, next_qubit])
        ]
    return gates


"""def get_default_observables(
        qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    observables = []
    for qubit in qubits:
        observables += [
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
        ]
    return observables"""

def get_default_observables(qubits: List[qml.wires.Wires]) -> List[qml.Operation]:
    observables = []
    for qubit in qubits:
        observables += [
            qml.PauliX(wires=qubit),
            qml.PauliY(wires=qubit),
            qml.PauliZ(wires=qubit),
        ]
    return observables


def get_bell_state() -> np.ndarray: # This doesn't use Cirq so I think we can keep it
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


def get_ghz_state(n_qubits: int = 3) -> np.ndarray: # Same with this one
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target