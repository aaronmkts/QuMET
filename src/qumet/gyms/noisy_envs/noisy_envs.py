import numpy as np
import pennylane as qml

from ...gyms.tools import (
    QuantumArchSearchEnv,
    get_bell_state,
    get_default_gates,
    get_default_observables,
    get_ghz_state,
)


class NoisyNQubitEnv(QuantumArchSearchEnv):
    def __init__(
        self,
        target: np.ndarray,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        n_qubits = int(np.log2(len(target)))
        qubits = qml.wires.Wires(range(n_qubits))
        state_observables = get_default_observables(qubits)
        action_gates = get_default_gates(qubits)
        super().__init__(
            target,
            qubits,
            state_observables,
            action_gates,
            fidelity_threshold,
            reward_penalty,
            max_timesteps,
            error_observables=error_rate,
            error_gates=error_rate,
        )


class NoisyTwoQubitEnv(NoisyNQubitEnv):
    def __init__(
        self,
        target: np.ndarray = get_bell_state(),
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        assert len(target) == 4, "Target must be of size 4"
        super().__init__(
            target, fidelity_threshold, reward_penalty, max_timesteps, error_rate
        )


class NoisyThreeQubitEnv(NoisyNQubitEnv):
    def __init__(
        self,
        target: np.ndarray = get_ghz_state(3),
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        assert len(target) == 8, "Target must be of size 8"
        super().__init__(
            target, fidelity_threshold, reward_penalty, max_timesteps, error_rate
        )
