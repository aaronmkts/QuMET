import pennylane as qml
import numpy as np
from qas_env import QuantumArchSearchEnv
from ..utils import *
import gymnasium


class BasicNQubitEnv(QuantumArchSearchEnv): #gymnasium.Env,
    def __init__(self,
                 target: np.ndarray,
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        n_qubits = int(np.log2(len(target)))
        qubits = qml.wires.Wires(range(n_qubits))
        state_observables = get_default_observables(qubits)
        action_gates = get_default_gates(qubits)
        super(BasicNQubitEnv,
              self).__init__(target, qubits, state_observables, action_gates,
                             fidelity_threshold, reward_penalty, max_timesteps)


class BasicTwoQubitEnv(BasicNQubitEnv):
    def __init__(self,
                 target: np.ndarray = get_bell_state(),
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        assert len(target) == 4, 'Target must be of size 4'
        super(BasicTwoQubitEnv, self).__init__(target, fidelity_threshold,
                                               reward_penalty, max_timesteps)
        #self.num = 1
        #self.ob_space = self.observation_space

class BasicThreeQubitEnv(BasicNQubitEnv):
    def __init__(self,
                 target: np.ndarray = get_ghz_state(3),
                 fidelity_threshold: float = 0.95,
                 reward_penalty: float = 0.01,
                 max_timesteps: int = 20):
        assert len(target) == 8, 'Target must be of size 8'
        super(BasicThreeQubitEnv, self).__init__(target, fidelity_threshold,
                                                 reward_penalty, max_timesteps)