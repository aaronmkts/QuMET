import sys
from contextlib import closing
from io import StringIO
from typing import Dict, List, Optional, Union

import pennylane as qml
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import gymnasium
from gymnasium.utils import seeding

############# STATE PREPERATION TOOLS #############

class QuantumArchSearchEnv(gymnasium.Env):
    metadata = {'render_modes': ['ansi', 'human']}

    def __init__(
        self,
        target: np.ndarray,
        qubits: List[qml.wires.Wires],
        state_observables: List[qml.operation],
        action_gates: List[qml.operation],
        fidelity_threshold: float,
        reward_penalty: float,
        max_timesteps: int,
        error_observables: Optional[float] = None,
        error_gates: Optional[float] = None,
    ):
        super(QuantumArchSearchEnv, self).__init__()

        # set parameters
        self.target = target
        self.qubits = qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates

        # set environment
        self.target_density = target * np.conj(target).T
        #self.simulator = qml.device('default.qubit', wires=len(self.qubits))
        self.device = qml.device('default.qubit', wires=len(self.qubits))
        self.circuit = qml.QNode(self._get_cirq, self.device)

        # set spaces
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.seed()


    def __str__(self):
        desc = 'QuantumArchSearchEnv('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target)
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        self.circuit_gates = []
        return self._get_obs()

    
    def _get_cirq(self, maybe_add_noise=False):
        ''' 
        Research/ understand how pennylane load circuits, 'QNODE'
        -More time define a circuit before execution, youll need to see how 
        to iteratively add gates ???

        '''  
        for gate in self.circuit_gates:
            qml.apply(gate)
            
            if maybe_add_noise and (self.error_gates is not None):
                noise_gate = qml.DepolarizingChannel(self.error_gates, wires=gate.qubits) 
                qml.apply(noise_gate)

        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = qml.BitFlip(self.error_observables, wires=len(self.qubits)) 
            qml.apply(noise_observable)

        return self.circuit

    def _get_obs(self):
        '''
        simply executing the circuit will be enough to get the expectation
        '''
        circuit = self._get_cirq(maybe_add_noise=False)#True
        obs = self.circuit(
            circuit, observables=self.state_observables)
        obs = circuit
        return np.array(obs).real
    

    def _get_fidelity(self):
        circuit = self._get_cirq(maybe_add_noise=True)#
        pred = self.simulator.simulate(circuit).final_state_vector#
        inner = np.inner(np.conj(pred), self.target)
        fidelity = np.conj(inner) * inner
        return fidelity.real

    def step(self, action):

        # update circuit
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)

        # compute observation
        observation = self._get_obs()

        # compute fidelity
        fidelity = self._get_fidelity()

        # compute reward
        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        # check if terminal
        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)

        # return info
        info = {'fidelity': fidelity, 'circuit': self._get_cirq()}

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
            

def get_default_gates(qubits: List[qml.wires.Wires]) -> List[qml.operation]:
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

def get_default_observables(qubits: List[qml.wires.Wires]) -> List[qml.operation]:
    observables = []
    for qubit in qubits:
        observables += [
            qml.PauliX(wires=qubit),
            qml.PauliY(wires=qubit),
            qml.PauliZ(wires=qubit),
        ]
    return observables


def get_bell_state() -> np.ndarray: # Generalise to N qubits?
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target