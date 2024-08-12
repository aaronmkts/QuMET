from functools import partial
import pennylane as qml


# noises = [None, qml.PhaseFlip, qml.PhaseDamping, qml.DepolarizingChannel, qml.AmplitudeDamping]



def add_noise_to_circuit(circuit, noise_dict, prob=0.01):
    """
    Adds noise to the circuit passed here, after each instance of the gate specified in the noise_dict.

    Args:
        circuit: The quantum circuit to which the noise is to be added.
        noise_dict: Dict {qml.noise:qml.gate} -> A dictionary where the keys are noise functions and the values are the corresponding gates to which the noise is applied.
        prob: The probability parameter for the noise.

    Returns:
        The quantum circuit with noise added.
    """
    noise_model_dict = {}

    for noise, gate in noise_dict.items():
        c0 = qml.noise.op_eq(gate)  # Find every instance of the specified gate
        n0 = qml.noise.partial_wires(noise, prob)
        noise_model_dict[c0] = n0  # Add the condition and noise operation to the dictionary

    noise_model = qml.NoiseModel(noise_model_dict)

    return partial(qml.transforms.add_noise, noise_model=noise_model)(circuit)

