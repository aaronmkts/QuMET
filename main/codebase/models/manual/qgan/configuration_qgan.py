
"Hybrid classical-quantum generative adversial network configuration"

from typing import Union
import pennylane as qml 


class QmlMixin:
    """Mixin for models built on top of Pennylane (QML)"""

    _device: Union[str, qml.Device]
    _n_qubits: int

    def _set_qml_device(self,
                        device: Union[str, qml.Device]):
        """
        Internal method to set a pennylane device according to its type

        Args:
            The backend to set. Can be a :class:`~pennylane.Device` or a string
            (valid name of the backend)

        """
        if isinstance(device, qml.Device):
            n_wires = len(device.wires)
            if n_wires != self._n_qubits:
                raise ValueError(
                    f"Invalid number of wires for backend {device.name}. "
                    f"Expected {self._n_qubits}, got {n_wires}"
                )
            self._device = device
        else:
            # shots left as default (1000)
            self._device = qml.device(device, wires=self._n_qubits)

    @property
    def device(self) -> qml.Device:
        return self._device

    @device.setter
    def device(self, backend: Union[str, qml.Device]):
        self._device = backend

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits: int):
        self._n_qubits = n_qubits




class QganConfig():
    def __init__(
        self, 
        input_size = 0,
        n_qubits = 2,
        n_a_qubits = 0,
        depth = 1,
        q_delta = 1,
        device = "default.qubit",
        diff_method = "parameter-shift",
        batch_ops = False, #GPU options
        mpi = False, #Distribution across nodes
        **kwargs,
    ):
        
        self.input_size = input_size
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.depth = depth
        self.q_delta = q_delta
        self.device = device
        self.diff_method = (diff_method if (batch_ops and mpi) == False else "adjoint")
        self.batch_ops = batch_ops
        self.mpi = mpi
