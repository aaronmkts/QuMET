import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pennylane.qnn import TorchLayer as TorchConnector

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


image_shape = (1, 28, 28)
# fmt:0ff
config = {
    "discriminator": {"image_shape": image_shape},
    "generator": {
        "device": "default.qubit",
        "n_qubits": 7,
        "n_a_qubits": 1,
        "shots": 10000,
        "depth": 3,
        "q_delta": 1,
        "diff_method": "best",
        "n_generators": 16,
        "q_delta": 1,
        'patch_shape': (7, 7),
        'image_shape': image_shape,
    },
}

class PWQGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, config):
        super().__init__()
        
        name = "generator"
        generator_config = config[name]

        q_delta = config[name]["q_delta"]

        self.n_generators = generator_config["n_generators"]
        self.device = generator_config["device"]
        self.n_qubits = generator_config["n_qubits"]
        self.n_a_qubits = generator_config["n_a_qubits"]
        self.depth = generator_config["depth"]
        self.diff_method = generator_config["diff_method"]
        self.patch_shape = generator_config["patch_shape"]
        self.image_shape = generator_config["image_shape"]
        self.q_device = qml.device(self.device, wires= self.n_qubits)
        self._construct_quantum_layers()
   
    def _construct_quantum_layers(self):
        qnode = qml.QNode(self.circuit, self.q_device, interface="torch", diff_method=self.diff_method)
        weight_shapes = {"weight": (self.depth * self.n_qubits * 3, self.n_qubits),
                         "bias": (self.depth * self.n_qubits * 3, self.n_qubits)}
        self.q_layers = nn.ModuleList(
            [TorchConnector(qnode, weight_shapes) for _ in range(self.n_generators)]
        )

    def partial_trace_and_postprocess(self, noise, q_layer):
        probs = q_layer(noise)
        probsgiven0 = probs[:2**(self.n_qubits - self.n_a_qubits)]
        probsgiven0 /= torch.sum(probs)
        
        # Post-Processing
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven
    
    def circuit(self, inputs, weight, bias):
        
        #appply affine transformation
        theta = torch.matmul(weight, inputs) + bias
        theta = torch.reshape(theta, (self.depth, self.n_qubits, 3))

        for i in range(self.depth):
            for j in range(self.n_qubits):
                qml.Rot(*theta[i][j], wires=j)

            for j in range(self.n_qubits-1):
                qml.CNOT(wires=[j, j+1])
            qml.Barrier()
        return qml.probs(wires=list(range(self.n_qubits)))
    
    def forward(self, x):
        special_shape = bool(self.patch_shape[0]) and bool(self.patch_shape[1])
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits )
        image_pixels = self.image_shape[2] ** 2
        pixels_per_patch = image_pixels // self.n_generators
        if special_shape and self.patch_shape[0] * self.patch_shape[1] != pixels_per_patch:
            raise ValueError("patch shape and patch size dont match!")
        output_images = torch.Tensor(x.size(0), 0)

        for q_layer in self.q_layers:
            patches = torch.Tensor(0, pixels_per_patch)
            for item in x:
                sub_generator_out = self.partial_trace_and_postprocess(item, q_layer).float().unsqueeze(0)
                if pixels_per_patch < patch_size:
                    sub_generator_out = sub_generator_out[:,:pixels_per_patch]
                patches = torch.cat((patches, sub_generator_out))
            output_images = torch.cat((output_images, patches), 1)

        if special_shape:
            final_out = torch.zeros(x.size(0), *self.image_shape)
            for i,img in enumerate(output_images):
                for patches_done, j in enumerate(range(0, img.shape[0], pixels_per_patch)):
                    patch = torch.reshape(img[j:j+pixels_per_patch], self.patch_shape)
                    starting_h = ((patches_done * self.patch_shape[1]) // self.image_shape[2]) * self.patch_shape[0]
                    starting_w = (patches_done * self.patch_shape[1]) % self.image_shape[2]
                    final_out[i, 0, starting_h:starting_h+self.patch_shape[0], starting_w:starting_w+self.patch_shape[1]] = patch
        else:
            final_out = output_images.view(output_images.shape[0], *self.image_shape)
        return final_out
    

        
if __name__ == "__main__":

    def center(coord, n):
        return np.array(coord) / n + 0.5 / n

    def compute_discretization(n_qubits, n_dim):
        format_string = "{:0" + str(n_qubits) + "b}"
        n = 2 ** (n_qubits // n_dim)
        dict_bins = {}
        from itertools import product

        
        for k, coordinates in enumerate(product(range(n), repeat=n_dim)):
            dict_bins.update({
                format_string.format(k): [coordinates, center(coordinates, n)]
            })
        return dict_bins
    

    x = compute_discretization(8, 2)
    breakpoint()

     '''

    df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'})
    df = df.scalars

    gauss = df[df['dir_name'] == 'final_result/gaussian_prior/2000_samples/software/tensorboard/lightning_logs/version_0']
    uniform = df[df['dir_name'] == 'final_result/uniform_prior/2000_samples/software/tensorboard/lightning_logs/version_0']
    
    plt.scatter(gauss['step'], gauss['val_log/val_mse_reduction'])
    plt.scatter(uniform['step'], uniform['val_log/val_mse_reduction'])

    # Set custom font size for x and y ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
  
    plt.xlabel('Iterations' , fontsize=16)
    plt.ylabel('Wasserstein Distance', fontsize=16)
    plt.legend(['Gaussian', 'Uniform'])

    plt.show()
    breakpoint()