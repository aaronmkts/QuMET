
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the flattened images saved previously.
# This file should contain an array of shape (n_samples, 784)
np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

fake_images_flat = np.load("fake_images_2component.npy")

# Reshape the flattened images into MNIST format (28x28)
fake_images = fake_images_flat.reshape(-1, 28, 28)

# Visualize a few samples
n_samples_to_show = 16  # Number of images to display
sample_indices = np.random.choice(fake_images.shape[0], n_samples_to_show, replace=False)

# Create a grid of subplots (e.g., 4x4 grid for 16 images)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_images[sample_indices[i]], cmap='gray')
    ax.axis('off')
    
plt.suptitle("Sample Fake MNIST Images from 2-component GMM")
plt.tight_layout()
plt.show()