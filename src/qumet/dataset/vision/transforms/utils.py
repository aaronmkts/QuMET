import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms as tv_transforms
import torch 

def filter_by_labels(dataset, labels: list):
    """
    Filters the dataset to only include specified labels.
    """
    X_data = dataset.data
    Y_data = dataset.targets

    # Create a mask for the desired labels
    mask = np.isin(Y_data, labels)
    
    # Filter the data and targets
    dataset.data = X_data[mask]
    dataset.targets = Y_data[mask]

    return dataset

def apply_pca(dataset, n_features: int):
    """
    Applies PCA to the dataset to reduce the number of features.
    """
    X_data = dataset.data.numpy()  # Convert to numpy array if necessary

    # Flatten the data if it's not already flattened
    X_flat_data = X_data.reshape(X_data.shape[0], -1)
    
    pca = PCA(n_components=n_features)
    X_data_pca = pca.fit_transform(X_flat_data)

    dataset.data = X_data_pca

    return dataset
