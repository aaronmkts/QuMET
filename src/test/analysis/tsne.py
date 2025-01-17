import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,".."
    )
)

def main():
    import torch
    from torchvision import datasets, transforms
    from sklearn.manifold import TSNE
    
    # Load MNIST data using torchvision
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
  
    # Extract data and labels
    X = train_dataset.data
    y = train_dataset.targets

    # Flatten the images from 28x28 to 784
    X = X.view(-1, 28*28).numpy()
    y = y.numpy()

    # ---- NEW PART: Subselect images that correspond to labels 1, 3, and 5 ----
    mask = np.isin(y, [0, 1, 9])
    X = X[mask]
    y = y[mask]

    # Subselect 2000 random samples to save time (optional; can adjust as needed)
    np.random.seed(42)
    idx = np.random.choice(len(X), 2000, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    # Perform TSNE# Remap labels to a contiguous range
    unique_labels = np.unique(y_sub)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    y_mapped = np.array([label_mapping[label] for label in y_sub])

    # Perform TSNE
    tsne = TSNE(n_components=2, random_state=32)
    X_tsne = tsne.fit_transform(X_sub)

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_mapped,
        cmap=plt.cm.get_cmap("tab10", len(unique_labels)),
        alpha=0.6
    )

    # Color bar with correct tick labels
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(len(unique_labels)))
    cbar.set_ticklabels(unique_labels)  # Original labels [0, 1, 9]

    plt.title("t-SNE visualization of MNIST (labels: 0, 1, 9)")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.show()

if __name__ == "__main__":
    main()
