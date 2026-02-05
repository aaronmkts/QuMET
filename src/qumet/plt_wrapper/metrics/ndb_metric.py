import numpy as np
import torch
from scipy.stats import norm
from sklearn.cluster import KMeans
from torchmetrics import Metric


class NDB_JSD_Metric(Metric):
    def __init__(
        self,
        number_of_bins=30,
        significance_level=0.05,
        z_threshold=None,
        whitening=False,
        max_dims=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.number_of_bins = number_of_bins
        self.significance_level = significance_level
        self.z_threshold = z_threshold
        self.whitening = whitening
        self.ndb_eps = 1e-6
        self.max_dims = max_dims

        # Add states to accumulate training and generated features
        self.add_state("training_features", default=[], dist_reduce_fx="cat")
        self.add_state("generated_features", default=[], dist_reduce_fx="cat")

        # Variables to be initialized later
        self.training_mean = None
        self.training_std = None
        self.bin_centers = None
        self.bin_proportions = None
        self.ref_sample_size = None
        self.used_d_indices = None

    def update(self, features: torch.Tensor, data_type: str):
        """
        Updates the metric state with new data.

        Args:
            features (torch.Tensor): The data features to update the state with.
            data_type (str): Type of data, either 'training' or 'generated'.
        """
        if data_type == "training":
            self.training_features.append(features)
        elif data_type == "generated":
            self.generated_features.append(features)
        else:
            raise ValueError("data_type must be 'training' or 'generated'")

    def compute(self):
        """
        Computes the NDB and JS metrics after all updates have been made.

        Returns:
            dict: A dictionary containing 'NDB' and 'JS' scores.
        """
        # Concatenate training and generated features
        training_samples = torch.cat(self.training_features, dim=0)
        generated_samples = torch.cat(self.generated_features, dim=0)

        training_samples_np = (
            training_samples.reshape(training_samples.size(0), -1).cpu().numpy()
        )
        generated_samples_np = (
            generated_samples.reshape(generated_samples.size(0), -1).cpu().numpy()
        )

        # Construct bins using training samples
        self.construct_bins(training_samples_np)

        # Assign generated samples to bins and compute metric
        n_generated = generated_samples_np.shape[0]
        generated_bin_proportions, _ = self.calculate_bin_proportions(
            generated_samples_np
        )

        # Compute different bins
        different_bins = self.two_proportions_z_test(
            self.bin_proportions,
            self.ref_sample_size,
            generated_bin_proportions,
            n_generated,
            significance_level=self.significance_level,
            z_threshold=self.z_threshold,
        )

        ndb = np.count_nonzero(different_bins)
        js = self.jensen_shannon_divergence(
            self.bin_proportions, generated_bin_proportions
        )

        return {"NDB": ndb, "JS": js}

    def construct_bins(self, training_samples):
        n, d = training_samples.shape
        k = self.number_of_bins

        if self.whitening:
            self.training_mean = np.mean(training_samples, axis=0)
            self.training_std = np.std(training_samples, axis=0) + self.ndb_eps
        else:
            self.training_mean = np.zeros(d)
            self.training_std = np.ones(d)

        if self.max_dims is None and d > 1000:
            self.max_dims = d // 6

        whitened_samples = (training_samples - self.training_mean) / self.training_std
        d_used = d if self.max_dims is None else min(d, self.max_dims)
        self.used_d_indices = np.random.choice(d, d_used, replace=False)

        # Perform KMeans clustering
        clusters = KMeans(n_clusters=k, max_iter=100, n_init="auto").fit(
            whitened_samples[:, self.used_d_indices]
        )

        bin_centers = np.zeros([k, d])

        for i in range(k):
            bin_centers[i, :] = np.mean(
                whitened_samples[clusters.labels_ == i, :], axis=0
            )

        # Organize bins by size
        _, label_counts = np.unique(clusters.labels_, return_counts=True)
        bin_order = np.argsort(-label_counts)
        self.bin_proportions = label_counts[bin_order] / np.sum(label_counts)
        self.bin_centers = bin_centers[bin_order, :]
        self.ref_sample_size = n

    def calculate_bin_proportions(self, samples):
        if self.bin_centers is None:
            raise ValueError(
                "Bins have not been constructed. Make sure to call construct_bins first."
            )
        n, d = samples.shape
        k = self.bin_centers.shape[0]
        D = np.zeros([n, k], dtype=samples.dtype)

        whitened_samples = (samples - self.training_mean) / self.training_std

        for i in range(k):
            D[:, i] = np.linalg.norm(
                whitened_samples[:, self.used_d_indices]
                - self.bin_centers[i, self.used_d_indices],
                ord=2,
                axis=1,
            )

        labels = np.argmin(D, axis=1)
        probs = np.zeros([k])
        label_vals, label_counts = np.unique(labels, return_counts=True)
        probs[label_vals] = label_counts / n
        return probs, labels

    @staticmethod
    def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
        # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
        # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        if z_threshold is not None:
            return np.abs(z) > z_threshold
        p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))  # Two-tailed test
        return p_values < significance_level

    @staticmethod
    def jensen_shannon_divergence(p, q):
        """
        Calculates the symmetric Jensen–Shannon divergence between the two PDFs.
        """
        m = (p + q) * 0.5
        return 0.5 * (
            NDB_JSD_Metric.kl_divergence(p, m) + NDB_JSD_Metric.kl_divergence(q, m)
        )

    @staticmethod
    def kl_divergence(p, q):
        """
        The Kullback–Leibler divergence.
        """
        eps = 1e-8
        p = p + eps
        q = q + eps
        return np.sum(p * np.log(p / q))
