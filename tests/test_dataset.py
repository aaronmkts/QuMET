"""Tests for qumet.dataset module — QuMETDataModule, AVAILABLE_DATASETS, get_dataset_info, get_dataset."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
from torch.utils.data import DataLoader, TensorDataset

from qumet.dataset import (
    QuMETDataModule,
    AVAILABLE_DATASETS,
    get_dataset_info,
    get_dataset,
)
from qumet.dataset.utils import QuMETDatasetInfo, DatasetSource, DatasetSplit


# ---------------------------------------------------------------------------
# get_dataset_info
# ---------------------------------------------------------------------------

class TestGetDatasetInfo:
    """Tests for get_dataset_info()."""

    def test_returns_info_for_known_vision_dataset(self):
        """get_dataset_info returns a QuMETDatasetInfo for a known vision dataset."""
        info = get_dataset_info("mnist")
        assert isinstance(info, QuMETDatasetInfo)
        assert info.name == "mnist"
        assert info.dataset_source == DatasetSource.TORCHVISION

    def test_returns_info_for_known_manual_dataset(self):
        """get_dataset_info returns info for a known manual dataset."""
        info = get_dataset_info("bars_and_stripes")
        assert isinstance(info, QuMETDatasetInfo)
        assert info.name == "bars_and_stripes"
        assert info.dataset_source == DatasetSource.MANUAL

    def test_case_insensitive_lookup(self):
        """get_dataset_info is case-insensitive."""
        info_lower = get_dataset_info("mnist")
        info_upper = get_dataset_info("MNIST")
        assert info_lower.name == info_upper.name

    def test_raises_valueerror_for_unknown_dataset(self):
        """get_dataset_info raises ValueError for unsupported dataset names."""
        with pytest.raises(ValueError, match="not supported"):
            get_dataset_info("nonexistent_dataset_xyz")


# ---------------------------------------------------------------------------
# AVAILABLE_DATASETS
# ---------------------------------------------------------------------------

class TestAvailableDatasets:
    """Tests for AVAILABLE_DATASETS constant."""

    def test_is_list_of_strings(self):
        """AVAILABLE_DATASETS is a list of strings."""
        assert isinstance(AVAILABLE_DATASETS, list)
        assert all(isinstance(d, str) for d in AVAILABLE_DATASETS)

    def test_contains_vision_datasets(self):
        """AVAILABLE_DATASETS includes vision datasets."""
        assert "mnist" in AVAILABLE_DATASETS
        assert "fashion_mnist" in AVAILABLE_DATASETS
        assert "cifar10" in AVAILABLE_DATASETS

    def test_contains_manual_datasets(self):
        """AVAILABLE_DATASETS includes manual datasets."""
        assert "bars_and_stripes" in AVAILABLE_DATASETS
        assert "2d_gaussian" in AVAILABLE_DATASETS

    def test_all_entries_have_valid_info(self):
        """Every entry in AVAILABLE_DATASETS returns valid info."""
        for name in AVAILABLE_DATASETS:
            info = get_dataset_info(name)
            assert isinstance(info, QuMETDatasetInfo)
            assert info.name == name


# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------

class TestGetDataset:
    """Tests for get_dataset()."""

    def test_raises_valueerror_for_unknown_dataset(self):
        """get_dataset raises ValueError for unsupported dataset names."""
        with pytest.raises(ValueError, match="not supported"):
            get_dataset(
                name="nonexistent_dataset_xyz",
                split="train",
                transform=None,
                discretise=False,
                n_samples=100,
                n_qubits=4,
            )

    def test_raises_assertion_error_for_invalid_split(self):
        """get_dataset asserts on invalid split values."""
        with pytest.raises(AssertionError, match="Unknown split"):
            get_dataset(
                name="mnist",
                split="invalid_split",
                transform=None,
                discretise=False,
                n_samples=100,
                n_qubits=4,
            )

    def test_returns_dataset_for_manual_dataset(self):
        """get_dataset returns a dataset for a known manual dataset."""
        ds = get_dataset(
            name="bars_and_stripes",
            split="train",
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
        )
        assert ds is not None
        assert hasattr(ds, '__getitem__')
        assert hasattr(ds, '__len__')


# ---------------------------------------------------------------------------
# QuMETDataModule — initialization
# ---------------------------------------------------------------------------

class TestQuMETDataModuleInit:
    """Tests for QuMETDataModule.__init__()."""

    def test_creates_datamodule_with_required_args(self):
        """QuMETDataModule can be created with required arguments."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        assert dm.name == "mnist"
        assert dm.batch_size == 32
        assert dm.n_samples == 100
        assert dm.n_qubits == 4

    def test_stores_dataset_info_on_init(self):
        """QuMETDataModule stores dataset_info from get_dataset_info on init."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        assert dm.dataset_info is not None
        assert isinstance(dm.dataset_info, QuMETDatasetInfo)
        assert dm.dataset_info.name == "mnist"

    def test_batch_size_set_to_1_for_probs_generation(self):
        """batch_size is forced to 1 when dataset_info.probs_generation is True."""
        dm = QuMETDataModule(
            name="bars_and_stripes",
            batch_size=64,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        # bars_and_stripes has probs_generation=True
        assert dm.batch_size == 1

    def test_initial_datasets_are_none(self):
        """Before prepare_data/setup, all dataset attributes are None."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        assert dm.train_dataset is None
        assert dm.val_dataset is None
        assert dm.test_dataset is None
        assert dm.pred_dataset is None

    def test_accepts_model_name(self):
        """QuMETDataModule accepts optional model_name."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
            model_name="test_model",
        )
        assert dm.model_name == "test_model"


# ---------------------------------------------------------------------------
# QuMETDataModule — dataloader error paths
# ---------------------------------------------------------------------------

class TestQuMETDataModuleDataloaders:
    """Tests for QuMETDataModule dataloader methods (error paths)."""

    def test_train_dataloader_raises_when_dataset_is_none(self):
        """train_dataloader raises RuntimeError when train_dataset is None."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        with pytest.raises(RuntimeError, match="train dataset is not available"):
            dm.train_dataloader()

    def test_val_dataloader_raises_when_dataset_is_none(self):
        """val_dataloader raises RuntimeError when val_dataset is None."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        with pytest.raises(RuntimeError, match="validation dataset is not available"):
            dm.val_dataloader()

    def test_test_dataloader_raises_when_dataset_is_none(self):
        """test_dataloader raises RuntimeError when test_dataset is None."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        with pytest.raises(RuntimeError, match="test dataset is not available"):
            dm.test_dataloader()

    def test_pred_dataloader_raises_when_dataset_is_none(self):
        """pred_dataloader raises RuntimeError when pred_dataset is None."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        with pytest.raises(RuntimeError, match="pred dataset is not available"):
            dm.pred_dataloader()

    def test_train_dataloader_returns_dataloader_when_dataset_set(self):
        """train_dataloader returns a DataLoader when train_dataset is set."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        # Manually set a simple dataset
        dm.train_dataset = TensorDataset(torch.randn(64, 1, 28, 28))
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 32

    def test_val_dataloader_returns_dataloader_when_dataset_set(self):
        """val_dataloader returns a DataLoader when val_dataset is set."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        dm.val_dataset = TensorDataset(torch.randn(32, 1, 28, 28))
        loader = dm.val_dataloader()
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 32
        # Validation loader should not shuffle
        # (shuffle is a property of the DataLoader, but we can't easily check without
        #  inspecting internals — we trust the implementation)

    def test_test_dataloader_returns_dataloader_when_dataset_set(self):
        """test_dataloader returns a DataLoader when test_dataset is set."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        dm.test_dataset = TensorDataset(torch.randn(16, 1, 28, 28))
        loader = dm.test_dataloader()
        assert isinstance(loader, DataLoader)

    def test_pred_dataloader_returns_dataloader_when_dataset_set(self):
        """pred_dataloader returns a DataLoader when pred_dataset is set."""
        dm = QuMETDataModule(
            name="mnist",
            batch_size=32,
            transform=None,
            discretise=False,
            n_samples=100,
            n_qubits=4,
            num_workers=0,
        )
        dm.pred_dataset = TensorDataset(torch.randn(8, 1, 28, 28))
        loader = dm.pred_dataloader()
        assert isinstance(loader, DataLoader)
