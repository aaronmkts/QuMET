"""Tests for qumet.dataset.utils module."""
import pytest
from qumet.dataset.utils import QuMETDatasetInfo


class TestQuMETDatasetInfo:
    """Tests for QuMETDatasetInfo dataclass."""

    def test_creation_with_all_splits(self):
        info = QuMETDatasetInfo(
            name="test_ds",
            dataset_source="manual",
            available_splits=["train", "validation", "test", "pred"],
        )
        assert info.name == "test_ds"
        assert info.train_split_available is True
        assert info.validation_split_available is True
        assert info.test_split_available is True
        assert info.pred_split_available is True

    def test_default_no_splits(self):
        info = QuMETDatasetInfo(name="minimal_ds", dataset_source="manual", available_splits=[])
        assert info.train_split_available is False
        assert info.validation_split_available is False
        assert info.test_split_available is False
        assert info.pred_split_available is False

    def test_train_only(self):
        info = QuMETDatasetInfo(name="train_only", dataset_source="manual", available_splits=["train"])
        assert info.train_split_available is True
        assert info.validation_split_available is False

    def test_getitem_raises_keyerror(self):
        info = QuMETDatasetInfo(name="test_ds", dataset_source="manual", available_splits=["train"])
        with pytest.raises(KeyError, match="nonexistent_key"):
            _ = info["nonexistent_key"]

    def test_getitem_returns_attr(self):
        info = QuMETDatasetInfo(name="test_ds", dataset_source="manual", available_splits=["train"])
        assert info["name"] == "test_ds"
        assert info["num_classes"] is None
