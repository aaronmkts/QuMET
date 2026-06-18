"""Tests for qumet.tools.utils module."""
import pytest
import numpy as np
import torch
import tempfile
import os
from qumet.tools.utils import (
    is_tensor,
    to_numpy,
    to_numpy_if_tensor,
    to_tensor,
    to_tensor_if_numpy,
    copy_weights,
    get_checkpoint_file,
    parse_accelerator,
)


class TestTensorUtils:
    """Tests for tensor/numpy conversion utilities."""

    def test_is_tensor_returns_true_for_tensor(self):
        t = torch.tensor([1.0, 2.0])
        assert is_tensor(t) is True

    def test_is_tensor_returns_false_for_numpy(self):
        a = np.array([1.0, 2.0])
        assert is_tensor(a) is False

    def test_is_tensor_returns_false_for_list(self):
        assert is_tensor([1, 2, 3]) is False

    def test_to_numpy_converts_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_if_tensor_converts_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = to_numpy_if_tensor(t)
        assert isinstance(result, np.ndarray)

    def test_to_numpy_if_tensor_passes_through_numpy(self):
        a = np.array([1.0, 2.0])
        result = to_numpy_if_tensor(a)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, a)

    def test_to_tensor_converts_numpy(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_tensor(a)
        assert torch.is_tensor(result)
        assert result.dtype == torch.float32

    def test_to_tensor_if_numpy_converts_numpy(self):
        a = np.array([1.0, 2.0])
        result = to_tensor_if_numpy(a)
        assert torch.is_tensor(result)

    def test_to_tensor_if_numpy_passes_through_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = to_tensor_if_numpy(t)
        assert result is t  # same object

    def test_copy_weights_copies_values(self):
        src = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        tgt = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        copy_weights(src, tgt)
        assert torch.equal(tgt, torch.tensor([1.0, 2.0, 3.0]))


class TestCheckpointFile:
    """Tests for get_checkpoint_file."""

    def test_finds_ckpt_file(self, tmp_path):
        (tmp_path / "model.ckpt").touch()
        (tmp_path / "other.txt").touch()
        result = get_checkpoint_file(str(tmp_path))
        assert result == "model.ckpt"

    def test_returns_none_when_no_ckpt(self, tmp_path):
        (tmp_path / "other.txt").touch()
        result = get_checkpoint_file(str(tmp_path))
        assert result is None


class TestParseAccelerator:
    """Tests for parse_accelerator."""

    def test_auto_returns_device(self):
        device = parse_accelerator("auto")
        assert isinstance(device, torch.device)

    def test_gpu_returns_cuda_device(self):
        device = parse_accelerator("gpu")
        assert device.type == "cuda"

    def test_cpu_returns_cpu_device(self):
        device = parse_accelerator("cpu")
        assert device.type == "cpu"

    def test_unsupported_raises(self):
        with pytest.raises(RuntimeError, match="Unsupported accelerator"):
            parse_accelerator("tpu")
