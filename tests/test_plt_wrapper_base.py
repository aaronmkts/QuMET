"""Tests for qumet.plt_wrapper.base module."""
import pytest
import torch
import lightning.pytorch as pl
from qumet.plt_wrapper.base import WrapperBase, ValidationResult


class TestWrapperBase:
    """Tests for WrapperBase LightningModule."""

    def test_initialization_stores_attributes(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-5,
            epochs=10,
            optimizer="adam",
        )
        assert wrapper.model is model
        assert wrapper.learning_rate == 1e-3
        assert wrapper.weight_decay == 1e-5
        assert wrapper.epochs == 10
        assert wrapper.optimizer == "adam"

    def test_default_values(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model)
        assert wrapper.learning_rate == 5e-4
        assert wrapper.weight_decay == 0.0
        assert wrapper.epochs == 1
        assert wrapper.optimizer is None

    def test_loss_fn_is_cross_entropy(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model)
        assert isinstance(wrapper.loss_fn, torch.nn.CrossEntropyLoss)

    def test_forward_raises_not_implemented(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model)
        with pytest.raises(NotImplementedError):
            wrapper.forward(torch.randn(4, 10))

    def test_training_step_raises_not_implemented(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model)
        with pytest.raises(NotImplementedError):
            wrapper.training_step((torch.randn(4, 10), torch.randint(0, 2, (4,))))

    def test_configure_optimizers_raises_not_implemented(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model)
        with pytest.raises(NotImplementedError):
            wrapper.configure_optimizers()

    def test_is_lightning_module(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model)
        assert isinstance(wrapper, pl.LightningModule)

    def test_freeze_modules_stored(self):
        model = torch.nn.Linear(10, 2)
        wrapper = WrapperBase(model=model, freeze_modules=["encoder"])
        assert wrapper.freeze_modules == ["encoder"]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_creation(self):
        result = ValidationResult()
        assert result.others == {}
        assert result.real_image is None
        assert result.fake_image is None
        assert result.recon_image is None
        assert result.label is None
        assert result.encode_latent is None
        assert result.real_distribution is None
        assert result.fake_distribution is None

    def test_can_set_fields(self):
        result = ValidationResult(
            real_image=torch.randn(3, 32, 32),
            fake_image=torch.randn(3, 32, 32),
            label=torch.tensor([1, 2, 3]),
        )
        assert result.real_image is not None
        assert result.fake_image is not None
        assert result.label is not None

    def test_others_dict_mutable(self):
        result = ValidationResult()
        result.others["key"] = "value"
        assert result.others["key"] == "value"
