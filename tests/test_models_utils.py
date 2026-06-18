"""Tests for qumet.models.utils module."""
import pytest
from qumet.models.utils import QumetModelInfo, ModelType, ModelTaskType


class TestModelType:
    """Tests for ModelType enum."""

    def test_has_expected_members(self):
        assert ModelType.QGAN.value == "qgan"
        assert ModelType.GAN.value == "gan"
        assert ModelType.QCBM.value == "qcbm"
        assert ModelType.VAE.value == "vae"

    def test_from_string(self):
        assert ModelType("qgan") == ModelType.QGAN
        assert ModelType("gan") == ModelType.GAN
        assert ModelType("qcbm") == ModelType.QCBM
        assert ModelType("vae") == ModelType.VAE

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            ModelType("invalid")


class TestModelTaskType:
    """Tests for ModelTaskType enum."""

    def test_generation_value(self):
        assert ModelTaskType.GENERATION.value == "generation"


class TestQumetModelInfo:
    """Tests for QumetModelInfo dataclass."""

    def test_creation_with_enum_types(self):
        info = QumetModelInfo(
            name="test_model",
            model_type=ModelType.QGAN,
            task_type=ModelTaskType.GENERATION,
            bitstring_sampling=True,
        )
        assert info.name == "test_model"
        assert info.model_type == ModelType.QGAN
        assert info.task_type == ModelTaskType.GENERATION

    def test_creation_with_string_types(self):
        info = QumetModelInfo(
            name="test_model",
            model_type="qgan",
            task_type="generation",
            bitstring_sampling=True,
        )
        assert info.model_type == ModelType.QGAN
        assert info.task_type == ModelTaskType.GENERATION

    def test_is_generation_model(self):
        info = QumetModelInfo(
            name="gen_model",
            model_type=ModelType.QGAN,
            task_type=ModelTaskType.GENERATION,
            bitstring_sampling=True,
        )
        assert info.is_generation_model is True

    def test_post_init_asserts_generation_model(self):
        """A generation model must have at least one sampling flag."""
        with pytest.raises(AssertionError, match="Must be a generative model"):
            QumetModelInfo(
                name="bad_model",
                model_type=ModelType.QGAN,
                task_type=ModelTaskType.GENERATION,
                bitstring_sampling=False,
                observable_sampling=False,
            )

    def test_observable_sampling_sufficient(self):
        info = QumetModelInfo(
            name="obs_model",
            model_type=ModelType.QGAN,
            task_type=ModelTaskType.GENERATION,
            bitstring_sampling=False,
            observable_sampling=True,
        )
        assert info.is_generation_model is True

    def test_both_sampling_flags(self):
        info = QumetModelInfo(
            name="full_model",
            model_type=ModelType.QCBM,
            task_type=ModelTaskType.GENERATION,
            bitstring_sampling=True,
            observable_sampling=True,
        )
        assert info.is_generation_model is True
