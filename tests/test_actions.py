"""Tests for qumet.actions module — train() and validate() functions."""
import pytest
from unittest.mock import MagicMock, patch, ANY
import os
import tempfile
from pathlib import Path

import torch
import lightning.pytorch as pl

from qumet.actions import train, validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModel(torch.nn.Module):
    """Minimal model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class FakeDataModule(pl.LightningDataModule):
    """Minimal data module that provides empty dataloaders."""
    def __init__(self, validation_split_available=True):
        super().__init__()
        self.dataset_info = MagicMock()
        self.dataset_info.validation_split_available = validation_split_available
        self.dataset_info.name = "test_ds"
        self.info = self.dataset_info

    def train_dataloader(self):
        return []

    def val_dataloader(self):
        return []


class FakeModelInfo:
    """Minimal model_info object."""
    def __init__(self, model_type_value="qgan", name="patchgan", is_generation_model=True):
        self.model_type = MagicMock()
        self.model_type.value = model_type_value
        self.name = name
        self.is_generation_model = is_generation_model


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

class TestTrain:
    """Tests for train() function."""

    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_calls_trainer_fit(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper
    ):
        """train() creates a Trainer and calls fit()."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        train(
            model=model,
            model_info=model_info,
            data_module=data_module,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
            metrics=None,
            metric_init_args=None,
            add_vae=False,
        )

        mock_trainer.validate.assert_called_once()
        mock_trainer.fit.assert_called_once()

    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_creates_save_path_when_provided(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper
    ):
        """train() creates the save_path directory if it doesn't exist."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "new_subdir", "checkpoints")
            assert not os.path.exists(save_path)

            train(
                model=model,
                model_info=model_info,
                data_module=data_module,
                task="image_generation",
                optimizer="adam",
                learning_rate=0.001,
                weight_decay=0.0,
                plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
                auto_requeue=False,
                save_path=save_path,
                visualizer=None,
                load_name=None,
                load_type=None,
                metrics=None,
                metric_init_args=None,
                add_vae=False,
            )

            assert os.path.isdir(save_path)

    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_passes_callbacks_to_trainer(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper
    ):
        """train() passes callbacks (including checkpoint + lr monitor) to Trainer."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        with tempfile.TemporaryDirectory() as tmp:
            train(
                model=model,
                model_info=model_info,
                data_module=data_module,
                task="image_generation",
                optimizer="adam",
                learning_rate=0.001,
                weight_decay=0.0,
                plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
                auto_requeue=False,
                save_path=tmp,
                visualizer=None,
                load_name=None,
                load_type=None,
                metrics=None,
                metric_init_args=None,
                add_vae=False,
            )

            # Check that callbacks were passed to Trainer
            call_args = mock_trainer_cls.call_args
            assert "callbacks" in call_args[1]
            callbacks = call_args[1]["callbacks"]
            # Should have at least ModelCheckpoint + LearningRateMonitor
            assert len(callbacks) >= 2

    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_without_save_path_skips_callbacks(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper
    ):
        """train() with save_path=None does not add checkpoint/lr callbacks."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        train(
            model=model,
            model_info=model_info,
            data_module=data_module,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
            metrics=None,
            metric_init_args=None,
            add_vae=False,
        )

        # When save_path is None, callbacks should not be in plt_trainer_args
        call_args = mock_trainer_cls.call_args
        # The callbacks key may not be present or may be empty
        cb = call_args[1].get("callbacks", None)
        # Either not present or empty
        assert cb is None or len(cb) == 0

    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_with_auto_requeue_adds_slurm_plugin(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper
    ):
        """train() with auto_requeue=True adds SLURMEnvironment plugin."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        train(
            model=model,
            model_info=model_info,
            data_module=data_module,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=True,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
            metrics=None,
            metric_init_args=None,
            add_vae=False,
        )

        call_args = mock_trainer_cls.call_args
        plugins = call_args[1].get("plugins")
        assert plugins is not None
        assert len(plugins) == 1

    @patch("qumet.actions.train.load_model")
    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_loads_checkpoint_when_load_name_provided(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper, mock_load_model
    ):
        """train() calls load_model when load_name is provided."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []
        mock_load_model.return_value = FakeModel()

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        train(
            model=model,
            model_info=model_info,
            data_module=data_module,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name="some_checkpoint.ckpt",
            load_type="best",
            metrics=None,
            metric_init_args=None,
            add_vae=False,
        )

        mock_load_model.assert_called_once_with(
            "some_checkpoint.ckpt", load_type="best", model=ANY
        )

    @patch("qumet.actions.train.Encoder")
    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_with_add_vae_creates_encoder(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper, mock_encoder_cls
    ):
        """train() with add_vae=True creates an Encoder and passes it to the wrapper."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_instance = MagicMock()
        mock_wrapper_cls.return_value = mock_wrapper_instance
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []
        mock_encoder = MagicMock()
        mock_encoder_cls.return_value = mock_encoder

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        train(
            model=model,
            model_info=model_info,
            data_module=data_module,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
            metrics=None,
            metric_init_args=None,
            add_vae=True,
        )

        mock_encoder_cls.assert_called_once()
        # Check that wrapper was called with encoder
        wrapper_call_args = mock_wrapper_cls.call_args
        assert "encoder" in wrapper_call_args[1]

    @patch("qumet.actions.train.get_model_wrapper")
    @patch("qumet.actions.train.select_callbacks")
    @patch("qumet.actions.train.pl.Trainer")
    def test_train_passes_deterministic_true(
        self, mock_trainer_cls, mock_select_callbacks, mock_get_wrapper
    ):
        """train() creates Trainer with deterministic=True."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_select_callbacks.return_value = []

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule()

        train(
            model=model,
            model_info=model_info,
            data_module=data_module,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
            metrics=None,
            metric_init_args=None,
            add_vae=False,
        )

        call_args = mock_trainer_cls.call_args
        assert call_args[1].get("deterministic") is True
        assert call_args[1].get("num_sanity_val_steps") == 0


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    """Tests for validate() function."""

    @patch("qumet.actions.validate.get_model_wrapper")
    @patch("qumet.actions.validate.pl.Trainer")
    def test_validate_calls_trainer_validate(
        self, mock_trainer_cls, mock_get_wrapper
    ):
        """validate() creates a Trainer and calls validate()."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule(validation_split_available=True)

        validate(
            model=model,
            model_info=model_info,
            data_module=data_module,
            dataset_info={"name": "test"},
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
        )

        mock_trainer.validate.assert_called_once()

    @patch("qumet.actions.validate.get_model_wrapper")
    @patch("qumet.actions.validate.pl.Trainer")
    def test_validate_skips_when_no_validation_split(
        self, mock_trainer_cls, mock_get_wrapper
    ):
        """validate() does not call trainer.validate when validation split unavailable."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule(validation_split_available=False)

        validate(
            model=model,
            model_info=model_info,
            data_module=data_module,
            dataset_info={"name": "test"},
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
        )

        mock_trainer.validate.assert_not_called()

    @patch("qumet.actions.validate.get_model_wrapper")
    @patch("qumet.actions.validate.pl.Trainer")
    def test_validate_creates_save_path_when_provided(
        self, mock_trainer_cls, mock_get_wrapper
    ):
        """validate() creates the save_path directory if it doesn't exist."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule(validation_split_available=True)

        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "validate_output")
            assert not os.path.exists(save_path)

            validate(
                model=model,
                model_info=model_info,
                data_module=data_module,
                dataset_info={"name": "test"},
                task="image_generation",
                optimizer="adam",
                learning_rate=0.001,
                plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
                auto_requeue=False,
                save_path=save_path,
                visualizer=None,
                load_name=None,
                load_type=None,
            )

            assert os.path.isdir(save_path)

    @patch("qumet.actions.validate.load_model")
    @patch("qumet.actions.validate.get_model_wrapper")
    @patch("qumet.actions.validate.pl.Trainer")
    def test_validate_loads_checkpoint_when_load_name_provided(
        self, mock_trainer_cls, mock_get_wrapper, mock_load_model
    ):
        """validate() calls load_model when load_name is provided."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        mock_load_model.return_value = FakeModel()

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule(validation_split_available=True)

        validate(
            model=model,
            model_info=model_info,
            data_module=data_module,
            dataset_info={"name": "test"},
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name="some_checkpoint.ckpt",
            load_type="best",
        )

        mock_load_model.assert_called_once_with(
            "some_checkpoint.ckpt", load_type="best", model=ANY
        )

    @patch("qumet.actions.validate.get_model_wrapper")
    @patch("qumet.actions.validate.pl.Trainer")
    def test_validate_with_auto_requeue_adds_slurm_plugin(
        self, mock_trainer_cls, mock_get_wrapper
    ):
        """validate() with auto_requeue=True adds SLURMEnvironment plugin."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule(validation_split_available=True)

        validate(
            model=model,
            model_info=model_info,
            data_module=data_module,
            dataset_info={"name": "test"},
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=True,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
        )

        call_args = mock_trainer_cls.call_args
        plugins = call_args[1].get("plugins")
        assert plugins is not None
        assert len(plugins) == 1

    @patch("qumet.actions.validate.get_model_wrapper")
    @patch("qumet.actions.validate.pl.Trainer")
    def test_validate_passes_dataset_info_to_wrapper(
        self, mock_trainer_cls, mock_get_wrapper
    ):
        """validate() passes dataset_info to the wrapper constructor."""
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_wrapper_cls = MagicMock()
        mock_wrapper_cls.return_value = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper_cls

        model = FakeModel()
        model_info = FakeModelInfo()
        data_module = FakeDataModule(validation_split_available=True)
        dataset_info = {"name": "test_ds", "num_classes": 10}

        validate(
            model=model,
            model_info=model_info,
            data_module=data_module,
            dataset_info=dataset_info,
            task="image_generation",
            optimizer="adam",
            learning_rate=0.001,
            plt_trainer_args={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
            auto_requeue=False,
            save_path=None,
            visualizer=None,
            load_name=None,
            load_type=None,
        )

        wrapper_call_args = mock_wrapper_cls.call_args
        assert "info" in wrapper_call_args[1]
        assert wrapper_call_args[1]["info"] == dataset_info
