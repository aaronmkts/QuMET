"""Tests for qumet.tools.checkpoint_load module."""
import pytest
import torch
import tempfile
import os
from qumet.tools.checkpoint_load import load_model


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_pt_checkpoint(self, tmp_path):
        """Test loading a PyTorch state dict checkpoint."""
        import torch.nn as nn
        model = nn.Linear(10, 2)
        ckpt_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(ckpt_path))

        new_model = nn.Linear(10, 2)
        loaded = load_model(str(ckpt_path), load_type="pt", model=new_model)
        # Weights should match
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(p1, p2)

    def test_load_pl_checkpoint(self, tmp_path):
        """Test loading a PyTorch Lightning checkpoint."""
        import torch.nn as nn
        import lightning.pytorch as pl

        class DummyPLModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

        original = DummyPLModule()
        ckpt_path = tmp_path / "model.ckpt"
        # Save a minimal checkpoint with unwrapped state_dict keys
        raw_state = original.state_dict()
        # Lightning prefixes keys with module name; unwrap them
        unwrapped = {k.replace("linear.", ""): v for k, v in raw_state.items()}
        checkpoint = {
            "state_dict": unwrapped,
            "epoch": 0,
            "global_step": 0,
        }
        torch.save(checkpoint, str(ckpt_path))

        new_model = nn.Linear(10, 2)
        loaded = load_model(str(ckpt_path), load_type="pl", model=new_model)
        for p1, p2 in zip(original.linear.parameters(), loaded.parameters()):
            assert torch.equal(p1, p2)

    def test_raises_on_unsupported_load_type(self, tmp_path):
        import torch.nn as nn
        model = nn.Linear(10, 2)
        ckpt_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(ckpt_path))
        with pytest.raises(ValueError, match="Unknown extension"):
            load_model(str(ckpt_path), load_type="h5", model=model)

    def test_raises_on_missing_file(self):
        import torch.nn as nn
        model = nn.Linear(10, 2)
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/model.pt", load_type="pt", model=model)
