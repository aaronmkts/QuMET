"""Regression tests for review priority bugs and high-risk public surfaces."""
from types import SimpleNamespace

import pytest
import torch

from qumet import models
from qumet.models.qgan import QGAN_MODELS, get_qgan_model_info, is_qgan_model
from qumet.plt_wrapper import get_model_wrapper
from qumet.plt_wrapper.qgan import (
    GANWrapper,
    PatchGANWrapper,
    QGANDiscreteGenModelWrapper,
    QGANProbsGenModelWrapper,
)
from qumet.plt_wrapper.vaeqgan_wrapper import VAEGANWrapper
from qumet.tools.callbacks import evaluation
from qumet.tools.callbacks.evaluation import GMMEvaluationCallback, PSNRCallback
from qumet.tools.callbacks.visualisation import GANImagesCallback, TSNEPlotCallback


class DummyLogger:
    def __init__(self):
        self.images = []

    def log_image(self, **kwargs):
        self.images.append(kwargs)


class DummyTrainer:
    def __init__(self, current_epoch=0, max_epochs=1):
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.logger = DummyLogger()


class DummyLightningModule:
    def __init__(self):
        self.logged = []

    def log(self, key, value, **kwargs):
        self.logged.append((key, value, kwargs))


class DummyGenerator(torch.nn.Module):
    n_qubits = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, *args, **kwargs):
        return torch.ones(2)


class DummyDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.ones(x.shape[0], dtype=x.dtype, device=x.device)


class DummyGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = DummyGenerator()
        self.discriminator = DummyDiscriminator()

    def forward(self, noise):
        return torch.zeros(noise.shape[0], 1, 28, 28)


def test_get_model_wrapper_add_vae_rejects_non_gan_model_type():
    """add_vae=True must not route non-QGAN/GAN models to VAEGANWrapper."""
    qcbm_info = models.get_model_info("qcbm")

    with pytest.raises(ValueError, match="model type qcbm"):
        get_model_wrapper(qcbm_info, "image_generation", add_vae=True)


def test_get_model_wrapper_add_vae_accepts_image_qgan_model_type():
    patchgan_info = models.get_model_info("patchgan")
    assert get_model_wrapper(patchgan_info, "image_generation", add_vae=True) is VAEGANWrapper


def test_cli_discretise_is_false_for_image_generation(monkeypatch):
    """Image tasks should not be discretised just because a non-empty literal exists."""
    from qumet import cli as cli_module

    captured = {}

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "QuMETDataModule", DummyDataModule)
    monkeypatch.setattr(cli_module.models, "get_model_info", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(cli_module.models, "get_model", lambda **kwargs: object())

    cli = object.__new__(cli_module.QuMETCLI)
    cli.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    cli.args = SimpleNamespace(
        model="gan",
        dataset="mnist",
        batch_size=4,
        transform=None,
        n_samples=8,
        n_qubits=3,
        num_workers=0,
        task="image_generation",
    )

    cli._setup_model_and_dataset()

    assert captured["discretise"] is False


def test_cli_discretise_is_true_for_discrete_and_probs_tasks(monkeypatch):
    from qumet import cli as cli_module

    captured = []

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.append(kwargs["discretise"])

    monkeypatch.setattr(cli_module, "QuMETDataModule", DummyDataModule)
    monkeypatch.setattr(cli_module.models, "get_model_info", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(cli_module.models, "get_model", lambda **kwargs: object())

    cli = object.__new__(cli_module.QuMETCLI)
    cli.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    for task in ("discrete_generation", "probs_generation"):
        cli.args = SimpleNamespace(
            model="efficientsu2",
            dataset="gaussian",
            batch_size=4,
            transform=None,
            n_samples=8,
            n_qubits=3,
            num_workers=0,
            task=task,
        )
        cli._setup_model_and_dataset()

    assert captured == [True, True]


def test_psnr_callback_passes_real_images_as_reference(monkeypatch):
    calls = []

    def fake_psnr(real, fake):
        calls.append((real.copy(), fake.copy()))
        return 7.0

    monkeypatch.setattr(evaluation, "peak_signal_noise_ratio", fake_psnr)
    callback = PSNRCallback()
    trainer = DummyTrainer(current_epoch=0)
    module = DummyLightningModule()
    real = torch.zeros(1, 1, 28, 28)
    fake = torch.ones(1, 1, 28, 28)
    callback.on_validation_batch_end(trainer, module, SimpleNamespace(real_image=real, fake_image=fake), None, 0)

    callback.on_validation_epoch_end(trainer, module)

    assert torch.as_tensor(calls[0][0]).equal(real.reshape(28, 28))
    assert torch.as_tensor(calls[0][1]).equal(fake.reshape(28, 28))
    assert module.logged[0][0] == "metrics/psnr"
    assert module.logged[0][1] == 7.0


def test_gmm_psnr_passes_real_images_as_reference(monkeypatch):
    calls = []

    def fake_psnr(real, fake):
        calls.append((real.copy(), fake.copy()))
        return 11.0

    monkeypatch.setattr(evaluation, "peak_signal_noise_ratio", fake_psnr)
    callback = GMMEvaluationCallback()
    real = torch.zeros(1, 1, 28, 28)
    fake = torch.ones(1, 1, 28, 28)

    assert callback.calculate_psnr(real, fake) == 11.0
    assert torch.as_tensor(calls[0][0]).equal(real.reshape(28, 28))
    assert torch.as_tensor(calls[0][1]).equal(fake.reshape(28, 28))


@pytest.mark.parametrize(
    ("name", "expected_type", "sampling_attr"),
    [
        ("efficientsu2", "qgan", "bitstring_sampling"),
        ("patchgan", "qgan", "observable_sampling"),
        ("qinr", "qgan", "observable_sampling"),
    ],
)
def test_qgan_model_registry_exposes_expected_generation_variants(name, expected_type, sampling_attr):
    assert is_qgan_model(name)
    info = get_qgan_model_info(name)
    assert info.name == name
    assert info.model_type.value == expected_type
    assert getattr(info, sampling_attr) is True
    assert callable(QGAN_MODELS[name]["get_model_fn_generation"])


def test_get_model_wrapper_maps_qgan_discrete_and_probs_wrappers():
    discrete_info = models.get_model_info("efficientsu2")
    probs_info = models.get_model_info("su2")

    assert get_model_wrapper(discrete_info, "discrete_generation") is QGANDiscreteGenModelWrapper
    assert get_model_wrapper(probs_info, "probs_generation") is QGANProbsGenModelWrapper


def test_get_model_wrapper_maps_image_gan_wrappers():
    assert get_model_wrapper(models.get_model_info("patchgan"), "image_generation") is PatchGANWrapper
    assert get_model_wrapper(models.get_model_info("gan"), "image_generation") is GANWrapper


def test_gan_images_callback_logs_real_fake_recon_and_other_images():
    callback = GANImagesCallback(batch_size=2, every_n_epochs=1, nrow=1)
    trainer = DummyTrainer(current_epoch=0)
    outputs = SimpleNamespace(
        real_image=torch.zeros(2, 1, 4, 4),
        fake_image=torch.ones(2, 1, 4, 4),
        recon_image=torch.full((2, 1, 4, 4), 0.5),
        others={"custom": torch.full((2, 1, 4, 4), 0.25)},
    )

    callback.on_validation_batch_end(trainer, DummyLightningModule(), outputs, None, 0)

    assert [entry["key"] for entry in trainer.logger.images] == [
        "images/real",
        "images/recon",
        "images/sample",
        "images/custom",
    ]


def test_tsne_callback_collect_samples_detaches_tensors():
    callback = TSNEPlotCallback()
    tensor = torch.ones(2, 3, requires_grad=True)
    storage = []

    callback._collect_samples(storage, tensor)

    assert len(storage) == 1
    assert storage[0].device.type == "cpu"
    assert storage[0].requires_grad is False
