import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src")
)
import itertools
from itertools import product

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qumet.dataset import QuMETDataModule, get_dataset, get_dataset_info
from qumet.models import get_model, get_model_info
from qumet.tools.checkpoint_load import *


def main():
    # print(os.listdir())

    dataset_info = get_dataset_info("mnist")
    model = get_model("apqgan", "image_generation", dataset_info)
    ckpt = torch.load("best.ckpt", weights_only=True)

    print(model.modules)

    ckpt_updated = {
        key.replace("model.", "", 1): value for key, value in ckpt["state_dict"].items()
    }

    assert ckpt_updated.keys() == model.state_dict().keys()

    module_list = model.state_dict().keys()

    print(module_list)

    # target_modules = ['encoder.encoder.0.weight', 'encoder.encoder.0.bias']

    # for name, param in model.named_parameters():
    #     print(name, "->", param.requires_grad)

    # for module_name, param in model.named_parameters():
    #     if module_name in target_modules:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    # print()

    # for name, param in model.named_parameters():
    #     print(name, "->", param.requires_grad)

    # print(model.state_dict()['encoder.encoder.0.bias'])
    # print(ckpt_updated['encoder.encoder.0.bias'])

    # model.load_state_dict(ckpt_updated)

    # print(model.state_dict()['encoder.encoder.0.bias'])

    # print(model.state_dict()['encoder.encoder.0.bias'])
    # print(ckpt["state_dict"]['model.encoder.encoder.0.bias'])

    # model.load_state_dict(ckpt["state_dict"],strict=False)

    # for name in ckpt["state_dict"].keys():
    # ckpt["state_dict"][name] = name.replace("model.","")
    # print(name)
    # print(name.replace("model.",""))
    # ckpt["state_dict"][name]
    # print(ckpt["state_dict"][name])
    # ckpt["state_dict"][name] = ckpt["state_dict"][name].lstrip("model.") # remove "model." from keys
    # print(name)
    # print(f"Layer: {name} | Shape: {param.shape}")# | Values: {param}")
    # print(ckpt["state_dict"].keys())

    # print(model.state_dict()['encoder.encoder.0.bias'])
    # print(ckpt["state_dict"]['model.encoder.encoder.0.bias'])

    # model.load_state_dict(ckpt["state_dict"],strict=False)

    # print(model.state_dict()['encoder.encoder.0.bias'])

    # for name, param in model.state_dict().items():
    #    print(f"Layer: {name} | Shape: {param.shape} | Values: {param}")

    # print(ckpt.keys())

    # print(ckpt["state_dict"].keys())
    # print(model.named_modules)

    # load_lightning_ckpt_to_unwrapped_model(ckpt["state_dict"],model)
    # load_unwrapped_ckpt(ckpt["state_dict"],model)

    # model.load_state_dict(ckpt["state_dict"])

    # for name, module in model.named_modules():
    #    print(module)

    # print(model.modules)
    # print(help(model))


if __name__ == "__main__":
    main()
