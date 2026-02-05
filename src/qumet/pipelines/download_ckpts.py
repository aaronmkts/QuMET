import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src")
)

import wandb

from qumet.tools.checkpoint_load import *


def main():

    run = wandb.init()
    artifact = run.use_artifact(
        "qumet/QMI-ModeCollapse/model-j9gn11m7:v1", type="model"
    )
    artifact_dir = artifact.download()
    print("fmnist017")
    # artifacts/model-si56momy:v1/model.ckpt


if __name__ == "__main__":
    main()
