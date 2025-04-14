import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"src"
     )
    )
import seaborn as sns
from qumet.dataset import QuMETDataModule
from qumet.models import get_model, get_model_info
from qumet.dataset import get_dataset, get_dataset_info
import torch.optim as optim
from qumet.tools.checkpoint_load import *
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from collections import defaultdict
import wandb

def main():
   
    run = wandb.init()
    artifact = run.use_artifact('qumet/QMI-ModeCollapse/model-j9gn11m7:v1', type='model')
    artifact_dir = artifact.download()
    print("fmnist017")
    #artifacts/model-si56momy:v1/model.ckpt

if __name__ == "__main__":
    main()
