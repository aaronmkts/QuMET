import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main"
     )
    )

import torch.nn as nn
from codebase.dataset.manual import TwoDGaussianDataset
from codebase.dataset.manual import GaussianDataset
def main():

    visuals = TwoDGaussianDataset().visualise()

if __name__ == "__main__":
    main()