# Getting Started using Conda

## Install Conda (for the first time)

If you don't have `conda` installed yet, fetch the download link from [this page](https://www.anaconda.com/download#downloads), download with wget and execute with all default settings. For example:
```bash
# Update the download link with the appropriate version and platform
wget https://repo.anaconda.com/archive/Anaconda3-<VERSION>-<PLATFORM>.sh
chmod +x Anaconda3-<VERSION>-<PLATFORM>.sh
./Anaconda3-<VERSION>-<PLATFORM>.sh -b
```
## Install environment using Conda

1. Clone the QuMET repository:
```shell
git clone git@github.com/aaronmkts/QuMET.git
```

2. Create your own branch to work on:
```shell
cd qumet
git checkout -b your_branch_name
```

3. Install required dependencies:
```shell
conda env create -f environment.yml
conda activate qumet
pip install -r requirements.txt
```

> **Common error:** when installing the `pip` requirements, make sure the conda environment is activated and the pip command points to your environment version. You can check this by running `which pip` or `which python` and making sure the resulting path includes "anaconda" and "mase".
