#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script initialise conda for qumet
# --------------------------------------------------------------------
set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# create and activate qumet env
conda env create -f ${DIR}/../main/environment.yml
eval "$(conda shell.bash hook)"
conda activate qumet

# check which python
current_python=$(which python)
if [[ ${current_python} = *"envs/qumet/bin/python" ]]; then
    python -m pip install --user --upgrade pip &&
        python -m pip install -r ${DIR}/../main/requirements.txt

    if [[ $? -eq 0 ]]; then
        echo "✅ Successfully installed all the requirements"
    else
        echo "❌ Failed to install the requirements"
        exit 1
    fi
else
    echo "❌ Failed to find the Python in qumet env. Current Python is at ${current_python}"
    exit 1
fi