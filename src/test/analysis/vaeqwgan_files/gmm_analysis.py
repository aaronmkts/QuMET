import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

# Adjust Python path
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." , "..","src"
    )
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from qumet.dataset import get_dataset, get_dataset_info
from qumet.models import get_model
from qumet.tools.checkpoint_load import load_model


def main():
    # -------------------------------
    # 1. Load Dataset
    # -------------------------------
    mnist = get_dataset('mnist', 'train', 'min-max', False, 5000, 8).data
    dataset_info = get_dataset_info("mnist")
    mnist = mnist.to(torch.float32) / 255.0  
    mnist = mnist.unsqueeze(1)  

    # -------------------------------
    # 2. Load Model
    # -------------------------------
    APQGAN = get_model("apqgan", "image_generation", dataset_info)
    checkpoint_path = "../artifacts/model-ibvzjwpg:v1/model.ckpt"
    model = load_model(checkpoint_path, "pl", APQGAN)
    model.eval()

    # -------------------------------
    # 3. Extract Latent Vectors
    # -------------------------------
    with torch.no_grad():
        mu, log_var, z = model.vae_forward(mnist)

    z_samples = z.cpu().numpy()  

    # -------------------------------
    # 4. Define BIC Scoring Function
    # -------------------------------
    def gmm_bic_score(estimator, X):
        return -estimator.bic(X)  # Negative BIC to make lower values better

    # -------------------------------
    # 5. Grid Search for Best GMM
    # -------------------------------
    param_grid = {
        "n_components": range(1, 8),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(z_samples)

    # -------------------------------
    # 6. Extract Results
    # -------------------------------
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]  # Convert back to positive BIC
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )

    # -------------------------------
    # 7. Find Best Parameters (Lowest BIC)
    # -------------------------------
    best_row = df.loc[df["BIC score"].idxmin()]
    best_n_components = best_row["Number of components"]
    best_covariance = best_row["Type of covariance"]
    best_bic = best_row["BIC score"]

    print(f"Lowest BIC Score: {best_bic:.2f}")
    print(f"Best Number of Components: {best_n_components}")
    print(f"Best Covariance Type: {best_covariance}")
    bic_value = df[
    (df["Number of components"] == 3) & (df["Type of covariance"] == "full")
    ]["BIC score"].values
    print(bic_value)
    # -------------------------------
    # 8. Plot the BIC Scores
    # -------------------------------
    plt.figure(figsize=(10, 6))
    ax_main = sns.barplot(
        data=df,
        x="Number of components",
        y="BIC score",
        hue="Type of covariance"
    )

    # Customizing the font sizes
    ax_main.set_xlabel("Number of Components", fontsize=18)
    ax_main.set_ylabel("BIC Score", fontsize=18)
    ax_main.set_xticks(range(len(df["Number of components"].unique())))
    ax_main.set_xticklabels(df["Number of components"].unique(), fontsize=18)
    ax_main.set_yticklabels(ax_main.get_yticks(), fontsize=18)

        
    # Increase legend size
    plt.legend(title="Type of Covariance", title_fontsize=20, fontsize=18)
    
    ax_main.spines['top'].set_visible(False)   # Hide top spine
    ax_main.spines['right'].set_visible(False) # Hide right spine
    ax_main.spines['bottom'].set_linewidth(1.5) # Make bottom spine thicker
    ax_main.spines['left'].set_linewidth(1.5)   # Make left spine thicker
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
