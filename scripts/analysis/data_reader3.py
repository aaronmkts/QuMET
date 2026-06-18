import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,".."
     )
    )
import torch 
import torch.nn as nn
import tempfile
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import torchvision
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def main():
    
        # Define the metric to plot by assigning the column name
    column_jsd = 'val_log/val_jsd_k50'  # Change this variable to plot different metrics
    column_recon = 'encoder/recon_loss'  # Change this variable to plot different metrics
    column_ndb = 'val_log/val_ndb_k50'  # Change this variable to plot different metrics

    log_dir_gauss = '../qumet_output/icassp2024/final_results/fashionmnist/gaussian_prior'
    log_dir_qvaegan = '../qumet_output/icassp2024/final_results/fashionmnist/qvaegan'
    log_dir_uniform = '../qumet_output/icassp2024/final_results/fashionmnist/uniform_prior'

    def process_model_data(log_dir, column, dropna=True):
        df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).scalars
        seed19 = df[df['dir_name'] == 'seed_19/software/tensorboard/lightning_logs/version_0']
        seed32 = df[df['dir_name'] == 'seed_32/software/tensorboard/lightning_logs/version_0']
        seed42 = df[df['dir_name'] == 'seed_42/software/tensorboard/lightning_logs/version_0']

        merged_df = pd.merge(seed19[['step', column]], seed32[['step', column]], on='step', suffixes=('_19', '_32'))
        merged_df = pd.merge(merged_df, seed42[['step', column]], on='step')
        merged_df.rename(columns={column: f'{column}_42'}, inplace=True)

        merged_df[f'mean_{column}'] = merged_df[[f'{column}_19', f'{column}_32', f'{column}_42']].mean(axis=1)
        merged_df[f'std_{column}'] = merged_df[[f'{column}_19', f'{column}_32', f'{column}_42']].std(axis=1)
        if dropna:
            merged_df = merged_df.dropna(subset=['std_'+f'{column}'])
            merged_df = merged_df.dropna(subset=['mean_'+f'{column}'])
            merged_df['step'] = range(0, len(merged_df))
        return merged_df

    # Process data
    gauss_merged_df_jsd = process_model_data(log_dir_gauss, column_jsd, dropna=True)
    uniform_merged_df_jsd = process_model_data(log_dir_uniform, column_jsd, dropna=True)
    qvaegan_merged_df_jsd = process_model_data(log_dir_qvaegan, column_jsd, dropna=True)

    gauss_merged_df_ndb = process_model_data(log_dir_gauss, column_ndb, dropna=True)
    uniform_merged_df_ndb = process_model_data(log_dir_uniform, column_ndb, dropna=True)
    qvaegan_merged_df_ndb = process_model_data(log_dir_qvaegan, column_ndb, dropna=True)

    qvaegan_merged_df_recon = process_model_data(log_dir_qvaegan, column_recon, dropna=False)

    main_xlabel = 'Iterations'
    main_ylabel = 'Reconstruction loss'
    inset_xlabel = 'Epochs'
    inset_ylabel = 'JSD'
    new_inset_xlabel = 'Epochs'
    new_inset_ylabel = 'NDB/K'

    # Create main plot
    fig, ax_main = plt.subplots(figsize=(12, 8))

    # Plot reconstruction loss
    ax_main.plot(qvaegan_merged_df_recon['step'], qvaegan_merged_df_recon[f'mean_{column_recon}'], color='tomato', alpha=0.8, label=r'VAE-QWGAN $q_{\omega}(z|x)$')
    ax_main.fill_between(qvaegan_merged_df_recon['step'], 
                        qvaegan_merged_df_recon[f'mean_{column_recon}'] - qvaegan_merged_df_recon[f'std_{column_recon}'], 
                        qvaegan_merged_df_recon[f'mean_{column_recon}'] + qvaegan_merged_df_recon[f'std_{column_recon}'], 
                        color='tomato', alpha=0.2)

    # Set labels and customize the main plot
    ax_main.set_xlabel(main_xlabel, fontsize=30)
    ax_main.set_ylabel(main_ylabel, fontsize=30)
    ax_main.tick_params(axis='both', which='major', labelsize=30)

    # Remove the top and right borders
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_linewidth(1.5)
    ax_main.spines['left'].set_linewidth(1.5)

    # Create the first inset for JSD plot
    ax_inset1 = inset_axes(ax_main, width="35%", height="35%", loc='upper right')

    # Plot JSD in the first inset
    ax_inset1.plot(gauss_merged_df_jsd['step'], gauss_merged_df_jsd[f'mean_{column_jsd}'], color='orange', alpha=0.8, label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$')
    ax_inset1.errorbar(gauss_merged_df_jsd['step'], gauss_merged_df_jsd[f'mean_{column_jsd}'], 
                    yerr=gauss_merged_df_jsd[f'std_{column_jsd}'], fmt='o', color='orange', 
                    label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$', capsize=4, markersize=5)

    ax_inset1.plot(uniform_merged_df_jsd['step'], uniform_merged_df_jsd[f'mean_{column_jsd}'], color='blueviolet', alpha=0.8, label=r'Uniform $U_{[0,1)}$')
    ax_inset1.errorbar(uniform_merged_df_jsd['step'], uniform_merged_df_jsd[f'mean_{column_jsd}'], 
                    yerr=uniform_merged_df_jsd[f'std_{column_jsd}'], fmt='o', color='blueviolet', 
                    label=r'Uniform $U_{[0,1)}$', capsize=4, markersize=5)

    ax_inset1.plot(qvaegan_merged_df_jsd['step'], qvaegan_merged_df_jsd[f'mean_{column_jsd}'], color='tomato', alpha=0.8, label=r'VAE-QWGAN $q_{\omega}(z|x)$')
    ax_inset1.errorbar(qvaegan_merged_df_jsd['step'], qvaegan_merged_df_jsd[f'mean_{column_jsd}'], 
                    yerr=qvaegan_merged_df_jsd[f'std_{column_jsd}'], fmt='o', color='tomato', 
                    label=r'VAE-QWGAN $q_{\omega}(z|x)$', capsize=4, markersize=5)

    # Set labels and customize the first inset plot
    ax_inset1.set_xlabel(inset_xlabel, fontsize=23)
    ax_inset1.set_ylabel(inset_ylabel, fontsize=23)
    ax_inset1.tick_params(axis='both', which='major', labelsize=23)

    # Remove the top and right borders of the first inset plot
    ax_inset1.spines['top'].set_visible(False)
    ax_inset1.spines['right'].set_visible(False)
    ax_inset1.spines['bottom'].set_linewidth(1.5)
    ax_inset1.spines['left'].set_linewidth(1.5)

    # Create the second inset for additional JSD plot
    ax_inset2 = inset_axes(ax_main, width="35%", height="35%", loc='upper left',
                       bbox_to_anchor=(0.17, -0.0,1,1),  # Adjust these values to shift the inset
                       bbox_transform=ax_main.transAxes)
  

    # Define additional JSD data for the second inset (example: `gauss_merged_df_jsd` as an example)
    # You can replace this with the actual data you want to plot in the second inset
    ax_inset2.plot(gauss_merged_df_ndb['step'], gauss_merged_df_ndb[f'mean_{column_ndb}'], color='orange', alpha=0.8, label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$')
    ax_inset2.errorbar(gauss_merged_df_ndb['step'], gauss_merged_df_ndb[f'mean_{column_ndb}'], 
                    yerr=gauss_merged_df_ndb[f'std_{column_ndb}'], fmt='o', color='orange', 
                    label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$', capsize=4, markersize=5)
    
    ax_inset2.plot(uniform_merged_df_ndb['step'], uniform_merged_df_ndb[f'mean_{column_ndb}'], color='blueviolet', alpha=0.8, label=r'Uniform $U_{[0,1)}$')
    ax_inset2.errorbar(uniform_merged_df_ndb['step'], uniform_merged_df_ndb[f'mean_{column_ndb}'], 
                    yerr=uniform_merged_df_ndb[f'std_{column_ndb}'], fmt='o', color='blueviolet', 
                    label=r'Uniform $U_{[0,1)}$', capsize=4, markersize=5)

    ax_inset2.plot(qvaegan_merged_df_ndb['step'], qvaegan_merged_df_ndb[f'mean_{column_ndb}'], color='tomato', alpha=0.8, label=r'VAE-QWGAN $q_{\omega}(z|x)$')
    ax_inset2.errorbar(qvaegan_merged_df_ndb['step'], qvaegan_merged_df_ndb[f'mean_{column_ndb}'], 
                    yerr=qvaegan_merged_df_ndb[f'std_{column_ndb}'], fmt='o', color='tomato', 
                    label=r'VAE-QWGAN $q_{\omega}(z|x)$', capsize=4, markersize=5)
    

    # Set labels and customize the second inset plot
    ax_inset2.set_xlabel(new_inset_xlabel, fontsize=23)
    ax_inset2.set_ylabel(new_inset_ylabel, fontsize=23)
    ax_inset2.tick_params(axis='both', which='major', labelsize=23)

    # Remove the top and right borders of the second inset plot
    ax_inset2.spines['top'].set_visible(False)
    ax_inset2.spines['right'].set_visible(False)
    ax_inset2.spines['bottom'].set_linewidth(1.5)
    ax_inset2.spines['left'].set_linewidth(1.5)

    # Show the combined plot
    plt.show()

    
if __name__ == "__main__":
    main()

