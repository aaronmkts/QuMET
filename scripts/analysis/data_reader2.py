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
    column_jsd = 'val_log/val_jsd_k30'  # Change this variable to plot different metrics
    column_recon = 'encoder/recon_loss'  # Change this variable to plot different metrics
    column_ndb = 'val_log/val_ndb_k30'
    
    column_disp_ndb ='val_log/val_ndb_k30_disp'
    column_disp_jsd ='val_log/val_jsd_k30_disp'

    log_dir_gauss = '../qumet_output/icassp2024/final_results/nwr/gaussian_prior'
    log_dir_qvaegan = '../qumet_output/icassp2024/final_results/nwr/qvaegan'
    log_dir_uniform = '../qumet_output/icassp2024/final_results/nwr/uniform_prior'

    def process_model_data(log_dir, column, dropna=True):
        df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).scalars
        seed19 = df[df['dir_name'] == 'seed_19/software/tensorboard/lightning_logs/version_0']
        seed32 = df[df['dir_name'] == 'seed_32/software/tensorboard/lightning_logs/version_0']
        seed9 = df[df['dir_name'] == 'seed_9/software/tensorboard/lightning_logs/version_0']
        seed23 = df[df['dir_name'] == 'seed_23/software/tensorboard/lightning_logs/version_0']
        seed42 = df[df['dir_name'] == 'software/tensorboard/lightning_logs/version_0']

        # Merging seed 19 and seed 32 data first
        merged_df = pd.merge(seed19[['step', column]], seed32[['step', column]], on='step', suffixes=('_19', '_32'))

        # Merging seed 9, 23, and 42
        merged_df = pd.merge(merged_df, seed9[['step', column]], on='step')
        merged_df = pd.merge(merged_df, seed23[['step', column]], on='step')
        merged_df = pd.merge(merged_df, seed42[['step', column]], on='step')

        # Renaming columns appropriately for each seed
        merged_df.rename(columns={column: f'{column}_42'}, inplace=True)
        merged_df.rename(columns={f'{column}_42': f'{column}_42', column+'_x': f'{column}_9', column+'_y': f'{column}_23'}, inplace=True)

        # Computing the mean and standard deviation across all seeds
        merged_df[f'mean_{column}'] = merged_df[[f'{column}_19', f'{column}_32', f'{column}_9', f'{column}_23', f'{column}_42']].mean(axis=1)
        merged_df[f'std_{column}'] = merged_df[[f'{column}_19', f'{column}_32', f'{column}_9', f'{column}_23', f'{column}_42']].std(axis=1)

        # Optionally drop rows with NaN values
        if dropna:
            merged_df = merged_df.dropna(subset=[f'std_{column}', f'mean_{column}'])
            merged_df['step'] = range(0, len(merged_df))

        return merged_df

    # Process data
    gauss_merged_df_jsd = process_model_data(log_dir_gauss, column_jsd, dropna=True)
    uniform_merged_df_jsd = process_model_data(log_dir_uniform, column_jsd, dropna=True)
    #qvaegan_merged_df_jsd = process_model_data(log_dir_qvaegan, column_jsd, dropna=True)

    gauss_merged_df_ndb = process_model_data(log_dir_gauss, column_ndb, dropna=True)
    uniform_merged_df_ndb = process_model_data(log_dir_uniform, column_ndb, dropna=True)

    qvaegan_merged_df_jsd = process_model_data(log_dir_qvaegan, column_disp_jsd, dropna=True)
    qvaegan_merged_df_ndb = process_model_data(log_dir_qvaegan, column_disp_ndb, dropna=True)


    breakpoint()


        # Define labelss
    main_xlabel = 'Epochs'
    main_ylabel = 'JSD'
    inset_xlabel = 'Epochs'
    inset_ylabel = 'NDB'

    # Create main plot
    fig, ax_main = plt.subplots(figsize=(12, 8))

    gauss_color = '#a1dab4'
    uniform_color = '#41b6c4'
    qvaegan_color = '#225ea8'
    # Plot reconstruction loss
   

    # Plot JSD in the inset
    ax_main.plot(gauss_merged_df_jsd['step'], gauss_merged_df_jsd[f'mean_{column_jsd}'], color=gauss_color, label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$',  linewidth=2)
    ax_main.errorbar(gauss_merged_df_jsd['step'], gauss_merged_df_jsd[f'mean_{column_jsd}'], 
                    yerr=gauss_merged_df_jsd[f'std_{column_jsd}'], fmt='o', color=gauss_color, 
                    label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$', capsize=4, markersize=8)

    ax_main.plot(uniform_merged_df_jsd['step'], uniform_merged_df_jsd[f'mean_{column_jsd}'], color= uniform_color, label=r'Uniform $U_{[0,1)}$',  linewidth=2)
    ax_main.errorbar(uniform_merged_df_jsd['step'], uniform_merged_df_jsd[f'mean_{column_jsd}'], 
                    yerr=uniform_merged_df_jsd[f'std_{column_jsd}'], fmt='o', color= uniform_color, 
                    label=r'Uniform $U_{[0,1)}$', capsize=4, markersize=8)

    ax_main.plot(qvaegan_merged_df_jsd['step'], qvaegan_merged_df_jsd[f'mean_{column_jsd}'], color= qvaegan_color, label=r'VAE-QWGAN $q_{\omega}(z|x)$', linewidth=2)
    ax_main.errorbar(qvaegan_merged_df_jsd['step'], qvaegan_merged_df_jsd[f'mean_{column_jsd}'], 
                    yerr=qvaegan_merged_df_jsd[f'std_{column_jsd}'], fmt='o', color=qvaegan_color, 
                    label=r'VAE-QWGAN $q_{\omega}(z|x)$', capsize=4, markersize=8)


    # Set labels and customize the main plot
    ax_main.set_xlabel(main_xlabel, fontsize=30)
    ax_main.set_ylabel(main_ylabel, fontsize=30)
    ax_main.tick_params(axis='both', which='major', labelsize=30)

    # Remove the top and right borders
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_linewidth(1.5)
    ax_main.spines['left'].set_linewidth(1.5)

    # Create inset for JSD plot
    ax_inset = inset_axes(ax_main, width="40%", height="36%", loc='upper right')

    # Plot JSD in the inset
    ax_inset.plot(gauss_merged_df_ndb['step'], gauss_merged_df_ndb[f'mean_{column_ndb}'], color=gauss_color, label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$')
    ax_inset.errorbar(gauss_merged_df_ndb['step'], gauss_merged_df_ndb[f'mean_{column_ndb}'], 
                    yerr=gauss_merged_df_ndb[f'std_{column_ndb}'], fmt='o', color=gauss_color, 
                    label=r'Gaussian $\mathcal{N}(0, \mathbb{I})$', capsize=4, markersize=5)
    ax_inset.plot(uniform_merged_df_ndb['step'], uniform_merged_df_ndb[f'mean_{column_ndb}'], color= uniform_color, label=r'Uniform $U_{[0,1)}$')
    ax_inset.errorbar(uniform_merged_df_ndb['step'], uniform_merged_df_ndb[f'mean_{column_ndb}'], 
                    yerr=uniform_merged_df_ndb[f'std_{column_ndb}'], fmt='o', color= uniform_color, 
                    label=r'Uniform $U_{[0,1)}$', capsize=4, markersize=5)
    
    ax_inset.plot(qvaegan_merged_df_ndb['step'], qvaegan_merged_df_ndb[f'mean_{column_ndb}'], color= qvaegan_color, label=r'VAE-QWGAN $q_{\omega}(z|x)$')
    ax_inset.errorbar(qvaegan_merged_df_ndb['step'], qvaegan_merged_df_ndb[f'mean_{column_ndb}'],
                    yerr=qvaegan_merged_df_ndb[f'std_{column_ndb}'], fmt='o', color= qvaegan_color,
                    label=r'VAE-QWGAN $q_{\omega}(z|x)$', capsize=4, markersize=5) 
    # Set labels and customize the inset plot
   # ax_inset.set_xlabel(inset_xlabel, fontsize=25)
    ax_inset.set_ylabel(inset_ylabel, fontsize=25)
    ax_inset.tick_params(axis='both', which='major', labelsize=25)

    # Remove the top and right borders of the inset plot
    ax_inset.spines['top'].set_visible(False)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['bottom'].set_linewidth(1.5)
    ax_inset.spines['left'].set_linewidth(1.5)


    # Show the combined plot
    plt.show()



    
if __name__ == "__main__":
    main()

