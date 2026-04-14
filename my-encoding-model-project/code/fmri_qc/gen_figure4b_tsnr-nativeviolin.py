import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore

def load_tsnr_data(tsnr_file):
    """Load tSNR data from a CSV file."""
    return pd.read_csv(tsnr_file)

def plot_violin(tsnr_data, output_file):
    """Generate a violin plot for tSNR values."""
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='participant', y='tsnr', data=tsnr_data)
    plt.title('Temporal Signal-to-Noise Ratio (tSNR) by Participant')
    plt.xlabel('Participant')
    plt.ylabel('tSNR')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_brain_map(correlation_data, brain_template, output_file):
    """Plot brain map colored by correlation values."""
    brain_img = nib.load(brain_template)
    brain_data = brain_img.get_fdata()

    # Normalize correlation data
    norm_corr = zscore(correlation_data)

    # Create a new NIfTI image for the brain map
    brain_map = np.zeros_like(brain_data)
    brain_map[brain_data > 0] = norm_corr  # Assuming brain_data is a mask

    new_img = nib.Nifti1Image(brain_map, affine=brain_img.affine)
    nib.save(new_img, output_file)

def main(tsnr_file, brain_template, output_violin, output_brain_map):
    """Main function to execute the analysis."""
    tsnr_data = load_tsnr_data(tsnr_file)
    plot_violin(tsnr_data, output_violin)

    # Example correlation data for brain mapping
    correlation_data = np.random.rand(tsnr_data.shape[0])  # Replace with actual correlation data
    plot_brain_map(correlation_data, brain_template, output_brain_map)

if __name__ == "__main__":
    tsnr_file = 'path/to/tsnr_data.csv'  # Update with actual path
    brain_template = 'path/to/brain_template.nii'  # Update with actual path
    output_violin = 'output/violin_plot.png'
    output_brain_map = 'output/brain_map.nii'
    
    main(tsnr_file, brain_template, output_violin, output_brain_map)