# gen_figure5_isc-fsaverage.py

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import zscore

def load_isc_data(isc_data_file):
    """Load inter-subject correlation data from a file."""
    return np.load(isc_data_file)

def plot_brain_map(data, title, output_file):
    """Plot brain map based on ISC data."""
    # Load the fsaverage brain template
    brain_img = nib.load('path/to/fsaverage/template.nii.gz')
    brain_data = brain_img.get_fdata()

    # Normalize the data for visualization
    norm_data = zscore(data)

    # Create a figure
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Z-score')
    plt.title(title)
    plt.axis('off')

    # Save the figure
    plt.savefig(output_file)
    plt.close()

def main():
    isc_data_file = 'path/to/isc_data.npy'  # Path to the ISC data file
    output_file = 'path/to/output_brain_map.png'  # Output file for the brain map

    # Load ISC data
    isc_data = load_isc_data(isc_data_file)

    # Plot brain map
    plot_brain_map(isc_data, 'Mean Inter-Subject Correlation', output_file)

if __name__ == "__main__":
    main()