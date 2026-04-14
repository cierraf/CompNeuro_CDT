# This script prepares data for inter-subject correlation (ISC) analysis on fsaverage.

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import zscore

def load_fmri_data(fmriprep_dir):
    # Load fMRI data from the specified directory
    # This function assumes that the data is in NIfTI format
    fmri_files = glob.glob(fmriprep_dir + '/*/*.nii.gz')
    fmri_data = [nib.load(f).get_fdata() for f in fmri_files]
    return fmri_data

def compute_isc(fmri_data):
    # Compute inter-subject correlation (ISC)
    isc_matrix = np.corrcoef(fmri_data)
    return isc_matrix

def zscore_isc(isc_matrix):
    # Z-score the ISC matrix
    return zscore(isc_matrix)

def plot_brain_map(isc_z, title='ISC Brain Map'):
    # Plot the ISC brain map
    plt.figure(figsize=(10, 8))
    plt.imshow(isc_z, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Z-score ISC')
    plt.title(title)
    plt.xlabel('Brain Regions')
    plt.ylabel('Brain Regions')
    plt.show()

def main(fmriprep_dir):
    # Main function to execute the ISC preparation
    fmri_data = load_fmri_data(fmriprep_dir)
    isc_matrix = compute_isc(fmri_data)
    isc_z = zscore_isc(isc_matrix)
    plot_brain_map(isc_z)

if __name__ == "__main__":
    fmriprep_dir = '/path/to/fmriprep_directory/'  # Update this path
    main(fmriprep_dir)