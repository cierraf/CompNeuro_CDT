# gen_figure4c_tsnr-fsaverage.py

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

def load_tsnr_data(tsnr_file):
    """Load tSNR data from a file."""
    tsnr_data = np.load(tsnr_file)
    return tsnr_data

def plot_tsnr_on_brain(tsnr_data, title='tSNR on fsaverage'):
    """Plot tSNR data on the fsaverage brain template."""
    # Load fsaverage template
    fsaverage = nib.load(nib.load('path/to/fsaverage/template'))
    
    # Create a brain map
    plotting.plot_stat_map(fsaverage, title=title, display_mode='ortho', 
                            cut_coords=(0, 0, 0), colorbar=True, 
                            threshold=0, vmax=np.max(tsnr_data), 
                            bg_img='path/to/background/image')

def main(tsnr_file):
    """Main function to execute the tSNR visualization."""
    tsnr_data = load_tsnr_data(tsnr_file)
    plot_tsnr_on_brain(tsnr_data)

if __name__ == "__main__":
    tsnr_file = 'path/to/tsnr_data.npy'  # Update with the actual path to the tSNR data file
    main(tsnr_file)