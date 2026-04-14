# This file visualizes recording locations across patients in a structural atlas.

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mpl_toolkits.mplot3d import Axes3D

def plot_electrode_locations(nwb_file, atlas='MNI152NLin2009cAsym'):
    # Load NWB data (this is a placeholder, actual loading will depend on the NWB library used)
    # nwb_data = load_nwb_data(nwb_file)
    
    # Placeholder for electrode locations (replace with actual data extraction from NWB)
    electrode_locations = np.random.rand(10, 3)  # Random locations for demonstration
    labels = [f'Electrode {i}' for i in range(len(electrode_locations))]
    
    # Load the atlas
    if atlas == 'MNI152NLin2009cAsym':
        # Load MNI152 template
        template = mne.datasets.fetch_mni152(template='MNI152NLin2009cAsym')
        brain = mne.viz.plot_brain(template)
    
    # Plot electrode locations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(electrode_locations[:, 0], electrode_locations[:, 1], electrode_locations[:, 2], c='r', marker='o')
    
    for i, label in enumerate(labels):
        ax.text(electrode_locations[i, 0], electrode_locations[i, 1], electrode_locations[i, 2], label)
    
    ax.set_title('Electrode Locations in Structural Atlas')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    
    plt.show()

# Example usage
# plot_electrode_locations('path_to_nwb_file.nwb')  # Uncomment and provide actual NWB file path
