import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import pearsonr

def load_eye_tracking_data(file_path):
    """Load eye tracking data from a CSV file."""
    return pd.read_csv(file_path)

def load_brain_response_data(file_path):
    """Load brain response data from a CSV file."""
    return pd.read_csv(file_path)

def compute_correlations(eye_tracking_data, brain_response_data):
    """Compute correlations between eye tracking metrics and brain responses."""
    correlations = {}
    for column in eye_tracking_data.columns:
        if column != 'participant_id':
            corr, _ = pearsonr(eye_tracking_data[column], brain_response_data['brain_activity'])
            correlations[column] = corr
    return correlations

def plot_brain_map(correlations, brain_template_path):
    """Plot brain map colored by correlation values."""
    brain_img = nib.load(brain_template_path)
    brain_data = brain_img.get_fdata()

    # Create a colormap based on correlations
    correlation_values = np.array(list(correlations.values()))
    correlation_map = np.zeros_like(brain_data)

    # Assuming the brain data is 3D and we map correlations to a specific slice
    correlation_map[30, :, :] = correlation_values  # Example: mapping to a specific slice

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_map[30, :, :], cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Brain Map Colored by Correlation with Eye Tracking')
    plt.axis('off')
    plt.show()

def main():
    eye_tracking_file = 'path/to/eye_tracking_data.csv'
    brain_response_file = 'path/to/brain_response_data.csv'
    brain_template_file = 'path/to/brain_template.nii'

    eye_tracking_data = load_eye_tracking_data(eye_tracking_file)
    brain_response_data = load_brain_response_data(brain_response_file)

    correlations = compute_correlations(eye_tracking_data, brain_response_data)
    plot_brain_map(correlations, brain_template_file)

if __name__ == "__main__":
    main()