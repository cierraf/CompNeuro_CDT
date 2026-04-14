# This script computes inter-subject correlation (ISC) on data projected to fsaverage.

import numpy as np
import nibabel as nib
import os
import argparse

def load_data(data_dir):
    # Load ISC data from the specified directory
    isc_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    isc_data = [np.load(os.path.join(data_dir, f)) for f in isc_files]
    return np.array(isc_data)

def compute_isc(isc_data):
    # Compute the mean ISC across subjects
    return np.mean(isc_data, axis=0)

def save_isc_to_nifti(isc_map, output_path):
    # Save the ISC map as a NIfTI file
    img = nib.Nifti1Image(isc_map, np.eye(4))
    nib.save(img, output_path)

def main(data_dir, output_path):
    isc_data = load_data(data_dir)
    isc_map = compute_isc(isc_data)
    save_isc_to_nifti(isc_map, output_path)
    print(f"ISC map saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ISC on data projected to fsaverage.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing ISC data files.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the ISC NIfTI file.')
    args = parser.parse_args()
    
    main(args.data_dir, args.output_path)