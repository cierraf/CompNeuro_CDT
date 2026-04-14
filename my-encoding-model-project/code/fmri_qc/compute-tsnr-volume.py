import numpy as np
import nibabel as nib
import os
import argparse

def compute_tsnr(volume_data):
    """Compute the temporal signal-to-noise ratio (tSNR) for a 4D fMRI volume."""
    mean_signal = np.mean(volume_data, axis=3)
    std_signal = np.std(volume_data, axis=3)
    tsnr = mean_signal / std_signal
    return tsnr

def save_tsnr_to_nifti(tsnr, output_path, affine, header):
    """Save the tSNR data to a NIfTI file."""
    tsnr_img = nib.Nifti1Image(tsnr, affine, header)
    nib.save(tsnr_img, output_path)

def main(fmriprep_dir, output_dir):
    """Main function to compute tSNR for each participant's fMRI data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each participant's directory
    for participant in os.listdir(fmriprep_dir):
        participant_dir = os.path.join(fmriprep_dir, participant)
        if os.path.isdir(participant_dir):
            # Load the 4D fMRI volume (assuming the file naming convention)
            volume_file = os.path.join(participant_dir, 'func', f'{participant}_task-movie_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
            if os.path.exists(volume_file):
                volume_data = nib.load(volume_file).get_fdata()
                tsnr = compute_tsnr(volume_data)

                # Save the tSNR result
                output_file = os.path.join(output_dir, f'{participant}_tsnr.nii.gz')
                affine = nib.load(volume_file).affine
                header = nib.load(volume_file).header
                save_tsnr_to_nifti(tsnr, output_file, affine, header)
                print(f"Computed tSNR for {participant} and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute voxel-wise tSNR for fMRI data.")
    parser.add_argument("--fmriprep_dir", type=str, required=True, help="Directory containing fMRI data processed by fMRIprep.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the computed tSNR NIfTI files.")
    args = parser.parse_args()

    main(args.fmriprep_dir, args.output_dir)