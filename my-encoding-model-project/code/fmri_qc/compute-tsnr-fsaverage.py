import numpy as np
import nibabel as nib
import os
import argparse

def compute_tsnr_fsaverage(fmriprep_dir, output_dir):
    # Load the fsaverage template
    fsaverage_template = nib.load(os.path.join(fmriprep_dir, 'fsaverage', 'template.nii.gz'))
    
    # Initialize an array to hold tSNR values
    tsnr_values = np.zeros(fsaverage_template.shape)

    # Iterate through each participant's data
    for participant in os.listdir(fmriprep_dir):
        participant_dir = os.path.join(fmriprep_dir, participant)
        if os.path.isdir(participant_dir):
            # Load the participant's preprocessed fMRI data
            fmri_data = nib.load(os.path.join(participant_dir, 'func', f'{participant}_task-movie_space-fsaverage_desc-preproc_bold.nii.gz'))
            fmri_data_array = fmri_data.get_fdata()

            # Compute the mean and standard deviation across time
            mean_signal = np.mean(fmri_data_array, axis=-1)
            std_signal = np.std(fmri_data_array, axis=-1)

            # Compute tSNR
            tsnr = mean_signal / std_signal
            tsnr_values += tsnr

    # Average tSNR values across participants
    tsnr_values /= len(os.listdir(fmriprep_dir))

    # Save the tSNR values to the output directory
    tsnr_img = nib.Nifti1Image(tsnr_values, fsaverage_template.affine)
    nib.save(tsnr_img, os.path.join(output_dir, 'tsnr_fsaverage.nii.gz'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute tSNR values for fMRI data normalized to fsaverage.')
    parser.add_argument('--fmriprep_dir', type=str, required=True, help='Directory containing fMRIprep processed data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the tSNR output.')
    
    args = parser.parse_args()
    compute_tsnr_fsaverage(args.fmriprep_dir, args.output_dir)