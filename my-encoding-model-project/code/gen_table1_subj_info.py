# This script generates a table summarizing patient demographics and recording information.

import pandas as pd
import os

def generate_subject_info_table(nwb_input_dir, bids_datadir):
    # Initialize lists to hold subject information
    subject_info = []

    # Process NWB files
    nwb_files = [f for f in os.listdir(nwb_input_dir) if f.endswith('.nwb')]
    for nwb_file in nwb_files:
        # Extract subject ID and other relevant information from NWB file
        subject_id = nwb_file.split('_')[0]  # Assuming subject ID is part of the filename
        # Here you would add code to extract more information from the NWB file
        # For demonstration, we will use placeholder values
        subject_info.append({
            'Subject ID': subject_id,
            'NWB File': nwb_file,
            'Recording Quality': 'Placeholder',  # Replace with actual extraction logic
            'Additional Info': 'Placeholder'  # Replace with actual extraction logic
        })

    # Process BIDS files
    bids_subjects = [d for d in os.listdir(bids_datadir) if os.path.isdir(os.path.join(bids_datadir, d))]
    for subject in bids_subjects:
        # Here you would add code to extract relevant information from BIDS files
        # For demonstration, we will use placeholder values
        subject_info.append({
            'Subject ID': subject,
            'NWB File': 'N/A',
            'Recording Quality': 'Placeholder',  # Replace with actual extraction logic
            'Additional Info': 'Placeholder'  # Replace with actual extraction logic
        })

    # Create a DataFrame from the subject information
    df = pd.DataFrame(subject_info)

    # Save the DataFrame to a CSV file
    output_file = os.path.join(nwb_input_dir, 'subject_info_table.csv')
    df.to_csv(output_file, index=False)
    print(f'Subject information table saved to {output_file}')

if __name__ == "__main__":
    # Example usage
    nwb_input_dir = '/path/to/nwb_files/'  # Replace with actual path
    bids_datadir = '/path/to/bids_files/'   # Replace with actual path
    generate_subject_info_table(nwb_input_dir, bids_datadir)