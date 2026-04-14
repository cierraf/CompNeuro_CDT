import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt
import os

def load_nwb_data(nwb_file_path):
    """
    Load NWB data from the specified file path.
    """
    # Load the NWB file using MNE
    nwb_data = mne.io.read_raw_nwb(nwb_file_path, preload=True)
    return nwb_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_lfp(data, lowcut, highcut, fs):
    """
    Apply a bandpass filter to the LFP data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def preprocess_lfp(nwb_file_path, lowcut=1.0, highcut=100.0):
    """
    Preprocess LFP data from the NWB file.
    """
    # Load the NWB data
    raw = load_nwb_data(nwb_file_path)
    
    # Get the sampling frequency
    fs = raw.info['sfreq']
    
    # Filter the LFP data
    lfp_data = raw.get_data()
    filtered_lfp = filter_lfp(lfp_data, lowcut, highcut, fs)
    
    return filtered_lfp

def save_filtered_lfp(filtered_lfp, output_path):
    """
    Save the filtered LFP data to a specified output path.
    """
    np.save(output_path, filtered_lfp)

if __name__ == "__main__":
    # Example usage
    nwb_file_path = 'path/to/nwb_file.nwb'
    output_path = 'path/to/filtered_lfp.npy'
    
    filtered_lfp = preprocess_lfp(nwb_file_path)
    save_filtered_lfp(filtered_lfp, output_path)