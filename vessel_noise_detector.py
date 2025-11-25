# ================================================================================ #
#   Authors: Farid Jedari-Eyvazi                                                   #
#   Contact: farid.je@dal.ca                                                       #
#   Organization: Dalhousie University (https://dal.ca/)                           #
#   Departmnet: Mathematics & Statistics                                           #
#   Date: Sept. 2025                                                               #
#   Description: This script processes acoustic data to detect vessel tonal noises.#
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #




# ================================================================
# ============== Import necessary Python packages ================
# ================================================================

# Standard libraries
import soundfile as sf
import os
import re
import json
import ast
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

# Ketos-specific imports for audio processing and neural networks
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.data_handling.parsing import load_audio_representation
from ketos.neural_networks.resnet import ResNetInterface
from ketos.neural_networks.dev_utils.detection import (
    batch_load_audio_file_data,  
    filter_by_threshold
)

# ================================================================
# ================================================================
# ================================================================


# ================================
# Function: Read WAV File
# ================================

def read_wav(wav_path, cal_dB, start_time, end_time):
    """
    Reads a segment of a WAV file, extracts the audio data, and applies a calibration factor.

    Args:
    - wav_path (str): Path to the WAV file to be read.
    - cal_dB (float): Calibration factor in decibels to adjust the amplitude of the audio data.
    - start_time (float): Start time of the segment to extract in seconds. If None, defaults to the beginning of the file.
    - end_time (float): End time of the segment to extract in seconds. If None, defaults to the end of the file.

    Returns:
    - audio_array (np.ndarray): Calibrated audio data as a NumPy array for the specified segment.
    - n_channels (int): Number of audio channels (e.g., 1 for mono, 2 for stereo).
    - sample_width (int): Width of each audio sample in bytes (1, 2, or 4 bytes).
    - sampling_rate (int): Sampling rate of the audio file in Hz (samples per second).
    - n_frames (int): Number of audio frames in the extracted segment.

    Raises:
    - ValueError: If the sample width is unsupported or if the segment times are invalid.
    - RuntimeError: For general errors encountered while reading the WAV file.
    """
    try:
        # Normalize the path for cross-platform compatibility
        wav_path = os.path.normpath(wav_path)

        # Open the WAV file for reading
        with sf.SoundFile(wav_path, 'r') as audio_file:
            # Extract audio file metadata
            sampling_rate = audio_file.samplerate  # Sampling rate of the audio
            n_channels = audio_file.channels  # Number of audio channels
            subtype = audio_file.subtype  # Audio data subtype (e.g., 'PCM_16')

            # Map audio subtype to sample width in bytes
            subtype_map = {
                'PCM_8': 1,  # 8-bit audio samples
                'PCM_16': 2, # 16-bit audio samples
                'PCM_24': 3, # 24-bit audio samples
                'PCM_32': 4  # 32-bit audio samples
            }
            sample_width = subtype_map.get(subtype, None)
            if sample_width is None:
                raise ValueError(f"Unsupported sample width for subtype: {subtype}")
            
            # Calculate frame indices for the segment
            total_frames = len(audio_file)
            start_frame = int(start_time * sampling_rate)  # Convert start time to frame index
            end_frame = int(end_time * sampling_rate) if end_time else total_frames  # Convert end time to frame index

            # Validate the segment times
            if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
                raise ValueError(f"Invalid segment times: start_time={start_time}s, end_time={end_time}s")

            # Read the specified segment of the audio file
            audio_file.seek(start_frame)  # Move to the start frame
            segment_frames = end_frame - start_frame  # Number of frames to read
            audio_array = audio_file.read(segment_frames)  # Read the audio data

            # Apply calibration factor
            calibration_factor = 10**(np.abs(cal_dB) / 20)  # Compute the calibration factor
            calibrated_audio_array = calibration_factor * audio_array  # Adjust audio data with calibration factor

            return calibrated_audio_array, n_channels, sample_width, sampling_rate, segment_frames

    except Exception as e:
        # Raise a runtime error if an exception occurs
        raise RuntimeError(f"Error reading WAV file '{wav_path}': {e}")


# ======================================================================
# Function: Define Recording Date and Time of a WAV File from its Name
# ======================================================================

def extract_date_time(filename, date_position, time_position):
    """
    Extracts date and time from the given filename based on specified positions.

    Args:
    - filename (str): The filename from which to extract date and time.
    - date_position (tuple): A tuple indicating the start and end indices for the date.
    - time_position (tuple): A tuple indicating the start and end indices for the time.

    Returns:
    - timestamp (int): timestamp from date and time positions within a filename. 
    """
    # Default values
    default_date = '19700101'
    default_time = '000000'

    # Get date_position and time_position indices
    date_start, date_end = date_position
    time_start, time_end = time_position
    
    if date_end > time_start:
        print("Warning: Date and time positions overlap in the filename. Using default values.")
        return 0

    if time_start > time_end:
        print("Warning: The start index for time is greater than the end index. Using default values.")
        return 0

    if date_start > date_end:
        print("Warning: The start index for date is greater than the end index. Using default values.")
        return 0
    
    if date_end > len(filename) or time_end > len(filename):
        print("Error: Index exceeds the length of the filename. Using default values.")
        return 0
    
    # Extract date and time from the filename based on provided positions
    date_str = filename[date_start:date_end + 1]
    time_str = filename[time_start:time_end + 1]
    
    # Validate date and time formats
    date_pattern = r'^\d{8}$|^\d{6}$'  # Matches YYYYMMDD or YYMMDD
    time_pattern = r'^\d{6}$|^\d{4}$'  # Matches HHMMSS or HHMM

    if not re.match(date_pattern, date_str):
        print(f"Warning: Invalid date format '{date_str}'. Using default date: {default_date}.")
        return 0

    if not re.match(time_pattern, time_str):
        print(f"Warning: Invalid time format '{time_str}'. Using default time: {default_time}.")
        return 0

    # Determine the full date and time format
    if len(date_str) == 8:  # YYYYMMDD
        date_format = '%Y%m%d'
    else:  # YYMMDD
        date_format = '%y%m%d'
    
    if len(time_str) == 6:  # HHMMSS
        time_format = '%H%M%S'
    else:  # HHMM
        time_format = '%H%M'

    # Combine date and time for timestamp
    full_datetime_str = f"{date_str}{time_str}"  # e.g., '202209261010' or '2209261010'
    full_datetime_format = date_format + time_format
    
    # Convert to timestamp
    try:
        timestamp = int(datetime.strptime(full_datetime_str, full_datetime_format).astimezone().timestamp())
        return timestamp
    except ValueError:
        print("Error: Could not parse date and time. Using default values.")
        return 0


# =======================================
# Function: Get File Name from Full Path
# =======================================

def get_filename_from_path(wav_path):
    """
    Extracts the filename from a given full file path.

    Args:
    - wav_path (str): The complete path of the file.

    Returns:
    - str: The filename extracted from the full path.
    """
    
    # Start from the end of the string and find the last '/'
    index = len(wav_path) - 1

    while index >= 0:
        if wav_path[index] == '/' or wav_path[index] == '\\':
            # Return the substring after the last separator
            return wav_path[index + 1:]
        index -= 1
    
    # If no '/' or '\\' is found, return the full path (it might be a filename only)
    return wav_path


# =====================================
# Function: Get audio file length
# =====================================

def get_audio_length(file_path):
    """
    Calculate the duration of an audio file in seconds.

    Args:
    - file_path (str): Path to the audio file (e.g., 'path/to/your/audiofile.wav' on Linux OR 'path\to\your\audiofile.wav' on Windows).

    Returns:
    - float: The length of the audio file in seconds.
    """
    
    # Open the audio file using soundfile
    with sf.SoundFile(file_path) as audio_file:
        # Get the sample rate of the audio file (samples per second)
        sample_rate = audio_file.samplerate
        
        # Get the total number of frames (samples) in the audio file
        num_frames = len(audio_file)
        
        # Calculate the length of the audio in seconds
        length = num_frames / sample_rate
    
    return length


# ================================
# Function: Mask Frequencies
# ================================

def mask_freq(data, fs, frequencies_to_mask, freq_bins, cutoff=100, order=5):
    """
    Masks out specific frequencies in the data based on a given list of frequencies and replaces
    the masked frequencies with the average background value of the surrounding data.

    Args:
    - data (np.ndarray): Spectrogram data to be masked.
    - fs (float): The sampling frequency of the data in Hz.
    - frequencies_to_mask (list of int): List of frequencies to be masked.
    - freq_bins (np.ndarray): Array of frequency bins corresponding to the rows of `data`.
    - cutoff (float): The cutoff frequency for the lowpass filter in Hz. Default is 100 Hz.
    - order (int): The order of the lowpass filter. Default is 5.

    Returns:
    - replaced_data (np.ndarray): The spectrogram data with masked frequencies replaced by the average.
    """
    # Create a set of frequencies including ±1 bands around the frequencies to mask
    frequencies_to_mask_extended = {freq for freq in frequencies_to_mask for freq in (freq - 1, freq, freq + 1)}
    
    # Create a boolean mask for the frequencies to be masked
    mask = np.isin(np.round(freq_bins), list(frequencies_to_mask_extended))

    # Ensure the mask is the same shape as the data
    mask = np.tile(mask[:, np.newaxis], (1, data.shape[1]))

    # Create a copy of the data to avoid modifying the original
    replaced_data = data.copy()
    
    # Apply lowpass filter to smooth the entire data
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    
    sos = signal.butter(order, normalized_cutoff, 
                    btype='low', 
                    analog=False, 
                    output='sos')
    # Filter each column separately to get a smoothed background
    smoothed_background = np.apply_along_axis(lambda col: signal.sosfiltfilt(sos, col), axis=0, arr=data)

    # Replace masked values with the smoothed background
    replaced_data[mask] = smoothed_background[mask]
    
    return replaced_data


# ================================
# Function: Load Configuration
# ================================

def load_config(json_file):
    """
    Load configuration parameters from a JSON file.

    Args:
    - json_file (str): Path to the JSON configuration file.

    Returns:
    - config (dict): Dictionary containing configuration parameters.

    Raises:
    - FileNotFoundError: If the JSON file does not exist.
    - json.JSONDecodeError: If the JSON file is invalid.
    """
    try:
        with open(json_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{json_file}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON configuration file '{json_file}': {e}")


# ================================
# Function: Generate Spectrogram
# ================================

def get_spectrogram(signal, sampling_rate, f_min, f_max, time_window=1.5, time_step=0.75):
    """
    Generates a spectrogram for a given signal, focusing on a specific frequency band.

    Args:
    - signal (numpy array): The input signal array.
    - sampling_rate (int): The sampling rate of the signal in Hz.
    - f_min (float): The minimum frequency to include in the spectrogram.
    - f_max (float): The maximum frequency to include in the spectrogram.
    - time_window (float): The length of each time window in seconds.
    - time_step (float): The length of each time step in seconds.

    Returns:
    - filtered_freq_bins (numpy array): Array of frequency bins within the specified range.
    - time_bins (numpy array): Array of time points corresponding to each segment.
    - magnitude (numpy array): Magnitude of the spectrogram within the specified frequency range.
    """
    # Compute the number of samples in each time window
    nperseg = int(time_window * sampling_rate)
    noverlap = int(time_step * sampling_rate)
    step = nperseg - noverlap  
    
    # Segment the signal and compute the FFT
    shape = ((len(signal) - noverlap) // step, nperseg)
    strides = (signal.strides[0] * step, signal.strides[0])
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    # Define a Hanning window function to apply to each segment
    window = np.hanning(nperseg)

    # Apply the window function and compute the FFT
    fft_segments = np.fft.rfft(segments * window, axis=-1)

    # Select indices of frequency bins within the desired frequency range
    frequency_bins = np.fft.rfftfreq(nperseg, 1 / sampling_rate)
    freq_indices = np.logical_and(frequency_bins >= f_min, frequency_bins <= f_max)
    filtered_freq_bins = frequency_bins[freq_indices]

    # Filter the FFT results to include only frequencies within f_min and f_max
    filtered_fft_segments = fft_segments[:, freq_indices]
    magnitude = np.abs(filtered_fft_segments)
    
    # Calculate time bins corresponding to each FFT segment
    time_bins = np.arange(magnitude.shape[0]) * (step / sampling_rate)
    
    return filtered_freq_bins, time_bins, np.transpose(magnitude)


# ================================
# Function: Apply Highpass Filter
# ================================

def apply_highpass_filter(signal_data, critical_freq, sampling_rate, order=5):
    """
    Applies a high-pass Butterworth filter to the input signal using
    second-order sections (SOS) for numerical stability.

    Args:
        signal_data (np.ndarray): Input signal to be filtered.
        critical_freq (float): Cutoff frequency for the high-pass filter in Hz.
        sampling_rate (float): Sampling rate of the input signal in Hz.
        order (int): Order of the Butterworth filter (default = 5).
                     Higher order -> steeper roll-off.

    Returns:
        filtered_signal (np.ndarray): Signal after applying the high-pass filter.
    """

    # Nyquist frequency = half of the sampling rate
    nyquist = 0.5 * sampling_rate

    # Normalize the cutoff frequency (must be between 0 and 1 for butter())
    normalized_cutoff = critical_freq / nyquist

    # Design a high-pass Butterworth filter in SOS (second-order-sections) format
    # SOS format splits the filter into biquad sections, which are much more
    # stable numerically for higher-order filters than (b, a) coefficients.
    sos = signal.butter(order, normalized_cutoff, 
                        btype='high', 
                        analog=False, 
                        output='sos')

    # Apply the filter forward and backward (zero-phase filtering).
    # sosfiltfilt avoids phase distortion while using the stable SOS format.
    filtered_signal = signal.sosfiltfilt(sos, signal_data)

    return filtered_signal


# =====================================
# Function: Detect Peaks Modified FAV
# =====================================

def detect_peaks_modified_FAV(signal_data, min_freq, max_freq, sigma_multiplier, sampling_rate, vessel_type):
    """
    Performs modified Frequency Amplitude Variation (FAV) peak detection.

    Args:
    - signal_data (np.ndarray): Input signal data to be analyzed.
    - min_freq (float): Minimum frequency for analysis.
    - max_freq (float): Maximum frequency for analysis.
    - sigma_multiplier (float): Cutoff value used for normalization.
    - sampling_rate (int): Sampling rate of the input signal.
    - vessel_type (str): Type of vessel affecting the cutoff frequency.

    Returns:
    - peaks (np.ndarray): Detected frequency peaks in the signal.
    """

    # Set critical frequency based on vessel type
    critical_freq = 5000 if vessel_type == "ship" else 2000
    
    # Determine the number of samples based on frequency range
    num_samples = 500 if (max_freq - min_freq) < 500 else int(max_freq - min_freq)

    # Resample the signal to the determined number of samples
    resampled_signal = signal.resample(signal_data, num_samples)
    
    # Apply a high-pass filter to the resampled signal
    filtered_signal = apply_highpass_filter(resampled_signal, critical_freq, sampling_rate, order=5)
    
    # Create a Tukey window to taper the signal
    window_tapper = signal.windows.tukey(num_samples, alpha=0.05)
    
    # Normalize the filtered signal using the specified sigma multiplier
    normalized_signal = (filtered_signal * window_tapper) / (sigma_multiplier * np.std(filtered_signal))
    
    # Identify indices of normalized signal exceeding the threshold
    peaks_idx = np.argwhere(normalized_signal > 1).flatten()
    
    # Calculate the size of each frequency bin
    freq_bin_size = (max_freq - min_freq) / len(normalized_signal)
    
    # Convert indices to actual frequency values
    peaks = min_freq + np.round(peaks_idx * freq_bin_size, 1)

    # Return an empty array if no peaks are found
    if len(peaks) == 0:
        return np.array([])

    # Sort the peaks for processing
    peaks = np.sort(peaks)
    reduced_peaks = []
    
    # Averaging logic for close peaks
    current_peak = peaks[0]
    for peak in peaks[1:]:
        if peak - current_peak < 3 * freq_bin_size:  # Check if within bin size
            current_peak = (current_peak + peak) / 2  # Average the peaks
        else:
            reduced_peaks.append(current_peak)  # Add the current peak to the reduced list
            current_peak = peak  # Update current peak
    reduced_peaks.append(current_peak)  # Append the last processed peak

    return np.array(reduced_peaks)  # Return the array of reduced peaks


# ================================
# Function: Detect Peaks FAV
# ================================

def detect_peaks_FAV(signal_data, min_freq, max_freq):
    """
    Performs Frequency Amplitude Variation (FAV) peak detection.

    Args:
    - signal_data (np.ndarray): Input signal data.
    - min_freq (float): Minimum frequency for analysis (not used in current function).
    - max_freq (float): Maximum frequency for analysis (not used in current function).

    Returns:
    - peaks (np.ndarray): Detected frequency peaks in the signal.
    """
    # Define parameters
    smoothing_window_size = 7
    shift_size = 3
    threshold_multiplier = 1.5
    
    # Convert signal to decibels
    # Avoid taking log of zero or negative values by adding a small constant
    signal_data = 20 * np.log10(np.maximum(signal_data, 1e-10))
    
    # Pad signal to handle boundary effects during smoothing
    padded_signal = np.concatenate([
        signal_data[smoothing_window_size-1:0:-1],  # Left padding
        signal_data,  # Original signal
        signal_data[-2:-smoothing_window_size-1:-1]  # Right padding
    ])

    # Create smoothing filter using Blackman window
    smoothing_filter = np.blackman(smoothing_window_size)
    smoothing_filter /= smoothing_filter.sum()
    
    # Smooth the signal
    smoothed_signal = np.convolve(padded_signal, smoothing_filter, mode='valid')
    
    # Calculate standard deviation of the smoothed signal
    std = np.std(smoothed_signal)
    
    # Compute the first derivative of the smoothed signal
    diff_signal = np.diff(smoothed_signal)
    
    # Cube the differences to enhance peaks
    cubed_diff = np.power(diff_signal, 3)
    
    # Invert and multiply to emphasize positive changes
    y = -cubed_diff[3:]  # Invert the cubic differences
    y = np.concatenate([y, np.zeros(shift_size)])  # Append zeros to match length
    cubed_diff *= y  # Emphasize positive changes
    cubed_diff *= (cubed_diff > 0)  # Keep only positive values
    
    # Apply thresholding to isolate significant peaks
    final_signal = cubed_diff * (cubed_diff > threshold_multiplier * std**2)
    
    # Detect peaks by comparing to neighbors
    y1 = np.concatenate([final_signal[1:], [0]])  # Shift right
    y2 = np.concatenate([[0], final_signal[:-1]])  # Shift left
    peaks_idx = np.argwhere(np.logical_and(final_signal > y1, final_signal >= y2)).flatten()
    
    freq_bin_size = (max_freq-min_freq) / len(final_signal) 
    peaks = min_freq + np.round(peaks_idx * freq_bin_size, 1)
   
    # Return an empty array if no peaks are found
    if len(peaks) == 0:
        return np.array([])

    # Sort the peaks for processing
    peaks = np.sort(peaks)
    reduced_peaks = []
    
    # Averaging logic for close peaks
    current_peak = peaks[0]
    for peak in peaks[1:]:
        if peak - current_peak < 3 * freq_bin_size:  # Check if within bin size
            current_peak = (current_peak + peak) / 2  # Average the peaks
        else:
            reduced_peaks.append(current_peak)  # Add the current peak to the reduced list
            current_peak = peak  # Update current peak
    reduced_peaks.append(current_peak)  # Append the last processed peak

    return np.array(peaks)  # Return the array of reduced peaks


# ===========================================
# Function: Calculate SPL per frequency band
# ===========================================

def get_spl(audio, sampling_rate, f_min, f_max, exclude_freqs=None):
    """
    Calculates the Sound Pressure Level (SPL) for a specific frequency band, 
    optionally excluding certain frequencies.

    Args:
        audio (np.ndarray): Input audio signal.
        sampling_rate (int): Sampling rate of the audio signal.
        f_min (float): Minimum frequency for SPL calculation.
        f_max (float): Maximum frequency for SPL calculation.
        exclude_freqs (list of floats): List of frequencies to exclude, e.g., [2000, 4000].

    Returns:
        spl (float): Sound Pressure Level in dB re. 1 µPa.
    """
    nyquist = 0.5 * sampling_rate

    # --- Band-pass filter for the desired frequency band ---
    sos_bandpass = signal.butter(2, [f_min / nyquist, f_max / nyquist], 
                                 btype='band', output='sos')

    # Detrend to remove DC
    audio_detrended = signal.detrend(audio, type='constant')

    # Apply band-pass filter
    filtered_audio = signal.sosfiltfilt(sos_bandpass, audio_detrended)

    # Apply Tukey window
    num_samples = len(filtered_audio)
    window_tapper = signal.windows.tukey(num_samples, alpha=0.1, sym=True)
    padded_audio = filtered_audio * window_tapper

    # Calculate RMS and SPL
    rms = np.sqrt(np.mean(padded_audio**2))
    spl = 20 * np.log10(rms / 1) if rms > 0 else -np.inf

    # --- Exclusion of specific frequencies ---
    if exclude_freqs:
        band_width = 2  # Hz width for notch
        for freq in exclude_freqs:
            low = (freq - band_width / 2) / nyquist
            high = (freq + band_width / 2) / nyquist
            sos_notch = signal.butter(2, [low, high], btype='bandstop', output='sos')
            padded_audio = signal.sosfiltfilt(sos_notch, padded_audio)

        # Reapply window
        padded_audio *= window_tapper

        # Recalculate RMS and SPL
        rms = np.sqrt(np.mean(padded_audio**2))
        spl = 20 * np.log10(rms / 1) if rms > 0 else -np.inf

    return spl


# ====================================================
# Function: Calculate SNLD per detected peak frequency
# ====================================================

def compute_snld(segment, target_freqs, sampling_rate, freq_bandwidth=3, surrounding_bandwidth=3):
    """
    Computes the Signal-to-Noise Ratio (SNLD) for specified target frequencies in an acoustic signal segment.
    
    Args:
    - segment: The acoustic signal data (array-like).
    - target_freqs: List or array of target frequencies for which SNLD is to be computed.
    - sampling_rate: The sampling rate of the acoustic signal.
    - freq_bandwidth: Bandwidth around each target frequency to measure signal power (default is 2 Hz).
    - surrounding_bandwidth: Bandwidth around each target frequency to measure background noise power (default is 5 Hz).

    Returns:
    - List of SNLD values (one for each target frequency) in linear scale.
    """
    snld = []  # Initialize the list to hold SNLD values for each target frequency
    
    for freq in target_freqs:
        # Define frequency ranges for signal and background noise
        f_s = freq - freq_bandwidth / 2 
        f_e = freq + freq_bandwidth / 2 
        
        # Measure the signal power in the frequency band around the target frequency
        peak_spl = get_spl(segment, sampling_rate, f_s, f_e, exclude_freqs=None)
        
        # Measure background noise power in the frequency band below the target frequency
        f_s_lower = freq - freq_bandwidth / 2 - surrounding_bandwidth
        f_e_lower = freq - freq_bandwidth / 2 
        spl_lower_bg = get_spl(segment, sampling_rate, f_s_lower, f_e_lower, exclude_freqs=None)
        
        # Measure background noise power in the frequency band above the target frequency
        f_s_upper = freq + freq_bandwidth / 2 
        f_e_upper = freq + freq_bandwidth / 2 + surrounding_bandwidth
        spl_upper_bg = get_spl(segment, sampling_rate, f_s_upper, f_e_upper, exclude_freqs=None)
        
        # Average background noise power
        background_spl = np.mean([spl_upper_bg, spl_lower_bg])
        
        # Compute SNLD
        if background_spl > 0:
            snld_ = (peak_spl - background_spl)  # SNLD in logarithmnic scale
        else:
            snld_ = np.inf  # Handle cases where background SPL is zero by assigning infinity
        
        snld.append(snld_)
        
    return snld


# ======================================
# Timestamp to DateTime Function
# ======================================

def timestamp2datetime(timestamp_):
    """
    Convert a Unix timestamp to a formatted datetime string.
    
    Args:
    - timestamp_ (float): A Unix timestamp representing seconds since the epoch.
    
    Returns:
    - str: A formatted datetime string in the format "YYYY-MM-DD HH:MM:SS".
    """
    # Convert the timestamp to a datetime object
    date_time_obj = datetime.fromtimestamp(timestamp_)
    
    # Format the datetime object to a string
    return date_time_obj.strftime("%Y-%m-%d %H:%M:%S")


# ======================================
# DateTime to Timestamp Function
# ======================================

def datetime2timestamp(date_time):
    """
    Convert a formatted datetime string to a Unix timestamp.
    
    Args:
    - date_time (str): A datetime string in the format "YYYY-MM-DD HH:MM:SS".
    
    Returns:
    - float: A Unix timestamp representing seconds since the epoch.
    
    Raises:
   -  ValueError: If the input string is not in the expected format.
    """
    # Convert the string to a datetime object
    date_time_obj = datetime.strptime(date_time, "%Y-%m-%d %H:%M")
    
    # Return the timestamp of the datetime object
    return date_time_obj.timestamp()

    
# ======================================
# Get AIS Timestamps Function
# ======================================

def get_ais_timestamps(df_ais):
    """
    Extracts and converts valid AIS timestamps from a DataFrame.
    
    Args:
    - df_ais (pd.DataFrame): DataFrame containing AIS data with a column 'DateTime'.
    
    Returns:
    - list: A list of timestamps converted to a suitable format. 
      Returns None if there's an error (e.g., missing 'DateTime' column).
    """
    try:
        # Check if the 'DateTime' column exists in the DataFrame
        if 'DateTime' not in df_ais.columns:
            raise ValueError("The 'DateTime' column is missing from the AIS table!")

        # Initialize an empty list to store the converted timestamps
        timestamp_ais = []

        # Iterate over each time entry in the 'DateTime' column
        for time in df_ais['DateTime']:
            try:
                # Check if the timestamp is not NaN
                if not pd.isna(time):
                    # Convert the date and time to timestamp format
                    timestamp_ais.append(datetime2timestamp(time))
                else:
                    timestamp_ais.append(np.nan)  # Append NaN missing times
            except Exception as e:
                # In case of invalid format, append NaN
                print(f"Error processing time '{time}': {e}")
                timestamp_ais.append(np.nan)

        return timestamp_ais
    
    except ValueError as e:
        print(f"Error: {e}")
        return None


# ======================================
# Check AIS Data Function
# ======================================

def check_ais(timestamp_ais, recording_date_time, start_time, segment_length):
    """
    Checks if any AIS timestamps fall within a specified time segment.
    
    Args:
    - timestamp_ais (list): List of AIS timestamps.
    - recording_date_time (datetime): The recording date and time to consider.
    - start_time (int): The start time in seconds since the epoch.
    - segment_length (int): The length of the time segment in seconds.
    
    Returns:
    - str: '1' if any timestamp is within the segment, '0' otherwise.
    """
    # Calculate segment start and end times
    segment_start_time = start_time + recording_date_time
    segment_end_time = segment_start_time + segment_length
    
    # Initialize ans to '0' (assuming default is no timestamps found)
    ans = '0'
    
    # Iterate through the timestamps in timestamp_ais
    for time in timestamp_ais:
        # Check if the current timestamp is within the segment
        if segment_start_time <= time <= segment_end_time:
            ans = '1'
            break  # Stop the loop if a timestamp is found within the range
            
    return ans


# ================================
# Deep-Learning Detector
# ================================

def ml_detector(wav_files_path, wav_path_list, spec_config_path, model_path, mode_, score_thr, results_dir,project, temp_folder=os.path.join('.', 'tmp_folder'), batch_size=8):
    """
    Detects vessel noise in audio files using a pre-trained machine learning model.
    The function processes the files in batches and labels predictions with scores above 0.5 as valid detections.

    Args:
    - wav_files_path (str): Directory path containing the WAV files to process.
    - wav_path_list (list): List of specific WAV file paths to process.
    - spec_config_path (str): Path to the JSON configuration file for spectrogram parameters.
    - model_path (str): Path to the trained deep-learning model file.
    - mode_ (str): Mode for saving results (e.g., 'w' for write, 'a' for append).
    - score_thr (float): ML score threshold to define presence of vessel noise.
    - results_dir: Directory to save the output CSV file.
    - temp_folder (str, optional): Directory for storing intermediate results (default is './tmp_folder' on Linux OR '.\tmp_folder' on Windows).
    - batch_size (int, optional): Number of audio segments to process per batch (default is 8).

    Returns:
    - pd.DataFrame: DataFrame containing detections with filename, start time, end time, label, and score.
    """
    wav_path_list = [os.path.basename(path) for path in wav_path_list]  
    
    # Load spectrogram configuration (duration, type, etc.)
    spec_config_ = load_audio_representation(spec_config_path)

    # Load the pre-trained model for vessel noise detection
    model = ResNetInterface.load(model_file=model_path, new_model_folder=temp_folder)

    # Initialize the audio loader for reading and processing WAV files
    audio_loader = AudioFrameLoader(
        path=wav_files_path,                   # Directory containing WAV files
        filename=wav_path_list,                # List of specific WAV files to process
        duration=spec_config_['duration'],     # Duration of each audio segment
        step=None,                             # No overlap between segments (step size is None)
        stop=False,                            # Continue processing till the end of each file
        representation=spec_config_['type'],   # Type of spectrogram (e.g., magnitude)
        representation_params=spec_config_     # Additional spectrogram parameters
    )

    # Initialize an empty DataFrame to store the detections
    detections = pd.DataFrame()

    # Create a generator to load audio data in batches
    batch_generator = batch_load_audio_file_data(loader=audio_loader, batch_size=batch_size)

    # Process each batch of audio data
    for batch_data in batch_generator:
        # Get predictions from the model for the current batch
        batch_predictions = model.run_on_batch(batch_data['data'], return_raw_output=True)

        # Organize the raw predictions into a DataFrame
        raw_output = {
            'filename': batch_data['filename'],  # Filename of the audio segment
            'start': batch_data['start'],        # Start time of the segment
            'end': batch_data['end'],            # End time of the segment
            'score': batch_predictions           # Prediction scores for the segments
        }

        # Apply the threshold filter to include only valid detections
        batch_detections = filter_by_threshold(raw_output, threshold=0.5)

        # Append the filtered detections to the main DataFrame
        detections = pd.concat([detections, batch_detections], ignore_index=True)

    # Modify the labels based on given score threshold
    for i, label in enumerate(detections['label']):
        if label == 0:
            detections.at[i, 'score'] = (1 - detections.at[i, 'score']).round(2)
            if detections.at[i, 'score'] >= score_thr:
                detections.at[i, 'label'] = 1
            else:
                detections.at[i, 'label'] = 0
        else:
            detections.at[i, 'score'] = (detections.at[i, 'score']).round(2)
            if detections.at[i, 'score'] >= score_thr:
                detections.at[i, 'label'] = 1
            else:
                detections.at[i, 'label'] = 0

    # Save the detections to a CSV file
    detections.to_csv(os.path.join('.', results_dir, f'{project}_ML_detections.csv'), mode=mode_, index=False)
    return detections    


# ================================
# Combine ML and MFAV results
# ================================

def merge_ml_mfav_results(df_ml, df_mfav, date_position, time_position):
    """
    Merges Machine Learning (ML) and MFAV detection results into a single DataFrame by comparing the start and end 
    times of detections for each audio file.

    Args:
    - df_ml (DataFrame): DataFrame containing ML detection results, with columns 'filename', 'start', 'end', 'label', and 'score'.
    - df_mfav (DataFrame): DataFrame containing MFAV detection results, with columns 'filename', 'start time', and 'end time'.
    - date_position (tuple): Tuple of two integers representing the start and end indices of the date part in the filename.
    - time_position (tuple): Tuple of two integers representing the start and end indices of the time part in the filename.

    Returns:
    - final_result (DataFrame): DataFrame containing merged results with 'filename', 'start', 'end', 'label', and 'score'.
    """
    
    # Iterate through the filenames of the ML dataframe to transform and adjust detection times

    for i, file in enumerate(df_ml['filename']):

        file_name = get_filename_from_path(file)

        # Extract the recording start time from each file name in Unix timestamp.
        recording_date_time = extract_date_time(file_name, date_position, time_position)
        
        # Convert Unix timestamps to datetime for both start and end times in the ML dataframe
        df_ml.loc[i, 'start time'] = timestamp2datetime(df_ml.loc[i, 'start'] + recording_date_time)
        df_ml.loc[i, 'end time'] = timestamp2datetime(df_ml.loc[i, 'end'] + recording_date_time)

    # Group the ML dataframe by 'filename' and store the grouped data in a dictionary for faster lookup
    df_ml_group_dict = {filename: group for filename, group in df_ml.groupby('filename')}
    
    # Group the MFAV dataframe by 'filename' for efficient processing
    df_mfav_grouped = [group for _, group in df_mfav.groupby('filename')]

    # Initialize an empty list to store the final merged results
    final_result_rows = []

    # Iterate through each grouped MFAV data
    for df_mfav_group in df_mfav_grouped:
        filename = df_mfav_group['filename'].iloc[0]
        # Check if there is a corresponding ML group for the current filename
        df_ml_group = df_ml_group_dict.get(filename, None)
        if df_ml_group is None:
            continue  # Skip if no corresponding ML group is found
        
        # Process each MFAV row and compare with ML rows
        for _, mfav_row in df_mfav_group.iterrows():
            df_ml_group_sub = pd.DataFrame()  # Temporary DataFrame to store ML rows that overlap with MFAV row
            
            # Compare MFAV start and end times with ML detection times
            for _, ml_row in df_ml_group.iterrows():
                # Convert ML and MFAV start/end times to datetime objects
                ml_row_start_time = datetime.strptime(ml_row['start time'], "%Y-%m-%d %H:%M:%S")
                ml_row_end_time = datetime.strptime(ml_row['end time'], "%Y-%m-%d %H:%M:%S")
                mfav_row_start_time = datetime.strptime(mfav_row['start time'], "%Y-%m-%d %H:%M:%S")
                mfav_row_end_time = datetime.strptime(mfav_row['end time'], "%Y-%m-%d %H:%M:%S")

                # Skip if there is no overlap between ML and MFAV time windows
                if (ml_row_end_time <= mfav_row_start_time) or (ml_row_start_time >= mfav_row_end_time):
                    continue

                # Add the matching ML row to the temporary dataframe
                df_ml_group_sub = pd.concat([df_ml_group_sub, pd.DataFrame([ml_row])], axis=0, ignore_index=True)

            # Create the combined row for the final result
            combined_row = {
                'filename': filename,
                'start': mfav_row['start time'],
                'end': mfav_row['end time']
            }

            # If the temporary dataframe has matching ML rows, calculate the label and score
            if not df_ml_group_sub.empty:
                # If any of the ML rows has a label of 1, set the combined label to 1 and calculate the mean score
                if df_ml_group_sub["label"].any() == 1:
                    combined_row["label"] = 1
                    combined_row["score"] = (df_ml_group_sub.loc[df_ml_group_sub["label"] == 1, "score"].mean()).round(2)
                else:
                    # Otherwise, set label to 0 and calculate the mean score for label 0
                    combined_row["label"] = 0
                    combined_row["score"] = (df_ml_group_sub["score"].mean().round(2))
            else:
                # If no matching ML rows are found, set label to 0 and score to NaN
                combined_row["label"] = 0
                combined_row["score"] = np.nan

            # Append the combined row to the result list
            final_result_rows.append(combined_row)

    # Convert the final result rows list into a DataFrame
    final_result = pd.DataFrame(final_result_rows)

    # Return the merged result DataFrame
    return final_result


# ================================
# Main Function
# ================================

def main(args):
    """
    Main function to process WAV files and generate frequency analysis results.

    Args:
    - args (argparse.Namespace): Command-line arguments.
    """
    ###########################################################################################     
    # Retrieve command-line arguments
    ais_filepath_ = os.path.normpath(args.ais_filepath)
    wav_files_path = os.path.normpath(args.data_directory)
    results_dir = os.path.normpath(args.results_directory)
    vessel_type = args.vessel_type
    freq_band = args.freq_band
    sigma_multiplier = args.sen
    cal_dB = args.cal
    mode_ = args.mode
    method_ = args.method
    spec_config_path = os.path.normpath(args.spec_config)
    config_path = os.path.normpath(args.config)
    model_path = os.path.normpath(args.model)
    score_thr = args.score_thr

    # Check if on macOS/Linux and if paths start with .\ to correct them
    if os.name == 'posix':
        if config_path.startswith('.\\'):
            config_path = './' + config_path[2:]
        if ais_filepath_.startswith('.\\'):
            ais_filepath_ = './' + ais_filepath_[2:]
        if wav_files_path.startswith('.\\'):
            wav_files_path = './' + wav_files_path[2:]
        if results_dir.startswith('.\\'):
            results_dir = './' + results_dir[2:]
        if spec_config_path.startswith('.\\'):
            spec_config_path = './' + spec_config_path[2:]
        if model_path.startswith('.\\'):
            model_path = './' + model_path[2:]
    
    # Check if on windows and if paths start with .\\ to correct them       
    elif os.name == 'nt':
        # For Windows
        if config_path.startswith('.\\'):
            config_path = '.' + config_path[1:]
        if ais_filepath_.startswith('.\\'):
            ais_filepath_ = '.' + ais_filepath_[1:]
        if wav_files_path.startswith('.\\'):
            wav_files_path = '.' + wav_files_path[1:]
        if results_dir.startswith('.\\'):
            results_dir = '.' + results_dir[1:]
        if spec_config_path.startswith('.\\'):
            spec_config_path = '.' + spec_config_path[1:]
        if model_path.startswith('.\\'):
            model_path = '.' + model_path[1:]

    # Parse frequency band from command-line argument
    if freq_band:
        try:
            freq_band = ast.literal_eval(freq_band)
            if not isinstance(freq_band, tuple) or len(freq_band) != 2:
                raise ValueError("Frequency band must be a tuple of length 2.")
        except (ValueError, SyntaxError) as e:
            print(f"Invalid frequency band format: {e}. Use a tuple like '(10, 100)'.")
            return
    else:
        freq_band = None
    
    min_band, max_band = (float(freq_band[0]), float(freq_band[1])) if freq_band else (None, None)

    ###########################################################################################     
    # Load configuration settings
    config = load_config(config_path)

    # Set the working parameters' value from the loaded JSON configuration file
    project = (config.get("project_name", "unknown_project"))
    date_position = (config.get("date_position"))
    time_position = (config.get("time_position"))
    ch2use = int(config.get("channel_number"))
    min_freqs = list(map(float, config.get("min_freqs", [])))
    max_freqs = list(map(float, config.get("max_freqs", [])))
    segment_length = float(config.get("segment_length", 60))
    system_noise_frequencies = list(map(float, config.get("system_noise_frequencies", [])))

    # If a frequency band is provided via the '--freq_band' command-line argument, append it to the frequency list.
    if min_band is not None and max_band is not None:
        min_freqs.append(min_band)
        max_freqs.append(max_band)       
    
    ###########################################################################################
    # Display loaded configuration parameters
    print('')
    print('###########################################################################################')
    print('############################# SIGNAL PROCESSING CONFIGURATION #############################')
    print('###########################################################################################')

    print(f"Project: {project}")
    print(f"Configuration file: {config_path}")
    print(f"Channel number: {ch2use}")
    print(f"cal_dB: {cal_dB}")
    print(f"low_end_freqs_list: {min_freqs}")
    print(f"high_end_freqs_list: {max_freqs}")
    print(f"MFAV sensitivity factor: {sigma_multiplier}")
    print(f"segment_length (FAV/MFAV): {segment_length}")
    print(f"system_noise_frequencies: {system_noise_frequencies}")
    print(f"method: {method_}")
    print(f"vessel type: {vessel_type}")
    print(f"ML score threshold: {score_thr}")

    print('###########################################################################################')
    print('###########################################################################################')
    print('###########################################################################################')
    print('')

    ###########################################################################################

    # Create the output directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ###########################################################################################
    
    # Check if the AIS file exists, and if so, extract the timestamps
    timestamp_ais = None  # Initialize timestamp_ais
    if os.path.exists(ais_filepath_): 
        df_ais = pd.read_csv(ais_filepath_)  # Read the AIS file into a DataFrame
        timestamp_ais = get_ais_timestamps(df_ais)  # Extract timestamps from the DataFrame

    ###########################################################################################
    if not method_ == 'ml':
        # Initialize DataFrames to store results
        df_FAV = pd.DataFrame(columns=['filename', 'start time', 'end time', 'freq. band (Hz)', 'number of peaks', 'peaks freq. (Hz)' , 'SPL (dB)', 'SNLD (dB)', 'AIS'])
        df_modified_FAV = pd.DataFrame(columns=['filename', 'start time', 'end time', 'freq. band (Hz)', 'number of peaks', 'peaks freq. (Hz)' , 'SPL (dB)', 'SNLD (dB)', 'AIS'])
        
        ###########################################################################################
        # Gather WAV file paths and process each WAV file
        wav_path_list = [os.path.join(wav_files_path, file) for file in os.listdir(wav_files_path) if file.endswith(".wav")]

        for wav_path in tqdm(wav_path_list, colour='cyan'):
            try:
                
                # Normalize the path for cross-platform compatibility
                wav_path = os.path.normpath(wav_path)

                file_name = get_filename_from_path(wav_path)

                recording_date_time = extract_date_time(file_name, date_position, time_position)

                audio_length = get_audio_length(wav_path)

                # Create segments of the audio based on segment length. Drop the remaining portion of 
                # the audio file if its length is less than 10% of the defined segment length.
                if (audio_length % segment_length) >= 0.10 * segment_length: 
                    segment_list = np.concatenate((np.arange(0, (audio_length - (audio_length % segment_length)), segment_length),
                                                    np.array([(audio_length - segment_length)])))
                else:
                    segment_list = np.arange(0, (audio_length - (audio_length % segment_length)), segment_length)

                # Process each segment
                for start_time in tqdm(segment_list, leave=False):
                    audio_array, n_channels, _, sampling_rate, _ = read_wav(wav_path, cal_dB, start_time, end_time=start_time + segment_length)
                    segment = audio_array[:, ch2use] if n_channels > 1 else audio_array

                    if (start_time == 0) & (np.sqrt(np.mean(segment[0:5*sampling_rate]**2)) > 2 * np.sqrt(np.mean(segment[5*sampling_rate:]**2))):
                        # Jump to the 5 sec. of the aduio file
                        segment = segment[5*sampling_rate:]
                        print(f'>>>>>>>>>> Starting at 5 s to exclude unstable initial records. <<<<<<<<<<')
                        # Convert start and end times to datetime format
                        onset = timestamp2datetime(start_time + 5 + recording_date_time)
                    else:
                        onset = timestamp2datetime(start_time + recording_date_time)
                        
                    termination = timestamp2datetime(start_time + segment_length + recording_date_time)

                    # Check for corresponding AIS data
                    if timestamp_ais != None:
                        ais_ = check_ais(timestamp_ais, recording_date_time, start_time, segment_length)      
                    else:
                        ais_ = "None"

                    # Process each frequency band for FAV and MFAV methods
                    for f_min, f_max in zip(min_freqs, max_freqs):

                        tqdm.write("")  # Adds a blank line for spacing
                        tqdm.write(f">>> Processing file: {wav_path} - Time duration = {onset} to {termination} - segment length = {segment_length} - freq band = {f_min}-{f_max} Hz")

                        # Compute Short-Time Fourier Transform (STFT)
                        f, _, spec_i = get_spectrogram(segment, sampling_rate, f_min, f_max, time_window=1.5, time_step=0.75)

                        # Replace the system noise frequencies with the background energy
                        spec = mask_freq(spec_i, sampling_rate, system_noise_frequencies, freq_bins=f, cutoff=1000, order=5)

                        # Extract signal data within the specified frequency range
                        freq_range = np.logical_and(f >= f_min, f <= f_max)
                        signal_data = np.mean(spec[freq_range, :], axis=1)
                        
                        # Calculate Sound Pressure Level (SPL)
                        spl = np.round(get_spl(segment, sampling_rate, f_min, f_max, system_noise_frequencies))

                        if method_ == 'fav':                  
                            # Detect peaks for FAV
                            peaks_FAV = detect_peaks_FAV(signal_data, f_min, f_max)
                            peaks_FAV_freqs = [peak for peak in peaks_FAV.flatten()]

                            # Compute Signal-to-Noise Level Difference (SNLD)
                            snld = compute_snld(segment, peaks_FAV_freqs, sampling_rate, freq_bandwidth=3, surrounding_bandwidth=3)

                            # Filter valid peaks based on SNLD threshold
                            valid_peaks_FAV_freqs = []
                            valid_snld_FAV = []

                            for peak, s in zip(peaks_FAV_freqs, snld):
                                if s >= 1:
                                    valid_peaks_FAV_freqs.append(peak)
                                    valid_snld_FAV.append(s)

                            count_valid_FAV = len(valid_peaks_FAV_freqs)

                            # Append results to DataFrame if there are valid peaks
                            df_FAV = pd.concat([df_FAV, pd.DataFrame({
                                'filename': [file_name],
                                'start time': [onset],
                                'end time': [termination],
                                'freq. band (Hz)': [f'{f_min} - {f_max}'],
                                'number of peaks': count_valid_FAV,
                                'peaks freq. (Hz)': [valid_peaks_FAV_freqs],
                                'SPL (dB)': [spl],
                                'SNLD (dB)': [np.round(valid_snld_FAV, 2)],
                                'AIS': [ais_]
                            })], ignore_index=True)

                        if (method_ == 'mfav') or (method_ == 'mfav-ml'):  
                            # Detect modified peaks
                            peaks_modified_FAV = detect_peaks_modified_FAV(signal_data, f_min, f_max, sigma_multiplier, sampling_rate, vessel_type)
                            peaks_modified_FAV_freqs = [peak for peak in peaks_modified_FAV.flatten()]

                            # Compute SNLD for modified peaks
                            snld = compute_snld(segment, peaks_modified_FAV_freqs, sampling_rate, freq_bandwidth=3, surrounding_bandwidth=3)

                            # Filter valid peaks for modified detection based on SNLD
                            valid_peaks_modified_FAV_freqs = []
                            valid_snld_modified_FAV = []

                            for peak, s in zip(peaks_modified_FAV_freqs, snld):
                                if s >= 1:
                                    valid_peaks_modified_FAV_freqs.append(peak)
                                    valid_snld_modified_FAV.append(s)

                            count_valid_modified_FAV = len(valid_peaks_modified_FAV_freqs)

                            # Append modified results to DataFrame if there are valid peaks
                            df_modified_FAV = pd.concat([df_modified_FAV, pd.DataFrame({
                                'filename': [file_name],
                                'start time': [onset],
                                'end time': [termination],
                                'freq. band (Hz)': [f'{f_min} - {f_max}'],
                                'number of peaks': count_valid_modified_FAV,
                                'peaks freq. (Hz)': [valid_peaks_modified_FAV_freqs],
                                'SPL (dB)': [spl],
                                'SNLD (dB)': [np.round(valid_snld_modified_FAV, 2)],
                                'AIS': [ais_]
                            })], ignore_index=True)

            except Exception as e:
                print(f"Error processing file '{wav_path}': {e}")

    # Check whether the 'ml' method is specified using the '--method' command-line argument 
    if (method_ == 'ml') or (method_ == 'mfav-ml'):

        tqdm.write("") 
        tqdm.write(">>> Running ML model")

        # Gather WAV file paths and process the WAV files in batches.
        wav_path_list = [os.path.join(wav_files_path, file) for file in os.listdir(wav_files_path) if file.endswith(".wav")]

        # Envoke the deep learning model for detection
        df_ml = ml_detector(wav_files_path, wav_path_list, spec_config_path, model_path, mode_, score_thr, results_dir, project, temp_folder = os.path.join('.', 'tmp_folder'), batch_size = 8)
        
        if method_ == 'mfav-ml':

            # filter the MFAV DataFrame to keep the frequency band taht matches the ml frequency band only
            df_modified_FAV_50_1000Hz = df_modified_FAV[df_modified_FAV['freq. band (Hz)'] == '50.0 - 1000.0']

            if not df_modified_FAV_50_1000Hz.empty:
        
                df_ml_proc = merge_ml_mfav_results(df_ml, df_modified_FAV_50_1000Hz, date_position, time_position)
                df_ml_proc_sorted = df_ml_proc.sort_values(by=['filename', 'start'], ascending=True)
                df_modified_FAV_50_1000Hz_sorted = df_modified_FAV_50_1000Hz.sort_values(by=['filename', 'start time'], ascending=True)
                df_ml_proc_sorted_reset = df_ml_proc_sorted.reset_index(drop=True)
                df_modified_FAV_50_1000Hz_sorted_reset = df_modified_FAV_50_1000Hz_sorted.reset_index(drop=True)

                ## Add the 'ml_label' and 'ml_score' columns to the MFAV DataFrame
                df_modified_FAV_50_1000Hz_sorted_reset['ml_label'] = df_ml_proc_sorted_reset['label']
                df_modified_FAV_50_1000Hz_sorted_reset['ml_score'] = df_ml_proc_sorted_reset['score'].round(2)

                ## Re-sort the final DataFrame by 'filename' and 'start time' to ensure correct order
                df_modified_FAV_50_1000Hz_sorted_aug = df_modified_FAV_50_1000Hz_sorted_reset.sort_values(by=['filename', 'start time'], ascending=True)

    # Save results to CSV files
    if method_ == 'fav':
        df_FAV.to_csv(os.path.join(results_dir, f'{project}_FAV_detections.csv'), mode=mode_, index=False)
    
    if (method_ == 'mfav') or (method_ == 'mfav-ml'):
        df_modified_FAV.to_csv(os.path.join(results_dir, f'{project}_MFAV_detections.csv'),mode=mode_, index=False)
    
    if method_ == 'mfav-ml':
        if not df_modified_FAV_50_1000Hz.empty:
            df_modified_FAV_50_1000Hz_sorted_aug.to_csv(os.path.join(results_dir, f'{project}_MFAV_ML_detections.csv'),mode=mode_, index=False)
    

# ================================
# Script Execution
# ================================

if __name__ == "__main__":
    # Set up the argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Process WAV files for audio analysis.")
    
    # Directory containing WAV files to process
    parser.add_argument(
        'data_directory', 
        type=str,
        help="Path to directory containing WAV files to process."
    )
    
    # Directory to save processed data results
    parser.add_argument(
        'results_directory', 
        type=str,  
        help="Path to directory to save processed data results."
    )
    
    # Directory for AIS data
    parser.add_argument(
        '--ais_filepath', 
        type=str, 
        default=os.path.normpath(os.path.join('.', 'ais', 'ais.csv')), 
        help="Path to the AIS data file (default: './ais/ais.csv')."
    )
    
    # Type of vessel to analyze (default is 'ship')
    parser.add_argument(
        '--vessel_type', 
        type=str, 
        default='ship', 
        help="Type of vessel to detect; options are 'ship' or 'ship/boat' (default: 'ship')."
    )
    
    # ML score threshold to define the presence of vessel noise    
    parser.add_argument(
        '--score_thr', 
        type=float, 
        default=0.5, 
        help="Machine learning score threshold for confirming the presence of vessel noise. For example, set to 0.85 to only consider detections with a score of 0.85 or higher (default: 0.50)."
    )

    # Specific frequency band for analysis provided as a string
    parser.add_argument(
        '--freq_band', 
        type=str, 
        default='', 
        help="Specific frequency band for analysis as a tuple, e.g., '(10, 100)' (default: '')."
    )
    
    # Configuration file for parameters
    parser.add_argument(
        '--config', 
        type=str, 
        default='./config.json', 
        help="Defines the path to the JSON configuration file containing the working parameters (default: 'config.json')."
    )
    
    # Calibration factor for hydrophone sensitivity in decibels
    parser.add_argument(
        '--cal', 
        type=float, 
        default=0, 
        help="Calibration factor, i.e., hydrophone sensitivity in decibels (default: 0)."
    )
    
    # Sensitivity factor for detecting tonal noise
    parser.add_argument(
        '--sen', 
        type=float, 
        default=3.5, 
        help="Sensitivity factor defining the number of standard-deviation of filtered signal - range: 3.5 and above (default: 3.5)."
    )

    # Mode of writing CSV files ('w' for write, 'a' for append)
    parser.add_argument(
        '--mode', 
        type=str, 
        default='a', 
        help="Specify the mode for writing to the CSV file. Use 'w' to overwrite the file or 'a' to append to the existing file. Default is 'a' (append)."
    )

    # Vessel noise detection method (Default is 'mfav').
    parser.add_argument(
        '--method', 
        type=str, 
        default='mfav', 
        help="""Specify the vessel noise detection method.
                select 'ml' to use deep-learning model 
                select 'fav' to use FAV method
                select 'mfav' to use modified_FAV (MFAV) method
                select 'mfav-ml' to invoke both ML and MFAV methods
                (Default is 'mfav')."""
    )
    
    # Path to the spectrogram configuration JSON file used in machine learning (default: './spec_config.json')
    parser.add_argument(
        '--spec_config', 
        type=str, 
        default='./spec_config.json', 
        help="Specify the path to the JSON file containing the spectrogram configuration for the machine learning model. Default is './spec_config.json'."
    )

    # Path to the machine learning model (default: './vessel_detector_dl_model.kt')
    parser.add_argument(
        '--model', 
        type=str, 
        default='./vessel_detector_dl_model.kt', 
        help="Specify the path to the trained deep-learning model. Default is './vessel_detector_dl_model.kt'."
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Try executing the main function with the provided arguments
    try:
        main(args)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}")
        parser.print_help()  # Display help if there's a type error in arguments
    except Exception as e:
        print(f"Unexpected error: {e}")
        parser.print_help()  # Display help for any other unexpected errors
