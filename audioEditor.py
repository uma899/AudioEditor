import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

def moving_average_filter(audio_data, window_size):
    """
    Applies a simple moving average low-pass filter to audio data.

    Args:
        audio_data (np.ndarray): The input audio samples. Can be mono or stereo.
        window_size (int): The number of samples to average over.
                           A larger window_size results in more aggressive low-pass filtering.

    Returns:
        np.ndarray: The filtered audio samples.
    """
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

    # Ensure window_size is odd for symmetric filtering around the current sample
    # If even, it biases slightly or requires more complex handling for true center.
    # For simplicity here, we'll just use it directly, but in Verilog, a fixed odd
    # window is often easier.
    
    # Pad the signal to handle edge cases (beginning of the audio)
    # The padding effectively assumes values before the start are zero or repeat the first value.
    # For a causal filter (which you'll implement in Verilog), you'd only use past samples.
    # Here, we do a non-causal average for better visual smoothing.
    
    # Determine the number of dimensions (mono or stereo)
    num_channels = 1
    if audio_data.ndim == 2:
        num_channels = audio_data.shape[1] # Number of columns for stereo

    filtered_audio = np.zeros_like(audio_data, dtype=audio_data.dtype)

    # Calculate half window size for padding and averaging
    half_window = window_size // 2

    # Process each channel independently
    for c in range(num_channels):
        channel_data = audio_data if num_channels == 1 else audio_data[:, c]
        
        # Pad the signal for edge effects. Reflect padding can be better than zero-padding
        # for avoiding artifacts at the start/end, but for Verilog causality, zero-padding
        # (or assuming initial state is 0) is more realistic.
        # Let's use 'edge' padding for better Python simulation results.
        padded_channel_data = np.pad(channel_data, (half_window, half_window), mode='edge')

        for i in range(len(channel_data)):
            # Calculate the average over the window
            # The window spans from (i) to (i + window_size - 1) in the padded data
            window_sum = np.sum(padded_channel_data[i : i + window_size])
            filtered_sample = window_sum // window_size # Integer division for audio samples

            # Store the filtered sample in the correct channel
            if num_channels == 1:
                filtered_audio[i] = filtered_sample
            else:
                filtered_audio[i, c] = filtered_sample

    return filtered_audio


# --- Main script execution ---
if __name__ == "__main__":
    input_audio_file = "petta.wav"  # Make sure you have a WAV file named this
    output_audio_file = "filtered_audio.wav"

    # Create a dummy WAV file if it doesn't exist for demonstration purposes
    if not os.path.exists(input_audio_file):
        print(f"'{input_audio_file}' not found. Creating a dummy sine wave audio file.")
        sample_rate = 44100  # samples per second
        duration = 3.0       # seconds
        frequency = 440      # Hz (A4 note)
        
        t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
        audio_data_float = 0.5 * np.sin(2. * np.pi * frequency * t) # Scale to -0.5 to 0.5
        
        # Convert to 16-bit PCM for typical WAV format
        audio_data_int16 = (audio_data_float * 32767).astype(np.int16)
        
        wavfile.write(input_audio_file, sample_rate, audio_data_int16)
        print(f"Dummy audio file '{input_audio_file}' created.")

    try:
        # Read the audio file
        # samplerate: number of samples per second (e.g., 44100 Hz)
        # audio_data: numpy array of audio samples
        samplerate, audio_data = wavfile.read(input_audio_file)
        print(f"Successfully loaded '{input_audio_file}'.")
        print(f"Sample rate: {samplerate} Hz")
        print(f"Audio data shape: {audio_data.shape}")
        print(f"Audio data type: {audio_data.dtype}")

        # Determine if mono or stereo
        is_stereo = audio_data.ndim == 2

        # Define the filter window size
        # A larger window size means more smoothing (more low-pass effect)
        # This value depends on your audio and desired effect.
        # For typical speech/music, values like 5-21 might be noticeable.
        # For Verilog, keep it a reasonable small odd number like 5, 7, 9 for simple implementation.
        filter_window_size = 900
        print(f"\nApplying moving average filter with window size: {filter_window_size}")

        # Apply the filter
        filtered_audio_data = moving_average_filter(audio_data, filter_window_size)
        print("Filtering complete.")

        # Save the filtered audio
        wavfile.write(output_audio_file, samplerate, filtered_audio_data)
        print(f"Filtered audio saved to '{output_audio_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_audio_file}' was not found. Please ensure it exists.")
        print("A dummy file was created for you. Please run the script again.")
    except Exception as e:
        print(f"An error occurred: {e}")

