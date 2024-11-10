import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, PROCESSED_DATA_PATH

def get_train_val_data():
    """Load the preprocessed training and validation data from .npy files."""
    X_train = np.load(f'{PROCESSED_DATA_PATH}/X_train.npy')
    Y_train = np.load(f'{PROCESSED_DATA_PATH}/Y_train.npy')
    X_val = np.load(f'{PROCESSED_DATA_PATH}/X_val.npy')
    Y_val = np.load(f'{PROCESSED_DATA_PATH}/Y_val.npy')
    
    return X_train, Y_train, X_val, Y_val

def load_audio(filename, sr=SAMPLE_RATE):
    """Load an audio file."""
    y, sr = librosa.load(filename, sr=sr, mono=False)  # Keep stereo if applicable
    return y, sr

def audio_to_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert audio to its spectrogram representation."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert to decibels for better dynamic range
    return D

def stft_to_audio(D, hop_length=HOP_LENGTH):
    """Convert the magnitude spectrogram back to audio."""
    D_amp = librosa.db_to_amplitude(D)
    return librosa.istft(D_amp, hop_length=hop_length)

def process_data(raw_data_path, num_stems=5):
    """Process the raw dataset into spectrograms."""
    # Create a folder for processed data if it doesn't exist
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Initialize lists to store data
    X_data = []
    Y_data = []

    # Loop through all songs in the raw data directory
    for song_dir in os.listdir(raw_data_path):
        song_path = os.path.join(raw_data_path, song_dir)
        
        if os.path.isdir(song_path):
            mixed_audio = None
            separated_stems = []

            for file in os.listdir(song_path):
                file_path = os.path.join(song_path, file)
                
                if file.endswith(".wav"):
                    if "mix" in file.lower():  # Find the mixed audio file
                        mixed_audio, sr = load_audio(file_path)
                        mixed_stft = audio_to_stft(mixed_audio)
                        X_data.append(mixed_stft)
                    elif "vocals" in file.lower():
                        vocals, sr = load_audio(file_path)
                        vocals_stft = audio_to_stft(vocals)
                        separated_stems.append(vocals_stft)
                    elif "bass" in file.lower():
                        bass, sr = load_audio(file_path)
                        bass_stft = audio_to_stft(bass)
                        separated_stems.append(bass_stft)
                    elif "drums" in file.lower():
                        drums, sr = load_audio(file_path)
                        drums_stft = audio_to_stft(drums)
                        separated_stems.append(drums_stft)
                    elif "other" in file.lower():
                        other, sr = load_audio(file_path)
                        other_stft = audio_to_stft(other)
                        separated_stems.append(other_stft)

            # Ensure that we have the correct number of stems
            if len(separated_stems) != num_stems:
                print(f"Warning: Expected {num_stems} stems, but found {len(separated_stems)} in {song_dir}.")
                continue  # Skip this song if the number of stems is incorrect
            
            # Append separated stems as the ground truth (Y_data)
            Y_data.append(np.array(separated_stems))

    # Convert to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Save the processed data
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_data.npy'), X_data)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'Y_data.npy'), Y_data)

    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

def create_train_val_split(X_data, Y_data, val_split=0.2):
    """Split the data into training and validation sets."""
    num_samples = len(X_data)
    val_size = int(num_samples * val_split)

    X_train = X_data[val_size:]
    Y_train = Y_data[val_size:]

    X_val = X_data[:val_size]
    Y_val = Y_data[:val_size]

    # Save the split data
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'Y_train.npy'), Y_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'Y_val.npy'), Y_val)

    print(f"Training and validation data split and saved to {PROCESSED_DATA_PATH}")
    
    return X_train, Y_train, X_val, Y_val

def preprocess_and_split_data(raw_data_path):
    """Load, preprocess, and split the data into training and validation sets."""
    # Process data into spectrograms
    process_data(raw_data_path)

    # Load the processed data
    X_data = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_data.npy'))
    Y_data = np.load(os.path.join(PROCESSED_DATA_PATH, 'Y_data.npy'))

    # Split into train and validation
    return create_train_val_split(X_data, Y_data)

if __name__ == "__main__":
    # Example usage:
    raw_data_path = "data/raw"  # Path to the raw dataset
    preprocess_and_split_data(raw_data_path)
