import os
import librosa
import numpy as np
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, PROCESSED_DATA_PATH


def load_audio(filename, sr=SAMPLE_RATE):
    """Load an audio file."""
    y, sr = librosa.load(filename, sr=sr, mono=False)
    return y, sr


def audio_to_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert audio to its spectrogram representation."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert to decibels
    return D


def process_musdb18_data(musdb18_path, is_train=True):
    """Process the MUSDB18 dataset into spectrograms."""
    # Define the folder for processed data
    data_type = 'train' if is_train else 'test'
    output_path = os.path.join(PROCESSED_DATA_PATH, data_type)
    os.makedirs(output_path, exist_ok=True)

    # Loop through each song in the dataset (train/test folders)
    for song_folder in os.listdir(os.path.join(musdb18_path, data_type)):
        song_path = os.path.join(musdb18_path, data_type, song_folder)

        if not os.path.isdir(song_path):
            continue
        
        # Read the mixture track (the combined source)
        mixture_path = os.path.join(song_path, 'mixture.wav')
        mixture_audio, _ = load_audio(mixture_path)

        # Compute the spectrogram for the mixture track
        mixture_stft = audio_to_stft(mixture_audio)

        # Initialize a list to store the separated stems spectrograms
        stems_stfts = []

        # Loop through each stem (vocals, drums, bass, etc.)
        stems = ['vocals', 'drums', 'bass', 'other']
        for stem in stems:
            stem_path = os.path.join(song_path, f'{stem}.wav')
            if os.path.exists(stem_path):
                stem_audio, _ = load_audio(stem_path)
                stem_stft = audio_to_stft(stem_audio)
                stems_stfts.append(stem_stft)

        # Ensure there are as many stems as expected
        if len(stems_stfts) != len(stems):
            print(f"Warning: Missing stems for {song_folder}. Skipping...")
            continue

        # Convert lists to numpy arrays
        X_data = np.expand_dims(mixture_stft, axis=-1)  # (height, width, 1)
        Y_data = np.stack(stems_stfts, axis=-1)         # (height, width, num_stems)

        # Save the processed data for this song
        song_output_path = os.path.join(output_path, song_folder)
        os.makedirs(song_output_path, exist_ok=True)

        np.save(os.path.join(song_output_path, 'X.npy'), X_data)  # Save mixture spectrogram
        np.save(os.path.join(song_output_path, 'Y.npy'), Y_data)  # Save separated stems spectrogram

    print(f"Data for {data_type} set processed and saved to {PROCESSED_DATA_PATH}")


def preprocess_musdb18(musdb18_path):
    """Preprocess the entire MUSDB18 dataset."""
    # Process training and validation sets (you can split based on your preference)
    process_musdb18_data(musdb18_path, is_train=True)
    process_musdb18_data(musdb18_path, is_train=False)


if __name__ == "__main__":
    musdb18_path = 'path_to_musdb18_dataset'  # Path to your MUSDB18 dataset
    preprocess_musdb18(musdb18_path)
