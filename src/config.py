# config.py

import os

# Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH = "models"
RESULTS_PATH = "results"
SAVE_BEST_MODEL = True 

# Hyperparameters
SAMPLE_RATE = 44100  # Audio sample rate
N_FFT = 2048         # FFT size
HOP_LENGTH = 512     # Hop length for STFT
NUM_STEMS = 5        # Number of stems to separate (e.g., vocals, bass, drums, piano, others)
INPUT_SHAPE = (256, 256, 1)  # Input shape for the spectrograms (height, width, channels)

# Training Parameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
