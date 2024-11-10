# inference.py
import numpy as np
import tensorflow as tf
from data_loader import load_audio, audio_to_stft, stft_to_audio
from model import unet_model
from config import MODEL_PATH

def separate_sources(model, mixed_audio):
    """Separate sources from mixed audio."""
    mixed_stft = audio_to_stft(mixed_audio)
    mixed_stft = mixed_stft[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    # Predict the separated sources
    predicted_stems = model.predict(mixed_stft)
    
    separated_audio = []
    for i in range(predicted_stems.shape[-1]):
        stem_stft = predicted_stems[0, ..., i]  # Extract each stem
        separated_audio.append(stft_to_audio(stem_stft))
    
    return separated_audio

def inference(audio_file):
    """Inference pipeline."""
    # Load the trained model
    model = tf.keras.models.load_model(f"{MODEL_PATH}/music_source_separation_model.h5")

    # Load the mixed audio
    mixed_audio, sr = load_audio(audio_file)

    # Separate sources
    separated_audio = separate_sources(model, mixed_audio)

    # Save the separated audio files (e.g., vocals, bass, etc.)
    for i, audio in enumerate(separated_audio):
        output_filename = f"results/separated_stem_{i+1}.wav"
        import soundfile as sf
        sf.write(output_filename, audio, sr)

if __name__ == '__main__':
    inference('path_to_mixed_audio.wav')
