import librosa
import numpy as np

def spectrogram_to_audio(stft_matrix, hop_length=512, n_fft=2048):
    """Convert a spectrogram back to audio."""
    # Inverse STFT
    audio = librosa.istft(stft_matrix, hop_length=hop_length, length=None)
    return audio

# Example usage for the separated stems
def save_audio_from_stft(stft, output_path, sr=22050):
    audio = spectrogram_to_audio(stft)
    librosa.output.write_wav(output_path, audio, sr)

# Save the separated audio
save_audio_from_stft(separated_stem_stft, 'output/separated_vocals.wav')
