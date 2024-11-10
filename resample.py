def resample_audio(audio, orig_sr, target_sr):
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

# Example: Resample the output to 44,100 Hz
audio_resampled = resample_audio(output_audio, orig_sr=22050, target_sr=44100)
