import matplotlib.pyplot as plt

def plot_spectrogram(spectrogram, title="Spectrogram"):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram, x_axis="time", y_axis="log")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.show()

# Visualize a spectrogram
plot_spectrogram(vocal_stft, title="Vocal Spectrogram")
