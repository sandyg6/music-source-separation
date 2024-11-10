# Save model
model.save("models/music_source_separation_model.h5")

# Load model
from tensorflow.keras.models import load_model
model = load_model("models/music_source_separation_model.h5")
