import tensorflow as tf
import numpy as np
from data_loader import get_train_val_data  # For validation data (use a separate test set in real-world case)
from model import unet_model
from config import MODEL_PATH
from bss_eval import bss_eval_sources  # For evaluating with BSS Eval metrics

def evaluate():
    # Load data (use validation or a dedicated test set)
    X_test, Y_test = get_train_val_data()[2], get_train_val_data()[3]  # Use validation data (or X_test, Y_test if available)

    # Load the trained model
    model = tf.keras.models.load_model(f"{MODEL_PATH}/music_source_separation_model.h5")

    # Perform inference on test data
    Y_pred = model.predict(X_test)

    # Evaluate the model on loss and MAE
    loss, mae = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

    # Calculate BSS Eval metrics (SDR, SIR, SAR)
    sdr, sir, sar, _ = bss_eval_sources(Y_test, Y_pred)
    print(f"SDR: {np.mean(sdr)}, SIR: {np.mean(sir)}, SAR: {np.mean(sar)}")

    # Optionally, save the evaluation results to a file
    with open(f"{MODEL_PATH}/evaluation_results.txt", 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test MAE: {mae}\n")
        f.write(f"SDR: {np.mean(sdr)}\n")
        f.write(f"SIR: {np.mean(sir)}\n")
        f.write(f"SAR: {np.mean(sar)}\n")
    
    print("Evaluation complete!")

if __name__ == '__main__':
    evaluate()
