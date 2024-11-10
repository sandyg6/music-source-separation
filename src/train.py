# train.py

import tensorflow as tf
from model import unet_model
from data_loader import get_train_val_data  # Import the function to load data
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH, SAVE_BEST_MODEL

def train():
    # Step 1: Load data
    X_train, Y_train, X_val, Y_val = get_train_val_data()  # This will now work after the update in data_loader.py

    # Step 2: Create the U-Net model
    model = unet_model()

    # Step 3: Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mean_squared_error', metrics=['mae'])

    # Step 4: Setup callbacks for saving the best model and early stopping
    callbacks = []
    
    # Save the best model based on validation loss
    if SAVE_BEST_MODEL:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"{MODEL_PATH}/best_model.h5", 
            monitor='val_loss', 
            save_best_only=True, 
            mode='min', 
            verbose=1
        )
        callbacks.append(checkpoint)

    # Early stopping to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,  # Stop training if validation loss doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Step 5: Train the model
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Step 6: Save the final model after training
    model.save(f"{MODEL_PATH}/music_source_separation_model.h5")

    print("Training complete and model saved!")

if __name__ == '__main__':
    train()
