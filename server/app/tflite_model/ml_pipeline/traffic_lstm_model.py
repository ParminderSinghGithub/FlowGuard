import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib  # For saving and loading the scaler

def sequence_generator(df, sequence_length=3, batch_size=32):
    """
    Generates batches of sequences for training or validation.
    """
    num_samples = len(df) - sequence_length
    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = range(start, end)
            X = np.array([df[['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3']].iloc[i:i + sequence_length].values for i in batch_indices])
            y = np.array([df['Vehicles_Normalized'].iloc[i + sequence_length] for i in batch_indices])
            yield X, y

def build_simple_lstm_model(input_shape):
    """
    Builds and compiles a simple and efficient LSTM model.
    """
    model = tf.keras.Sequential([
        # Single LSTM Layer
        tf.keras.layers.LSTM(64, activation='tanh', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),  # Regularization

        # Fully Connected Layers
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

if __name__ == "__main__":
    # Load feature-engineered data
    feature_engineered_data_path = "feature_engineered_data.csv"
    df = pd.read_csv(feature_engineered_data_path)
    
    # Train-test split (80% training, 20% testing)
    train_size = int(0.8 * len(df))
    df_train = df[:train_size]
    df_test = df[train_size:]

    # Fit MinMaxScaler on the 'Vehicles' column of the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train['Vehicles_Normalized'] = scaler.fit_transform(df_train[['Vehicles']])

    # Save the fitted scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")

    # Normalize the test data using the same scaler
    df_test['Vehicles_Normalized'] = scaler.transform(df_test[['Vehicles']])

    # Generate lag and rolling mean features for training and testing data
    for df_set in [df_train, df_test]:
        df_set['Lag_1'] = df_set['Vehicles_Normalized'].shift(1)
        df_set['Lag_2'] = df_set['Vehicles_Normalized'].shift(2)
        df_set['Lag_3'] = df_set['Vehicles_Normalized'].shift(3)
        df_set['Rolling_Mean_3'] = df_set['Vehicles_Normalized'].rolling(window=3).mean()
        df_set.dropna(inplace=True)

    # Determine input shape for the model
    sequence_length = 3
    input_shape = (sequence_length, 4)  # 4 features: 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3'

    # Build the simple LSTM model
    simple_lstm_model = build_simple_lstm_model(input_shape)

    # Create generators for training and validation data
    batch_size = 32
    train_generator = sequence_generator(df_train, sequence_length, batch_size)
    val_generator = sequence_generator(df_test, sequence_length, batch_size)

    # Calculate steps per epoch
    steps_per_epoch = (len(df_train) - sequence_length) // batch_size
    validation_steps = (len(df_test) - sequence_length) // batch_size

    # Train the model with advanced callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_simple_lstm_model.keras', save_best_only=True, monitor='val_loss', mode='min'
    )

    history = simple_lstm_model.fit(
        train_generator,
        epochs=50,  # Fewer epochs for simplicity
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the final trained model
    simple_lstm_model.save("traffic_lstm_model.h5")
    print("Simple LSTM model saved as 'traffic_simple_lstm_model.h5'")
