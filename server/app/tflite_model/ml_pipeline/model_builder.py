# tflite_model/ml_pipeline/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os

def inspect_data_shapes(data_dict):
    """Helper function to inspect data shapes"""
    print("\nDATA SHAPE INSPECTION:")
    for name, data in data_dict.items():
        print(f"{name}: shape={data.shape}, dtype={data.dtype}")
    print()

def build_spatiotemporal_model(input_shape=(10, 5)):
    """Build enhanced LSTM model with spatial attention"""
    main_input = tf.keras.Input(shape=input_shape, name='temporal_features')
    
    # Temporal processing branch
    lstm_out = layers.LSTM(64, return_sequences=True)(main_input)
    lstm_out = layers.LSTM(32, return_sequences=False)(lstm_out)  # Ensure no sequence output
    
    # Spatial context branch
    spatial_input = tf.keras.Input(shape=(3,), name='spatial_features')
    spatial_dense = layers.Dense(16, activation='relu')(spatial_input)
    
    # Prepare inputs for attention
    combined = layers.Concatenate()([lstm_out, spatial_dense])
    
    # Reshape for attention (create sequence dimension)
    query_value = layers.Reshape((1, 48))(combined)  # 32+16=48
    
    # Modified attention layer with custom name
    attention_layer = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=16,
        name="custom_multi_head_attention"  # Add this name
    )
    attention_out = attention_layer(query_value, query_value)  # Self-attention
    
    
    # Output layers
    main_output = layers.Dense(32, activation='relu')(attention_out)
    main_output = layers.Dense(1, name='speed_prediction')(main_output)
    
    # Auxiliary output
    aux_output = layers.Dense(1, activation='sigmoid', name='bottleneck_risk')(combined)
    
    model = Model(
        inputs=[main_input, spatial_input],
        outputs=[main_output, aux_output]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'speed_prediction': 'mse', 'bottleneck_risk': 'binary_crossentropy'},
        metrics={'speed_prediction': ['mae'], 'bottleneck_risk': ['accuracy']}
    )
    return model

def load_and_prepare_data():
    """Load and prepare your actual data here"""
    # Dummy data with proper shapes
    temporal_train = np.random.rand(1000, 10, 5).astype(np.float32)
    spatial_train = np.random.rand(1000, 3).astype(np.float32)
    speed_labels = np.random.rand(1000, 1).astype(np.float32)
    bottleneck_labels = np.random.randint(0, 2, (1000, 1)).astype(np.float32)
    
    return temporal_train, spatial_train, speed_labels, bottleneck_labels

def train_model():
    # Load data
    temporal_train, spatial_train, speed_labels, bottleneck_labels = load_and_prepare_data()
    
    # Inspect data shapes before training
    inspect_data_shapes({
        'temporal_train': temporal_train,
        'spatial_train': spatial_train,
        'speed_labels': speed_labels,
        'bottleneck_labels': bottleneck_labels
    })
    
    # Build model
    model = build_spatiotemporal_model(input_shape=(10, 5))
    model.summary()
    
    # Train model
    history = model.fit(
        x=[temporal_train, spatial_train],
        y=[speed_labels, bottleneck_labels],
        epochs=10,  # Reduced for demonstration
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    # Replace the import with direct path definition
    H5_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "traffic_lstm_model.h5")
    os.makedirs(os.path.dirname(H5_MODEL_PATH), exist_ok=True)
    model.save(H5_MODEL_PATH)
    print(f"\nModel saved to {H5_MODEL_PATH}")
    
    return history, H5_MODEL_PATH

if __name__ == "__main__":
    print("Starting model training...")
    history, model_path = train_model()
    print("Training completed successfully!")