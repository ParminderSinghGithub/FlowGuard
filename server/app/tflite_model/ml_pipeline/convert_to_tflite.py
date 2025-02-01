import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.initializers import Orthogonal
import h5py
import json

def fix_batch_shape(h5_model_path):
    """
    Fixes the batch_shape issue in the HDF5 model configuration.
    """
    with h5py.File(h5_model_path, 'r+') as f:
        model_config = f.attrs.get('model_config')
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        model_config = model_config.replace('"batch_shape":', '"batch_input_shape":')
        f.attrs.modify('model_config', model_config.encode('utf-8') if isinstance(model_config, str) else model_config)

def fix_seed_argument(h5_model_path):
    """
    Fixes the seed argument issue in the HDF5 model configuration.
    """
    with h5py.File(h5_model_path, 'r+') as f:
        model_config = f.attrs.get('model_config')
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        model_config_dict = json.loads(model_config)
        
        def clean_dict(d):
            if isinstance(d, dict):
                d.pop('seed', None)
                for key in d:
                    clean_dict(d[key])
            elif isinstance(d, list):
                for item in d:
                    clean_dict(item)
        
        clean_dict(model_config_dict)
        cleaned_config = json.dumps(model_config_dict)
        
        f.attrs.modify('model_config', cleaned_config.encode('utf-8') if isinstance(cleaned_config, str) else cleaned_config)

def fix_optimizer_config(h5_model_path):
    """
    Fixes unsupported arguments in the optimizer configuration.
    """
    with h5py.File(h5_model_path, 'r+') as f:
        training_config = f.attrs.get('training_config')
        if isinstance(training_config, bytes):
            training_config = training_config.decode('utf-8')
        training_config_dict = json.loads(training_config)
        
        # Remove unsupported optimizer arguments
        optimizer_config = training_config_dict['optimizer_config']
        unsupported_args = ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'loss_scale_factor', 'gradient_accumulation_steps']
        for arg in unsupported_args:
            optimizer_config['config'].pop(arg, None)
        
        cleaned_config = json.dumps(training_config_dict)
        
        f.attrs.modify('training_config', cleaned_config.encode('utf-8') if isinstance(cleaned_config, str) else cleaned_config)

def convert_to_tflite(h5_model_path, tflite_model_path):
    """
    Converts an HDF5 model to TensorFlow Lite format.
    
    Args:
        h5_model_path (str): Path to the saved .h5 model.
        tflite_model_path (str): Path to save the .tflite model.
    """
    # Fix the batch_shape, seed, and optimizer issues
    fix_batch_shape(h5_model_path)
    fix_seed_argument(h5_model_path)
    fix_optimizer_config(h5_model_path)

    # Load the HDF5 model with custom objects
    custom_objects = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "DTypePolicy": tf.keras.mixed_precision.Policy,
        "OrthogonalInitializer": Orthogonal
    }
    model = load_model(h5_model_path, custom_objects=custom_objects)
    
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set supported ops and disable lowering tensor list ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    # Save the TFLite model to the specified path
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

if __name__ == "__main__":
    # Example usage
    h5_model_path = "traffic_lstm_model.h5"
    tflite_model_path = "traffic_lstm_model.tflite"
    
    # Convert the model
    convert_to_tflite(h5_model_path, tflite_model_path)
