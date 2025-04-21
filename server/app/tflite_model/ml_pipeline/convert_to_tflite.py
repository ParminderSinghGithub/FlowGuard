# tflite_model/ml_pipeline/convert_to_tflite.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import h5py
import json
import os

def fix_batch_shape(h5_model_path):
    """Fix batch_shape issue in HDF5 model configuration"""
    try:
        with h5py.File(h5_model_path, 'r+') as f:
            if 'model_config' not in f.attrs:
                raise ValueError("No model_config attribute found in HDF5 file")
            
            model_config = f.attrs['model_config']
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            
            updated_config = model_config.replace('"batch_shape":', '"batch_input_shape":')
            f.attrs.modify('model_config', updated_config.encode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed fixing batch shape: {str(e)}")

def fix_seed_argument(h5_model_path):
    """Remove seed arguments from model configuration"""
    try:
        with h5py.File(h5_model_path, 'r+') as f:
            model_config = f.attrs.get('model_config')
            if not model_config:
                raise ValueError("Empty model_config in HDF5 file")
            
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            
            config_dict = json.loads(model_config)
            
            def remove_seed(node):
                if isinstance(node, dict):
                    node.pop('seed', None)
                    for v in node.values():
                        remove_seed(v)
                elif isinstance(node, list):
                    for item in node:
                        remove_seed(item)
            
            remove_seed(config_dict)
            updated_config = json.dumps(config_dict)
            f.attrs.modify('model_config', updated_config.encode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed removing seed arguments: {str(e)}")

def fix_optimizer_config(h5_model_path):
    """Clean optimizer configuration for TFLite compatibility"""
    try:
        with h5py.File(h5_model_path, 'r+') as f:
            training_config = f.attrs.get('training_config')
            if not training_config:
                print("Warning: No training_config found in HDF5 file")
                return
            
            if isinstance(training_config, bytes):
                training_config = training_config.decode('utf-8')
            
            config_dict = json.loads(training_config)
            optimizer_config = config_dict.get('optimizer_config', {})
            
            unsupported_args = [
                'weight_decay', 'use_ema', 'ema_momentum',
                'ema_overwrite_frequency', 'loss_scale_factor',
                'gradient_accumulation_steps'
            ]
            for arg in unsupported_args:
                optimizer_config.get('config', {}).pop(arg, None)
            
            updated_config = json.dumps(config_dict)
            f.attrs.modify('training_config', updated_config.encode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed fixing optimizer config: {str(e)}")

def convert_to_tflite(h5_model_path, tflite_model_path):
    """
    Convert HDF5 model to TensorFlow Lite format with comprehensive validation
    
    Args:
        h5_model_path (str): Path to input .h5 model
        tflite_model_path (str): Path for output .tflite model
    """
    try:
        if not os.path.exists(h5_model_path):
            raise FileNotFoundError(f"Input model not found: {h5_model_path}")
        
        output_dir = os.path.dirname(tflite_model_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Write permission denied for: {output_dir}")

        print(f"Preparing model at {h5_model_path}")
        fix_batch_shape(h5_model_path)
        fix_seed_argument(h5_model_path)
        fix_optimizer_config(h5_model_path)

        print("Loading model with custom objects...")
        custom_objects = {
            "mse": tf.keras.losses.MeanSquaredError(),
            "DTypePolicy": tf.keras.mixed_precision.Policy,
            "OrthogonalInitializer": Orthogonal,
            "custom_multi_head_attention": tf.keras.layers.MultiHeadAttention
        }
        
        try:
            model = load_model(h5_model_path, custom_objects=custom_objects)
        except Exception as load_error:
            raise ValueError(f"Model loading failed: {str(load_error)}") from load_error

        print(f"Successfully loaded model with {len(model.layers)} layers")
        print(f"Model input shape: {model.input_shape}")
        print(f"TensorFlow version: {tf.__version__}")

        print("Configuring TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False

        print("Starting model conversion...")
        try:
            tflite_model = converter.convert()
        except Exception as conv_error:
            raise RuntimeError(f"Conversion failed: {str(conv_error)}") from conv_error

        if not tflite_model:
            raise ValueError("Conversion produced empty model")

        print(f"Saving model to {tflite_model_path}")
        try:
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
        except Exception as save_error:
            raise IOError(f"Failed to save model: {str(save_error)}") from save_error

        if not os.path.exists(tflite_model_path):
            raise RuntimeError("Output file not created after successful conversion")
        if os.path.getsize(tflite_model_path) == 0:
            raise RuntimeError("Output file is empty")

        size_mb = os.path.getsize(tflite_model_path) / 1e6
        print(f"TFLite model saved to {tflite_model_path} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        convert_to_tflite(
            h5_model_path="../models/traffic_lstm_model.h5",
            tflite_model_path="../models/traffic_lstm_model.tflite"
        )
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        exit(1)
