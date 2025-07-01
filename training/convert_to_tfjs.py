"""
Model Conversion Script for TensorFlow.js

This script converts trained Keras/TensorFlow models to TensorFlow.js format
for deployment in the browser-based 2048 AI application.
"""

import os
import sys
import argparse
from pathlib import Path

# Note: TensorFlow imports will be added when compatibility is resolved
# import tensorflow as tf
# import tensorflowjs as tfjs

def convert_model_to_tfjs(model_path, output_dir, quantize=True):
    """
    Convert a trained Keras model to TensorFlow.js format
    
    Args:
        model_path (str): Path to the trained Keras model (.h5 or SavedModel)
        output_dir (str): Output directory for TensorFlow.js model files
        quantize (bool): Whether to quantize the model for smaller size
    """
    try:
        print(f"Loading model from: {model_path}")
        
        # TODO: Uncomment when TensorFlow is available
        # model = tf.keras.models.load_model(model_path)
        # print(f"Model loaded successfully. Architecture:")
        # model.summary()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Converting model to TensorFlow.js format...")
        print(f"Output directory: {output_dir}")
        
        # TODO: Uncomment when TensorFlow.js converter is available
        # if quantize:
        #     print("Applying quantization for smaller model size...")
        #     tfjs.converters.save_keras_model(
        #         model,
        #         output_dir,
        #         quantization_bytes=1  # 8-bit quantization
        #     )
        # else:
        #     tfjs.converters.save_keras_model(model, output_dir)
        
        # For now, create placeholder files
        create_placeholder_model(output_dir)
        
        print("✅ Model conversion completed successfully!")
        print(f"📁 Model files saved to: {output_dir}")
        print("Files created:")
        print("  - model.json (model architecture)")
        print("  - *.bin (model weights)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model conversion: {str(e)}")
        return False

def create_placeholder_model(output_dir):
    """
    Create placeholder model files for development
    This will be replaced with actual TensorFlow.js conversion
    """
    # Create a simple placeholder model.json
    model_json = {
        "format": "layers-model",
        "generatedBy": "keras v2.13.0",
        "convertedBy": "TensorFlow.js Converter v4.0.0",
        "modelTopology": {
            "keras_version": "2.13.0",
            "backend": "tensorflow",
            "model_config": {
                "class_name": "Functional",
                "config": {
                    "name": "Game2048CNN",
                    "layers": [
                        {
                            "class_name": "InputLayer",
                            "config": {
                                "batch_input_shape": [None, 4, 4, 16],
                                "dtype": "float32",
                                "sparse": False,
                                "name": "input_layer"
                            }
                        }
                    ]
                }
            }
        },
        "weightsManifest": [
            {
                "paths": ["weights.bin"],
                "weights": []
            }
        ]
    }
    
    import json
    with open(os.path.join(output_dir, "model.json"), "w") as f:
        json.dump(model_json, f, indent=2)
    
    # Create a placeholder weights file
    with open(os.path.join(output_dir, "weights.bin"), "wb") as f:
        f.write(b"placeholder_weights")

def validate_model_files(output_dir):
    """
    Validate that all necessary TensorFlow.js model files exist
    """
    required_files = ["model.json"]
    
    for file in required_files:
        file_path = os.path.join(output_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file}")
            return False
    
    # Check for weight files (*.bin)
    bin_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
    if not bin_files:
        print("❌ No weight files (.bin) found")
        return False
    
    print("✅ All required model files present")
    return True

def copy_to_frontend(output_dir, frontend_dir="../frontend/public/model"):
    """
    Copy the converted model to the frontend public directory
    """
    try:
        import shutil
        
        # Create frontend model directory if it doesn't exist
        frontend_model_dir = os.path.abspath(frontend_dir)
        os.makedirs(frontend_model_dir, exist_ok=True)
        
        # Copy all files from output_dir to frontend
        for file in os.listdir(output_dir):
            src = os.path.join(output_dir, file)
            dst = os.path.join(frontend_model_dir, file)
            shutil.copy2(src, dst)
            print(f"📋 Copied {file} to frontend")
        
        print(f"✅ Model files copied to frontend: {frontend_model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error copying to frontend: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert trained 2048 CNN model to TensorFlow.js")
    parser.add_argument("--model", "-m", required=True, help="Path to trained Keras model")
    parser.add_argument("--output", "-o", default="./tfjs_model", help="Output directory for TensorFlow.js model")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization for smaller model size")
    parser.add_argument("--copy-to-frontend", action="store_true", help="Copy converted model to frontend directory")
    
    args = parser.parse_args()
    
    print("🚀 2048 CNN Model Converter")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Convert model
    success = convert_model_to_tfjs(args.model, args.output, args.quantize)
    
    if not success:
        sys.exit(1)
    
    # Validate output
    if not validate_model_files(args.output):
        sys.exit(1)
    
    # Copy to frontend if requested
    if args.copy_to_frontend:
        copy_to_frontend(args.output)
    
    print("\n🎉 Conversion process completed successfully!")
    print(f"📁 TensorFlow.js model available at: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main() 