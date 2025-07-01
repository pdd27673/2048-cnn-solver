# 2048 CNN Training

This directory contains the Python scripts for training the CNN model to play 2048.

## 🚨 VIRTUAL ENVIRONMENT REQUIRED

**ALWAYS use a Python virtual environment when working with this project!**

## Quick Setup

### 1. Create and Activate Virtual Environment
```bash
# Navigate to training directory
cd training/

# Create virtual environment (only needed once)
python3 -m venv venv

# Activate virtual environment (required every time)
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows
```

### 2. Install Dependencies
```bash
# Make sure venv is activated first!
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check TensorFlow
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

# Check TensorFlow.js converter
python -c "import tensorflowjs; print('TensorFlow.js converter available')"
```

## Training Commands

Always activate the virtual environment first:
```bash
source venv/bin/activate
```

### Train a Model
```bash
# Basic training
python train_cnn.py --episodes 1000

# Training with custom parameters
python train_cnn.py --episodes 2000 --batch-size 64 --save-interval 100
```

### Convert Model for Web Deployment
```bash
# Convert trained model to TensorFlow.js format
python convert_to_tfjs.py --model models/game2048_cnn.h5 --quantize --copy-to-frontend
```

### Run Tests
```bash
# Run all tests
python -m pytest test_*.py -v

# Run specific test
python test_trainer.py
```

## Files

- `train_cnn.py` - Main training script
- `convert_to_tfjs.py` - Model conversion for web deployment
- `trainer.py` - CNN trainer class implementation
- `model.py` - CNN model architecture
- `game_engine.py` - 2048 game logic for training
- `requirements.txt` - Python dependencies
- `test_*.py` - Unit and integration tests

## Virtual Environment Best Practices

1. **Always activate before working:**
   ```bash
   source venv/bin/activate
   ```

2. **Check if activated:**
   - Your prompt should show `(venv)`
   - `which python` should point to `venv/bin/python`

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

4. **Never commit venv/ directory:**
   - Already in `.gitignore`
   - Others should create their own venv

5. **Update requirements.txt when adding packages:**
   ```bash
   pip freeze > requirements.txt
   ```

## Troubleshooting

### "ImportError: No module named 'tensorflow'"
```bash
# Check if venv is activated
echo $VIRTUAL_ENV

# If not activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "Command not found: python"
```bash
# Use python3 explicitly
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### GPU Not Detected
```bash
# Check GPU availability
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## For Google Colab

If you prefer using Google Colab for GPU training, see the detailed setup guide in `../colab.md`. 