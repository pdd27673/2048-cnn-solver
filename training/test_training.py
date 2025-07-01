#!/usr/bin/env python3
"""
Test script to verify CNN training functionality works correctly
This tests the core training loop without running full episodes
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test TensorFlow setup
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} loaded")
    
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision policy: {policy.name}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}")
    else:
        print("ℹ️  No GPU - using CPU")
        
except Exception as e:
    print(f"❌ TensorFlow setup failed: {e}")
    exit(1)

# Test model imports
try:
    from model import create_model, compile_model
    from game_engine import encode_board_for_cnn
    print("✅ Model imports successful")
except Exception as e:
    print(f"❌ Model import failed: {e}")
    exit(1)

def test_model_creation():
    """Test that we can create and compile the model"""
    print("\n🧪 Testing model creation...")
    try:
        model = create_model()
        model = compile_model(model, learning_rate=0.001)
        print("✅ Model created and compiled successfully")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_board_encoding():
    """Test board encoding functionality"""
    print("\n🧪 Testing board encoding...")
    try:
        # Create a sample board (as numpy array)
        board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 0],
            [0, 0, 0, 0]
        ], dtype=np.int32)
        
        encoded = encode_board_for_cnn(board)
        expected_shape = (4, 4, 16)
        
        if encoded.shape == expected_shape:
            print(f"✅ Board encoding successful: {encoded.shape}")
            # Test that encoding actually works
            assert encoded[0, 0, 0] == 1.0  # 2^1 = 2, so channel 0 should be 1
            assert encoded[0, 1, 1] == 1.0  # 2^2 = 4, so channel 1 should be 1
            assert encoded[3, 3, :].sum() == 0  # Empty cell should be all zeros
            print("✅ Encoding values correct")
            return True
        else:
            print(f"❌ Wrong encoding shape: got {encoded.shape}, expected {expected_shape}")
            return False
    except Exception as e:
        print(f"❌ Board encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🧪 Testing training step...")
    try:
        # Create model
        model = create_model()
        model = compile_model(model, learning_rate=0.001)
        
        # Create target model
        target_model = create_model()
        target_model.set_weights(model.get_weights())
        
        # Generate sample data
        batch_size = 8
        states = np.random.rand(batch_size, 4, 4, 16).astype(np.float32)
        actions = np.random.randint(0, 4, size=batch_size).astype(np.int64)
        rewards = np.random.rand(batch_size).astype(np.float32)
        next_states = np.random.rand(batch_size, 4, 4, 16).astype(np.float32)
        dones = np.random.choice([True, False], size=batch_size)
        
        # Define training step (simplified version of the main code)
        @tf.function
        def test_train_step(states, actions, rewards, next_states, dones):
            with tf.GradientTape() as tape:
                # Get predictions
                current_values, current_policies = model(states, training=True)
                next_values, _ = target_model(next_states, training=False)
                
                # Ensure dtype consistency
                rewards = tf.cast(rewards, current_values.dtype)
                gamma = tf.cast(0.99, current_values.dtype)
                next_values_squeezed = tf.cast(tf.squeeze(next_values), current_values.dtype)
                
                # Calculate target values
                target_values = tf.where(
                    dones,
                    rewards,
                    rewards + gamma * next_values_squeezed
                )
                target_values = tf.expand_dims(target_values, -1)
                target_values = tf.cast(target_values, current_values.dtype)
                
                # Calculate losses
                value_loss = tf.reduce_mean(tf.keras.losses.mse(target_values, current_values))
                policy_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    actions, current_policies, from_logits=False
                ))
                
                total_loss = value_loss + policy_loss
            
            # Calculate and apply gradients
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return total_loss, value_loss, policy_loss
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states)
        actions_tensor = tf.convert_to_tensor(actions)
        rewards_tensor = tf.convert_to_tensor(rewards)
        next_states_tensor = tf.convert_to_tensor(next_states)
        dones_tensor = tf.convert_to_tensor(dones)
        
        # Run training step
        total_loss, value_loss, policy_loss = test_train_step(
            states_tensor, actions_tensor, rewards_tensor, 
            next_states_tensor, dones_tensor
        )
        
        print(f"✅ Training step successful!")
        print(f"   Total loss: {float(total_loss.numpy()):.4f}")
        print(f"   Value loss: {float(value_loss.numpy()):.4f}")
        print(f"   Policy loss: {float(policy_loss.numpy()):.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_prediction():
    """Test model prediction"""
    print("\n🧪 Testing model prediction...")
    try:
        model = create_model()
        model = compile_model(model, learning_rate=0.001)
        
        # Create sample board state
        state = np.random.rand(1, 4, 4, 16).astype(np.float32)
        
        # Get prediction
        value, policy = model(state, training=False)
        
        print(f"✅ Model prediction successful!")
        print(f"   Value shape: {value.shape}")
        print(f"   Policy shape: {policy.shape}")
        print(f"   Value: {float(value[0]):.4f}")
        print(f"   Policy: {[f'{float(p):.3f}' for p in policy[0]]}")
        return True
        
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting CNN Training Tests\n")
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Board Encoding", test_board_encoding),
        ("Model Prediction", test_model_prediction),
        ("Training Step", test_training_step),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)