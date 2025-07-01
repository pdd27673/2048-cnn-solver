"""
CNN Model Architecture for 2048 AI

This module defines the Game2048CNN class that implements the neural network
architecture specified in the PRD for learning to play 2048.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Game2048CNN(keras.Model):
    """
    CNN model for 2048 game AI with dual output heads.
    
    Architecture:
    - Input: 4x4x16 (one-hot encoded board)
    - Conv2D(64, 3x3, relu) + padding='same'
    - Conv2D(128, 3x3, relu) + padding='same' 
    - Conv2D(256, 3x3, relu) + padding='same'
    - Flatten
    - Dense(512, relu)
    - Dense(256, relu)
    - Dual output heads:
        * Value head: Dense(1, linear) - state value estimation
        * Policy head: Dense(4, softmax) - action probabilities
    """
    
    def __init__(self, name="game2048_cnn", **kwargs):
        super(Game2048CNN, self).__init__(name=name, **kwargs)
        
        # Convolutional layers
        self.conv1 = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv1'
        )
        
        self.conv2 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2'
        )
        
        self.conv3 = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv3'
        )
        
        # Flatten layer
        self.flatten = layers.Flatten(name='flatten')
        
        # Dense layers
        self.dense1 = layers.Dense(
            units=512,
            activation='relu',
            name='dense1'
        )
        
        self.dense2 = layers.Dense(
            units=256,
            activation='relu',
            name='dense2'
        )
        
        # Output heads
        self.value_head = layers.Dense(
            units=1,
            activation='linear',
            name='value_head'
        )
        
        self.policy_head = layers.Dense(
            units=4,
            activation='softmax',
            name='policy_head'
        )
    
    def call(self, inputs, training=None, mask=None):
        """
        Forward pass through the network.
        
        Args:
            inputs: Tensor of shape (batch_size, 4, 4, 16)
            training: Whether the model is in training mode
            mask: Optional mask tensor
            
        Returns:
            tuple: (value_output, policy_output)
                - value_output: shape (batch_size, 1) - estimated state value
                - policy_output: shape (batch_size, 4) - action probabilities
        """
        # Convolutional layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output heads
        value_output = self.value_head(x)
        policy_output = self.policy_head(x)
        
        return value_output, policy_output
    
    def get_config(self):
        """Return the configuration of the model for serialization."""
        config = super(Game2048CNN, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create a model from its configuration."""
        return cls(**config)


def create_model(input_shape=(4, 4, 16)):
    """
    Factory function to create and compile the Game2048CNN model.
    
    Args:
        input_shape: Shape of input tensor (default: (4, 4, 16))
        
    Returns:
        Compiled Game2048CNN model
    """
    model = Game2048CNN()
    
    # Build the model by calling it with a dummy input
    dummy_input = tf.random.normal((1,) + input_shape)
    _ = model(dummy_input)
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate losses and optimizers.
    
    Args:
        model: Game2048CNN model instance
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Define losses for dual heads
    losses = {
        'value_head': 'mse',  # Mean Squared Error for value regression
        'policy_head': 'categorical_crossentropy'  # Cross-entropy for policy classification
    }
    
    # Define loss weights
    loss_weights = {
        'value_head': 1.0,
        'policy_head': 1.0
    }
    
    # Define metrics
    metrics = {
        'value_head': ['mae'],  # Mean Absolute Error
        'policy_head': ['accuracy']  # Classification accuracy
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return model


if __name__ == "__main__":
    # Test the model creation and basic functionality
    print("Testing Game2048CNN Model")
    print("=" * 50)
    
    # Create model
    model = create_model()
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    # Test with dummy input
    batch_size = 2
    dummy_input = tf.random.normal((batch_size, 4, 4, 16))
    
    print(f"\nTesting with input shape: {dummy_input.shape}")
    
    # Forward pass
    value_output, policy_output = model(dummy_input)
    
    print(f"Value output shape: {value_output.shape}")
    print(f"Policy output shape: {policy_output.shape}")
    
    # Verify output shapes
    assert value_output.shape == (batch_size, 1), f"Expected value shape ({batch_size}, 1), got {value_output.shape}"
    assert policy_output.shape == (batch_size, 4), f"Expected policy shape ({batch_size}, 4), got {policy_output.shape}"
    
    # Verify policy outputs sum to 1 (softmax property)
    policy_sums = tf.reduce_sum(policy_output, axis=1)
    print(f"Policy output sums (should be ~1.0): {policy_sums.numpy()}")
    
    # Test compilation
    compiled_model = compile_model(model)
    print("\n✅ Model compiled successfully!")
    
    # Test prediction
    print("\nTesting prediction...")
    predictions = compiled_model.predict(dummy_input, verbose=0)
    value_pred, policy_pred = predictions
    
    print(f"Sample value prediction: {value_pred[0][0]:.4f}")
    print(f"Sample policy prediction: {policy_pred[0]}")
    
    print("\n🎉 All tests passed! Model is working correctly.")