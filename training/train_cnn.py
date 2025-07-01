"""
CNN Training Script for 2048 AI

This script will train a CNN model to play 2048 using self-play and reinforcement learning.

IMPORTANT: Always run this script in a Python virtual environment!
Before running, ensure you have activated the venv:
    source venv/bin/activate  # On macOS/Linux
    pip install -r requirements.txt
"""

import numpy as np
import random
from collections import deque
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any, Union

# Note: TensorFlow import will be added when compatibility is resolved
try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflowjs as tfjs
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Running in limited mode.")

# Import our local modules
from game_engine import Game2048Engine, encode_board_for_cnn
from model import Game2048CNN, create_model, compile_model

class Game2048:
    """
    Python implementation of 2048 game for training
    """
    def __init__(self):
        self.game_engine = Game2048Engine()
    
    def add_random_tile(self):
        """Add a random 2 or 4 tile to an empty position"""
        return self.game_engine._add_random_tile()
    
    def get_valid_moves(self):
        """Get list of valid moves (0=up, 1=right, 2=down, 3=left)"""
        return self.game_engine.get_valid_moves()
    
    def make_move(self, direction):
        """Make a move in the specified direction"""
        return self.game_engine.move(direction)
    
    def is_game_over(self):
        """Check if the game is over"""
        return self.game_engine.is_game_over()
    
    def get_max_tile(self):
        """Get the maximum tile value on the board"""
        return self.game_engine.get_max_tile()
    
    @property
    def board(self):
        return self.game_engine.board
    
    @property
    def score(self):
        return self.game_engine.score

def encode_board(board):
    """
    Encode board as 4x4x16 tensor (one-hot encoding)
    Each channel represents a power of 2 (2^1 to 2^16)
    """
    return encode_board_for_cnn(board)

class CNNTrainer:
    """
    CNN Trainer class for 2048 AI with reinforcement learning
    """
    def __init__(self, model=None, lr=0.001, batch_size=32, memory_size=100000, 
                 gamma=0.99, epsilon_start=1.0, epsilon_final=0.1, 
                 epsilon_decay=200000, target_update=10000):
        """
        Initialize the CNN trainer with hyperparameters
        
        Args:
            model: Optional pre-trained model. If None, creates a new one (when TF is available)
            lr: Learning rate for the optimizer
            batch_size: Batch size for training
            memory_size: Maximum size of replay memory
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate (1.0 = 100% random actions)
            epsilon_final: Final exploration rate after decay
            epsilon_decay: Number of steps over which to decay epsilon
            target_update: How often to update target network (in steps)
        """
        # Training hyperparameters
        self.learning_rate = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        
        # Memory buffer for experience replay
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Initialize model architecture if TensorFlow is available
        self.model = None
        self.target_model = None
        
        if TF_AVAILABLE:
            if model is None:
                self.model = create_model()
                self.model = compile_model(self.model, learning_rate=lr)
                
                # Create target network with same architecture but different weights
                self.target_model = create_model() 
                self.target_model.set_weights(self.model.get_weights())
            else:
                self.model = model
                # Create a copy for the target network
                self.target_model = keras.models.clone_model(model)
                self.target_model.set_weights(model.get_weights())
        else:
            print("Warning: TensorFlow not available. Model will not be created.")
    
    def get_epsilon_for_step(self, step):
        """
        Get epsilon value (exploration rate) for the current step
        
        Args:
            step: Current training step
            
        Returns:
            float: Exploration rate (epsilon)
        """
        # Linear epsilon decay schedule
        if step >= self.epsilon_decay:
            return self.epsilon_final
        else:
            return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                   (1 - step / self.epsilon_decay)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay memory
        
        Args:
            state: Current state (encoded board)
            action: Action taken (0-3)
            reward: Reward received
            next_state: Next state after taking action
            done: Whether the game ended after this action
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_batch(self, batch_size=None):
        """
        Sample a random batch from replay memory
        
        Args:
            batch_size: Size of batch to sample (defaults to self.batch_size)
            
        Returns:
            tuple: Batch of experiences (states, actions, rewards, next_states, dones)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        
        # Separate into component arrays
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return states, actions, rewards, next_states, dones
    
    def self_play_episode(self):
        """
        Play one game episode with epsilon-greedy exploration
        Returns game statistics
        """
        game = Game2048()
        episode_data = []
        
        while not game.is_game_over():
            state = encode_board(game.board)
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                break
                
            # Update epsilon for current step
            self.epsilon = self.get_epsilon_for_step(self.steps_done)
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.choice(valid_moves)
            else:
                # TODO: Use model prediction when TensorFlow is available
                action = random.choice(valid_moves)
            
            old_score = game.score
            game.make_move(action)
            reward = (game.score - old_score) / 1000.0  # Normalize reward
            
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward
            })
        
        return game.score, game.get_max_tile(), len(episode_data)
    
    def train_batch(self, batch_size=32):
        """
        Train on a batch from memory
        Placeholder until TensorFlow is available
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        # TODO: Implement actual training when TensorFlow is available
        return 0.0
    
    def save_model(self, filepath):
        """Save the trained model"""
        # TODO: Implement model saving
        pass
    
    def load_model(self, filepath):
        """Load a trained model"""
        # TODO: Implement model loading
        pass

def main():
    """Main training loop"""
    trainer = CNNTrainer()
    
    print("Starting 2048 CNN Training...")
    print(f"Training started at: {datetime.now()}")
    
    # Training configuration
    num_episodes = 1000
    save_interval = 100
    
    stats = {
        'episode_scores': [],
        'episode_max_tiles': [],
        'episode_lengths': []
    }
    
    for episode in range(num_episodes):
        score, max_tile, length = trainer.self_play_episode()
        
        stats['episode_scores'].append(float(score))
        stats['episode_max_tiles'].append(float(max_tile))
        stats['episode_lengths'].append(int(length))
        
        if episode % 10 == 0:
            avg_score = np.mean(stats['episode_scores'][-10:])
            avg_max_tile = np.mean(stats['episode_max_tiles'][-10:])
            print(f"Episode {episode}: Avg Score: {avg_score:.1f}, Avg Max Tile: {avg_max_tile:.1f}")
        
        if episode % save_interval == 0 and episode > 0:
            # Save training statistics
            with open(f'training_stats_episode_{episode}.json', 'w') as f:
                json.dump(stats, f, indent=2)
    
    print("Training completed!")
    print(f"Final average score: {np.mean(stats['episode_scores'][-100:]):.1f}")
    print(f"Final average max tile: {np.mean(stats['episode_max_tiles'][-100:]):.1f}")

if __name__ == "__main__":
    main() 