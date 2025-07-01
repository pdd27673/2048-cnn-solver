"""
CNN Training Script for 2048 AI - Optimized for Google Colab T4 GPU

This script trains a CNN model to play 2048 using self-play and reinforcement learning.
Optimized for Google Colab with T4 GPU acceleration.

Colab Setup:
    1. Runtime -> Change runtime type -> T4 GPU
    2. Install requirements: !pip install -r requirements.txt
    3. Run training with: !python train_cnn.py --colab

Local Setup:
    source venv/bin/activate  # On macOS/Linux
    pip install -r requirements.txt
    python train_cnn.py
"""

import numpy as np
import random
from collections import deque
import json
import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any, Union
import time
import matplotlib.pyplot as plt
from pathlib import Path

# TensorFlow imports with GPU optimization
try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflowjs as tfjs
    
    # Configure GPU memory growth (important for Colab)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration enabled: {len(gpus)} GPU(s) detected")
            print(f"GPU devices: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU detected - using CPU")
    
    # Enable mixed precision for T4 (improves performance)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy: {policy.name}")
    
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
    Optimized for T4 GPU training with larger batch sizes and memory efficient operations
    """
    def __init__(self, model=None, lr=0.001, batch_size=128, memory_size=200000, 
                 gamma=0.99, epsilon_start=1.0, epsilon_final=0.05, 
                 epsilon_decay=300000, target_update=5000, colab_mode=False):
        """
        Initialize the CNN trainer with hyperparameters optimized for T4 GPU
        
        Args:
            model: Optional pre-trained model. If None, creates a new one (when TF is available)
            lr: Learning rate for the optimizer (0.001 good for T4)
            batch_size: Batch size for training (128 optimal for T4)
            memory_size: Maximum size of replay memory (200k for T4)
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate (1.0 = 100% random actions)
            epsilon_final: Final exploration rate after decay (0.05 for better convergence)
            epsilon_decay: Number of steps over which to decay epsilon
            target_update: How often to update target network (5k for faster updates)
            colab_mode: Enable Colab-specific optimizations
        """
        # Training hyperparameters
        self.learning_rate = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.colab_mode = colab_mode
        
        # Performance tracking
        self.training_start_time = time.time()
        self.last_save_time = time.time()
        self.batch_times = deque(maxlen=100)
        self.losses = deque(maxlen=1000)
        
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
                
                # Compile with mixed precision optimizer for T4
                if colab_mode:
                    optimizer = tf.keras.optimizers.Adam(lr)
                    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                    self.model.compile(
                        optimizer=optimizer,
                        loss={'value_head': 'mse', 'policy_head': 'categorical_crossentropy'},
                        metrics={'value_head': 'mae', 'policy_head': 'accuracy'}
                    )
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
    
    def self_play_episode(self, max_moves=2000):
        """
        Play one game episode with epsilon-greedy exploration
        Optimized for T4 GPU with batch predictions and move limiting
        
        Args:
            max_moves: Maximum moves per episode to prevent infinite games
            
        Returns:
            tuple: (score, max_tile, episode_length, avg_confidence)
        """
        game = Game2048()
        episode_data = []
        move_count = 0
        confidence_scores = []
        
        while not game.is_game_over() and move_count < max_moves:
            state = encode_board(game.board)
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                break
                
            # Update epsilon for current step
            self.epsilon = self.get_epsilon_for_step(self.steps_done)
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.choice(valid_moves)
                confidence_scores.append(0.25)  # Random confidence
            else:
                # Use model prediction when not exploring
                if TF_AVAILABLE and self.model:
                    # Model returns (value, policy) tuple
                    state_batch = np.expand_dims(state, axis=0)
                    value_pred, policy_pred = self.model(state_batch, training=False)
                    policy_probs = policy_pred[0].numpy()  # Get first batch item
                    
                    # Filter for valid moves only and choose best
                    valid_probs = np.full(4, -np.inf)
                    for move in valid_moves:
                        valid_probs[move] = policy_probs[move]
                    
                    if np.any(np.isfinite(valid_probs)):
                        action = np.argmax(valid_probs)
                        confidence_scores.append(policy_probs[action])
                    else:
                        action = random.choice(valid_moves)
                        confidence_scores.append(0.25)  # Random confidence
                else:
                    action = random.choice(valid_moves)
                    confidence_scores.append(0.25)
            
            # Store previous state and score
            prev_state = state
            old_score = game.score
            
            # Execute the move
            game.make_move(action)
            reward = (game.score - old_score) / 1000.0  # Normalize reward
            
            # Get new state
            next_state = encode_board(game.board)
            done = game.is_game_over()
            
            # Store experience in replay memory
            self.store_experience(prev_state, action, reward, next_state, done)
            
            # Add to episode data
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward
            })
            
            # Increment counters
            self.steps_done += 1
            move_count += 1
            
            # Periodically update target network
            if TF_AVAILABLE and self.target_model and self.steps_done % self.target_update == 0:
                self.target_model.set_weights(self.model.get_weights())
                print(f"Target network updated at step {self.steps_done}, epsilon: {self.epsilon:.4f}")
            
            # Periodically train on a batch (less frequent for T4 efficiency)
            if TF_AVAILABLE and self.model and len(self.memory) >= self.batch_size and self.steps_done % 8 == 0:
                loss = self.train_batch()
                if loss is not None:
                    self.losses.append(loss)
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        return game.score, game.get_max_tile(), len(episode_data), avg_confidence
    
    def train_batch(self, batch_size=None):
        """
        Train on a batch from memory using dual-output model (value + policy)
        Optimized for T4 GPU with efficient tensor operations
        """
        if not TF_AVAILABLE or self.model is None:
            return 0.0
        
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample a batch from memory
        batch_start = time.time()
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        # Use tf.function for faster execution on GPU
        @tf.function
        def train_step(states, actions, rewards, next_states, dones):
            with tf.GradientTape() as tape:
                # Get current predictions (value, policy)
                current_values, current_policies = self.model(states, training=True)
                
                # Get next state values from target model
                next_values, _ = self.target_model(next_states, training=False)
                
                # Calculate target values using temporal difference learning
                target_values = tf.where(
                    dones,
                    rewards,
                    rewards + self.gamma * tf.squeeze(next_values)
                )
                target_values = tf.expand_dims(target_values, -1)
                
                # Calculate losses
                value_loss = tf.keras.losses.mse(target_values, current_values)
                policy_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    actions, current_policies, from_logits=False
                )
                
                # Combined loss with mixed precision scaling
                total_loss = value_loss + policy_loss
                if self.colab_mode:
                    scaled_loss = self.model.optimizer.get_scaled_loss(total_loss)
                else:
                    scaled_loss = total_loss
            
            # Calculate gradients
            if self.colab_mode:
                gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                gradients = self.model.optimizer.get_unscaled_gradients(gradients)
            else:
                gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            
            # Apply gradients
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return total_loss
        
        # Convert to tensors for tf.function
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int64)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.bool)
        
        # Execute training step
        total_loss = train_step(states_tensor, actions_tensor, rewards_tensor, 
                              next_states_tensor, dones_tensor)
        
        # Track timing
        batch_time = time.time() - batch_start
        self.batch_times.append(batch_time)
        
        return float(total_loss.numpy())
    
    def save_model(self, filepath, save_tfjs=True):
        """Save the trained model with Colab optimizations"""
        if TF_AVAILABLE and self.model:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save as keras format (recommended for TF 2.x)
            self.model.save(f"{filepath}.keras")
            
            # Save training metadata
            metadata = {
                'training_time': time.time() - self.training_start_time,
                'steps_done': self.steps_done,
                'epsilon': self.epsilon,
                'batch_size': self.batch_size,
                'memory_size': len(self.memory),
                'avg_batch_time': np.mean(list(self.batch_times)) if self.batch_times else 0,
                'recent_loss': np.mean(list(self.losses)[-100:]) if self.losses else 0,
                'colab_mode': self.colab_mode
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save as TensorFlow.js format (if tfjs available and requested)
            if save_tfjs:
                try:
                    tfjs_dir = f"{filepath}_tfjs"
                    tfjs.converters.save_keras_model(
                        self.model, 
                        tfjs_dir,
                        quantization_bytes=2  # Quantize to reduce size
                    )
                    print(f"Model saved to {filepath}.keras, {tfjs_dir}/, and metadata")
                except Exception as e:
                    print(f"Model saved to {filepath}.keras and metadata (TensorFlow.js conversion failed: {e})")
            else:
                print(f"Model saved to {filepath}.keras and metadata")
        else:
            print("Model not saved: TensorFlow not available or model not initialized")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if TF_AVAILABLE:
            try:
                self.model = keras.models.load_model(filepath)
                # Create target network with same weights
                self.target_model = keras.models.clone_model(self.model)
                self.target_model.set_weights(self.model.get_weights())
                print(f"Model loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print("Model not loaded: TensorFlow not available")
            return False

def create_training_plots(stats, episode, final=False):
    """Create training progress plots for Colab visualization"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'2048 CNN Training Progress - Episode {episode}', fontsize=16)
        
        # Plot 1: Scores over time
        axes[0, 0].plot(stats['episode_scores'])
        axes[0, 0].set_title('Episode Scores')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True)
        
        # Plot 2: Max tiles over time
        axes[0, 1].plot(stats['episode_max_tiles'])
        axes[0, 1].set_title('Max Tiles Achieved')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Max Tile')
        axes[0, 1].grid(True)
        
        # Plot 3: Epsilon decay
        axes[0, 2].plot(stats['epsilon_values'])
        axes[0, 2].set_title('Exploration Rate (Epsilon)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Epsilon')
        axes[0, 2].grid(True)
        
        # Plot 4: Confidence scores
        axes[1, 0].plot(stats['confidence_scores'])
        axes[1, 0].set_title('AI Confidence')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].grid(True)
        
        # Plot 5: Episode length
        axes[1, 1].plot(stats['episode_lengths'])
        axes[1, 1].set_title('Episode Length (Moves)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Moves per Episode')
        axes[1, 1].grid(True)
        
        # Plot 6: Memory usage
        axes[1, 2].plot(stats['memory_usage'])
        axes[1, 2].set_title('Memory Buffer Usage')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Experiences Stored')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if final:
            plt.savefig('training_plots_final.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'training_plots_episode_{episode}.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")

def main(args):
    """Main training loop optimized for T4 GPU"""
    # Initialize trainer with Colab optimizations
    trainer = CNNTrainer(
        lr=args.learning_rate,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon_decay=args.epsilon_decay,
        colab_mode=args.colab
    )
    
    print("Starting 2048 CNN Training - T4 GPU Optimized")
    print(f"Training started at: {datetime.now()}")
    print(f"Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Memory size: {args.memory_size}")
    print(f"  Colab mode: {args.colab}")
    print(f"  Save interval: {args.save_interval}")
    if TF_AVAILABLE:
        print(f"  GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Training configuration
    num_episodes = args.episodes
    save_interval = args.save_interval
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    stats = {
        'episode_scores': [],
        'episode_max_tiles': [],
        'episode_lengths': [],
        'epsilon_values': [],
        'confidence_scores': [],
        'training_time': [],
        'memory_usage': []
    }
    
    # Performance tracking
    episode_times = deque(maxlen=100)
    best_score = 0
    best_max_tile = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if trainer.load_model(args.resume):
            print(f"Resumed training from {args.resume}")
        else:
            print(f"Failed to resume from {args.resume}, starting fresh")
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        score, max_tile, length, confidence = trainer.self_play_episode()
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Record statistics
        stats['episode_scores'].append(float(score))
        stats['episode_max_tiles'].append(int(max_tile))
        stats['episode_lengths'].append(int(length))
        stats['epsilon_values'].append(float(trainer.epsilon))
        stats['confidence_scores'].append(float(confidence))
        stats['training_time'].append(float(episode_time))
        stats['memory_usage'].append(len(trainer.memory))
        
        # Track best performance
        if score > best_score:
            best_score = score
        if max_tile > best_max_tile:
            best_max_tile = max_tile
        
        if episode % 10 == 0:
            avg_score = np.mean(stats['episode_scores'][-10:])
            avg_max_tile = np.mean(stats['episode_max_tiles'][-10:])
            avg_confidence = np.mean(stats['confidence_scores'][-10:])
            avg_episode_time = np.mean(list(episode_times))
            curr_epsilon = trainer.epsilon
            recent_loss = np.mean(list(trainer.losses)[-10:]) if trainer.losses else 0
            
            print(f"Episode {episode}: Score: {avg_score:.1f}, MaxTile: {avg_max_tile:.0f}, "
                  f"Confidence: {avg_confidence:.3f}, Epsilon: {curr_epsilon:.4f}, "
                  f"Loss: {recent_loss:.4f}, Time: {avg_episode_time:.2f}s")
            
            # Memory usage info for Colab
            if args.colab and episode % 50 == 0:
                memory_pct = len(trainer.memory) / trainer.memory_size * 100
                print(f"  Memory usage: {len(trainer.memory)}/{trainer.memory_size} ({memory_pct:.1f}%)")
                print(f"  Best so far - Score: {best_score}, Max Tile: {best_max_tile}")
        
        if episode % save_interval == 0 and episode > 0:
            # Save training statistics
            stats_file = f'training_stats_episode_{episode}.json'
            with open(stats_file, 'w') as f:
                # Convert NumPy types to Python native types for JSON serialization
                fixed_stats = {
                    'episode_scores': [float(x) for x in stats['episode_scores']],
                    'episode_max_tiles': [int(x) for x in stats['episode_max_tiles']],
                    'episode_lengths': [int(x) for x in stats['episode_lengths']],
                    'epsilon_values': [float(x) for x in stats['epsilon_values']],
                    'confidence_scores': [float(x) for x in stats['confidence_scores']],
                    'training_time': [float(x) for x in stats['training_time']],
                    'memory_usage': [int(x) for x in stats['memory_usage']],
                    'best_score': float(best_score),
                    'best_max_tile': int(best_max_tile),
                    'total_training_time': time.time() - trainer.training_start_time
                }
                json.dump(fixed_stats, f, indent=2)
            
            # Save model checkpoint
            if TF_AVAILABLE:
                model_path = f"models/2048_cnn_model_episode_{episode}"
                trainer.save_model(model_path, save_tfjs=args.save_tfjs)
                
                # Create training plots for Colab
                if args.colab and episode >= 100:
                    create_training_plots(stats, episode)
            
            print(f"Checkpoint saved at episode {episode}")
    
    print("Training completed!")
    print(f"Final average score: {np.mean(stats['episode_scores'][-100:]):.1f}")
    print(f"Final average max tile: {np.mean(stats['episode_max_tiles'][-100:]):.1f}")
    print(f"Best score achieved: {best_score}")
    print(f"Best max tile achieved: {best_max_tile}")
    print(f"Total training time: {(time.time() - trainer.training_start_time) / 3600:.2f} hours")
    print(f"Final model saved with {len(trainer.memory)} experiences")
    
    # Save final model
    if TF_AVAILABLE:
        final_model_path = "models/2048_cnn_model_final"
        trainer.save_model(final_model_path, save_tfjs=args.save_tfjs)
        print(f"Final model saved to {final_model_path}")
    
    # Generate final training plots
    if args.colab:
        create_training_plots(stats, num_episodes, final=True)
        print("Training plots saved to 'training_plots.png'")
    
    return trainer, stats

def parse_arguments():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description='Train 2048 CNN - T4 GPU Optimized')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=15000,
                       help='Number of training episodes (default: 15000)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--memory-size', type=int, default=200000,
                       help='Replay memory size (default: 200000)')
    parser.add_argument('--epsilon-decay', type=int, default=300000,
                       help='Steps for epsilon decay (default: 300000)')
    parser.add_argument('--save-interval', type=int, default=500,
                       help='Episodes between model saves (default: 500)')
    
    # Platform options
    parser.add_argument('--colab', action='store_true',
                       help='Enable Colab-specific optimizations')
    parser.add_argument('--save-tfjs', action='store_true', default=True,
                       help='Save TensorFlow.js model (default: True)')
    parser.add_argument('--no-tfjs', action='store_true',
                       help='Skip TensorFlow.js conversion')
    
    # Resume training
    parser.add_argument('--resume', type=str,
                       help='Path to model checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Handle no-tfjs flag
    if args.no_tfjs:
        args.save_tfjs = False
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    # Auto-detect Colab environment
    if 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules:
        args.colab = True
        print("Google Colab environment detected - enabling Colab optimizations")
    
    main(args)