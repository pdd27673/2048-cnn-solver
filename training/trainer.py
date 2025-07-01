"""
CNN Trainer for 2048 AI using self-play and reinforcement learning.

This module implements the CNNTrainer class that handles:
- Self-play episode generation with epsilon-greedy exploration
- Experience replay memory management
- Batch training with combined value and policy losses
- Model checkpointing and progress monitoring
"""

import numpy as np
import tensorflow as tf
import random
import json
import os
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from game_engine import Game2048Engine, encode_board_for_cnn
from model import Game2048CNN, create_model, compile_model


class CNNTrainer:
    """
    CNN Trainer class for 2048 AI using reinforcement learning with self-play.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        memory_size: int = 100000,
        batch_size: int = 32,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        gamma: float = 0.95,
        model: Optional[Game2048CNN] = None
    ):
        """
        Initialize the CNN trainer.
        
        Args:
            learning_rate: Learning rate for the optimizer
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            gamma: Discount factor for future rewards
            model: Pre-initialized model (if None, creates new one)
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize or use provided model
        if model is None:
            self.model = create_model()
            self.model = compile_model(self.model, learning_rate=learning_rate)
        else:
            self.model = model
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_games': 0,
            'avg_score': 0.0,
            'avg_max_tile': 0.0,
            'loss_history': [],
            'epsilon_history': []
        }
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store an experience tuple in memory.
        
        Args:
            state: Current game state (4x4x16 encoded)
            action: Action taken (0-3)
            reward: Reward received
            next_state: Next game state (4x4x16 encoded)
            done: Whether the game ended
        """
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample_batch(self) -> Optional[List[Dict]]:
        """
        Sample a batch of experiences from memory.
        
        Returns:
            List of experience dictionaries, or None if not enough experiences
        """
        if len(self.memory) < self.batch_size:
            return None
        
        return random.sample(self.memory, self.batch_size)
    
    def get_action(self, state: np.ndarray, valid_moves: List[int]) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current game state (4x4x16 encoded)
            valid_moves: List of valid moves (0-3)
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            # Random exploration
            return random.choice(valid_moves)
        
        # Exploit: use model to predict best action
        state_batch = np.expand_dims(state, axis=0)
        value_pred, policy_pred = self.model(state_batch, training=False)
        
        # Get policy probabilities for valid moves only
        policy_probs = policy_pred.numpy()[0]
        valid_probs = np.zeros(4)
        for move in valid_moves:
            valid_probs[move] = policy_probs[move]
        
        # Choose action with highest probability among valid moves
        if np.sum(valid_probs) > 0:
            return np.argmax(valid_probs)
        else:
            return random.choice(valid_moves)
    
    def calculate_reward(
        self,
        old_score: int,
        new_score: int,
        old_max_tile: int,
        new_max_tile: int,
        game_over: bool
    ) -> float:
        """
        Calculate reward for a move.
        
        Args:
            old_score: Score before move
            new_score: Score after move
            old_max_tile: Max tile before move
            new_max_tile: Max tile after move
            game_over: Whether game ended
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Score improvement reward
        score_diff = new_score - old_score
        if score_diff > 0:
            reward += np.log2(score_diff + 1) / 100.0
        
        # Max tile improvement reward
        if new_max_tile > old_max_tile:
            reward += np.log2(new_max_tile / old_max_tile) / 10.0
        
        # Game over penalty
        if game_over:
            reward -= 1.0
        
        return reward
    
    def self_play_episode(self) -> Tuple[int, int, int]:
        """
        Play one game episode with epsilon-greedy exploration.
        Stores experiences in memory for later training.
        
        Returns:
            Tuple of (final_score, max_tile, num_moves)
        """
        game = Game2048Engine()
        num_moves = 0
        
        while not game.is_game_over():
            # Get current state
            current_state = encode_board_for_cnn(game.get_board())
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                break
            
            # Get action
            action = self.get_action(current_state, valid_moves)
            
            # Store game state before move
            old_score = game.get_score()
            old_max_tile = game.get_max_tile()
            
            # Make move
            move_successful = game.move(action)
            
            if move_successful:
                num_moves += 1
                
                # Get new state and calculate reward
                new_state = encode_board_for_cnn(game.get_board())
                new_score = game.get_score()
                new_max_tile = game.get_max_tile()
                game_over = game.is_game_over()
                
                reward = self.calculate_reward(
                    old_score, new_score, old_max_tile, new_max_tile, game_over
                )
                
                # Store experience
                self.store_experience(
                    current_state, action, reward, new_state, game_over
                )
        
        return game.get_score(), game.get_max_tile(), num_moves
    
    def train_batch(self) -> Optional[float]:
        """
        Train the model on a batch of experiences from memory.
        
        Returns:
            Combined loss value, or None if not enough experiences
        """
        batch = self.sample_batch()
        if batch is None:
            return None
        
        # Prepare training data
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        # Get current Q-values (value predictions)
        current_values, current_policies = self.model(states, training=False)
        current_values = current_values.numpy().flatten()
        
        # Get next Q-values for non-terminal states
        next_values, _ = self.model(next_states, training=False)
        next_values = next_values.numpy().flatten()
        
        # Calculate target values using temporal difference learning
        target_values = rewards.copy()
        for i in range(len(batch)):
            if not dones[i]:
                target_values[i] += self.gamma * next_values[i]
        
        # Create target policy (one-hot encoded actions)
        target_policies = tf.keras.utils.to_categorical(actions, num_classes=4)
        
        # Train the model
        with tf.GradientTape() as tape:
            # Forward pass
            pred_values, pred_policies = self.model(states, training=True)
            
            # Calculate losses
            value_loss = tf.keras.losses.mse(target_values, pred_values)
            policy_loss = tf.keras.losses.categorical_crossentropy(
                target_policies, pred_policies
            )
            
            # Combined loss
            total_loss = tf.reduce_mean(value_loss) + tf.reduce_mean(policy_loss)
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        return total_loss.numpy()
    
    def update_epsilon(self) -> None:
        """Update epsilon for exploration decay."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def train(
        self,
        num_episodes: int = 1000,
        train_interval: int = 5,
        save_interval: int = 100,
        log_interval: int = 10,
        model_save_path: str = "models/game2048_cnn.h5"
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            num_episodes: Number of self-play episodes
            train_interval: Train every N episodes
            save_interval: Save model every N episodes
            log_interval: Log progress every N episodes
            model_save_path: Path to save trained model
            
        Returns:
            Training statistics dictionary
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Model will be saved to: {model_save_path}")
        
        # Create model directory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        episode_scores = []
        episode_max_tiles = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            # Play episode
            score, max_tile, num_moves = self.self_play_episode()
            
            episode_scores.append(score)
            episode_max_tiles.append(max_tile)
            episode_lengths.append(num_moves)
            
            # Train on batch
            if episode % train_interval == 0 and episode > 0:
                loss = self.train_batch()
                if loss is not None:
                    self.training_stats['loss_history'].append(float(loss))
            
            # Update exploration rate
            self.update_epsilon()
            self.training_stats['epsilon_history'].append(self.epsilon)
            
            # Logging
            if episode % log_interval == 0:
                recent_scores = episode_scores[-log_interval:]
                recent_max_tiles = episode_max_tiles[-log_interval:]
                avg_score = np.mean(recent_scores)
                avg_max_tile = np.mean(recent_max_tiles)
                
                print(f"Episode {episode:4d} | "
                      f"Avg Score: {avg_score:6.1f} | "
                      f"Avg Max Tile: {avg_max_tile:4.1f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Memory: {len(self.memory):5d}")
            
            # Save model
            if episode % save_interval == 0 and episode > 0:
                self.save_model(model_save_path)
                print(f"Model saved at episode {episode}")
        
        # Final save
        self.save_model(model_save_path)
        
        # Update final stats
        self.training_stats.update({
            'episodes': num_episodes,
            'total_games': len(episode_scores),
            'avg_score': float(np.mean(episode_scores[-100:])),  # Last 100 episodes
            'avg_max_tile': float(np.mean(episode_max_tiles[-100:])),
            'final_epsilon': self.epsilon
        })
        
        print(f"\nTraining completed!")
        print(f"Final average score (last 100): {self.training_stats['avg_score']:.1f}")
        print(f"Final average max tile (last 100): {self.training_stats['avg_max_tile']:.1f}")
        
        return self.training_stats
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        self.model.save(filepath)
        
        # Also save training stats
        stats_path = filepath.replace('.h5', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        
        # Try to load training stats
        stats_path = filepath.replace('.h5', '_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)


def main():
    """Main training function."""
    print("CNN Training for 2048 AI")
    print("=" * 50)
    
    # Training configuration
    config = {
        'learning_rate': 0.001,
        'memory_size': 50000,
        'batch_size': 32,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'gamma': 0.95,
        'num_episodes': 500,
        'train_interval': 5,
        'save_interval': 100,
        'log_interval': 10
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create trainer
    trainer = CNNTrainer(
        learning_rate=config['learning_rate'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        epsilon=config['epsilon'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_min=config['epsilon_min'],
        gamma=config['gamma']
    )
    
    # Start training
    stats = trainer.train(
        num_episodes=config['num_episodes'],
        train_interval=config['train_interval'],
        save_interval=config['save_interval'],
        log_interval=config['log_interval']
    )
    
    # Save final statistics
    with open('training_results.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTraining results saved to training_results.json")


if __name__ == "__main__":
    main()