"""
CNN Training Script for 2048 AI

This script will train a CNN model to play 2048 using self-play and reinforcement learning.
Currently using placeholder structure due to TensorFlow Python 3.13 compatibility issues.
"""

import numpy as np
import random
from collections import deque
import json
import os
from datetime import datetime

# Note: TensorFlow import will be added when compatibility is resolved
# import tensorflow as tf

class Game2048:
    """
    Python implementation of 2048 game for training
    """
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
    
    def add_random_tile(self):
        """Add a random 2 or 4 tile to an empty position"""
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4
            return True
        return False
    
    def get_valid_moves(self):
        """Get list of valid moves (0=up, 1=right, 2=down, 3=left)"""
        valid_moves = []
        original_board = self.board.copy()
        original_score = self.score
        
        for move in range(4):
            if self.make_move(move):
                valid_moves.append(move)
            self.board = original_board.copy()
            self.score = original_score
        
        return valid_moves
    
    def make_move(self, direction):
        """Make a move in the specified direction"""
        # Placeholder implementation
        # Will be completed in task 2 (Implement Core Game Engine)
        return False
    
    def is_game_over(self):
        """Check if the game is over"""
        return len(self.get_valid_moves()) == 0
    
    def get_max_tile(self):
        """Get the maximum tile value on the board"""
        return np.max(self.board)

def encode_board(board):
    """
    Encode board as 4x4x16 tensor (one-hot encoding)
    Each channel represents a power of 2 (2^1 to 2^16)
    """
    encoded = np.zeros((4, 4, 16))
    for i in range(4):
        for j in range(4):
            if board[i, j] > 0:
                power = int(np.log2(board[i, j])) - 1
                if 0 <= power < 16:
                    encoded[i, j, power] = 1
    return encoded

class CNNTrainer:
    """
    CNN Trainer class for 2048 AI
    Placeholder structure until TensorFlow compatibility is resolved
    """
    def __init__(self, lr=0.001):
        self.learning_rate = lr
        self.memory = deque(maxlen=100000)
        self.epsilon = 0.1  # Exploration rate
        self.model = None  # Will be initialized with TensorFlow
        
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