"""
Python implementation of 2048 game engine for training purposes.
This mirrors the functionality of the TypeScript GameEngine.
"""

import numpy as np
import random
from typing import List, Tuple, Optional


class Game2048Engine:
    """
    Python implementation of the 2048 game engine
    """
    
    def __init__(self):
        """Initialize a new game with a 4x4 board and two random tiles"""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.move_history = []
        self._add_random_tile()
        self._add_random_tile()
    
    def _add_random_tile(self) -> bool:
        """
        Add a random tile (2 with 90% probability, 4 with 10% probability) 
        to an empty position on the board.
        
        Returns:
            bool: True if a tile was added, False if board is full
        """
        # Find empty cells
        empty_cells = [(i, j) for i in range(4) for j in range(4) 
                       if self.board[i, j] == 0]
        
        if not empty_cells:
            return False
        
        # Choose random empty cell
        row, col = random.choice(empty_cells)
        
        # Add 2 (90%) or 4 (10%)
        self.board[row, col] = 2 if random.random() < 0.9 else 4
        return True
    
    def _boards_equal(self, board1: np.ndarray, board2: np.ndarray) -> bool:
        """Check if two boards are equal"""
        return np.array_equal(board1, board2)
    
    def move(self, direction: int) -> bool:
        """
        Make a move in the specified direction.
        
        Args:
            direction (int): 0=up, 1=right, 2=down, 3=left
            
        Returns:
            bool: True if the move changed the board, False otherwise
        """
        if direction not in [0, 1, 2, 3]:
            return False
        
        # Store previous state
        previous_board = self.board.copy()
        previous_score = self.score
        
        # Execute move
        if direction == 0:
            self._move_up()
        elif direction == 1:
            self._move_right()
        elif direction == 2:
            self._move_down()
        elif direction == 3:
            self._move_left()
        
        # Check if board changed
        changed = not self._boards_equal(previous_board, self.board)
        
        if changed:
            # Add random tile and record move
            self._add_random_tile()
            self.move_history.append(direction)
            return True
        else:
            # Revert changes if no move happened
            self.board = previous_board
            self.score = previous_score
            return False
    
    def _move_left(self) -> None:
        """Move tiles left"""
        for row in range(4):
            # Extract non-zero tiles
            tiles = [self.board[row, col] for col in range(4) if self.board[row, col] != 0]
            
            # Merge tiles
            merged = []
            i = 0
            while i < len(tiles):
                if i < len(tiles) - 1 and tiles[i] == tiles[i + 1]:
                    # Merge tiles
                    merged_value = tiles[i] * 2
                    merged.append(merged_value)
                    self.score += merged_value
                    i += 2  # Skip next tile
                else:
                    merged.append(tiles[i])
                    i += 1
            
            # Pad with zeros
            while len(merged) < 4:
                merged.append(0)
            
            # Update board
            self.board[row] = merged
    
    def _move_right(self) -> None:
        """Move tiles right"""
        for row in range(4):
            # Extract non-zero tiles
            tiles = [self.board[row, col] for col in range(4) if self.board[row, col] != 0]
            
            # Merge tiles from right to left
            merged = []
            i = len(tiles) - 1
            while i >= 0:
                if i > 0 and tiles[i] == tiles[i - 1]:
                    # Merge tiles
                    merged_value = tiles[i] * 2
                    merged.insert(0, merged_value)
                    self.score += merged_value
                    i -= 2  # Skip previous tile
                else:
                    merged.insert(0, tiles[i])
                    i -= 1
            
            # Pad with zeros at the beginning
            while len(merged) < 4:
                merged.insert(0, 0)
            
            # Update board
            self.board[row] = merged
    
    def _move_up(self) -> None:
        """Move tiles up"""
        for col in range(4):
            # Extract column
            tiles = [self.board[row, col] for row in range(4) if self.board[row, col] != 0]
            
            # Merge tiles
            merged = []
            i = 0
            while i < len(tiles):
                if i < len(tiles) - 1 and tiles[i] == tiles[i + 1]:
                    # Merge tiles
                    merged_value = tiles[i] * 2
                    merged.append(merged_value)
                    self.score += merged_value
                    i += 2  # Skip next tile
                else:
                    merged.append(tiles[i])
                    i += 1
            
            # Pad with zeros
            while len(merged) < 4:
                merged.append(0)
            
            # Update column
            for row in range(4):
                self.board[row, col] = merged[row]
    
    def _move_down(self) -> None:
        """Move tiles down"""
        for col in range(4):
            # Extract column
            tiles = [self.board[row, col] for row in range(4) if self.board[row, col] != 0]
            
            # Merge tiles from bottom to top
            merged = []
            i = len(tiles) - 1
            while i >= 0:
                if i > 0 and tiles[i] == tiles[i - 1]:
                    # Merge tiles
                    merged_value = tiles[i] * 2
                    merged.insert(0, merged_value)
                    self.score += merged_value
                    i -= 2  # Skip previous tile
                else:
                    merged.insert(0, tiles[i])
                    i -= 1
            
            # Pad with zeros at the beginning
            while len(merged) < 4:
                merged.insert(0, 0)
            
            # Update column
            for row in range(4):
                self.board[row, col] = merged[row]
    
    def get_valid_moves(self) -> List[int]:
        """
        Get list of valid moves that would change the board state.
        
        Returns:
            List[int]: List of valid directions (0=up, 1=right, 2=down, 3=left)
        """
        valid_moves = []
        original_board = self.board.copy()
        original_score = self.score
        
        for direction in range(4):
            if self._test_move(direction):
                valid_moves.append(direction)
        
        # Restore original state
        self.board = original_board
        self.score = original_score
        
        return valid_moves
    
    def _test_move(self, direction: int) -> bool:
        """Test if a move would change the board without modifying state"""
        test_board = self.board.copy()
        test_score = self.score
        
        # Execute move
        if direction == 0:
            self._move_up()
        elif direction == 1:
            self._move_right()
        elif direction == 2:
            self._move_down()
        elif direction == 3:
            self._move_left()
        
        # Check if board changed
        changed = not self._boards_equal(test_board, self.board)
        
        # Restore state
        self.board = test_board
        self.score = test_score
        
        return changed
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over (no valid moves remaining).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return len(self.get_valid_moves()) == 0
    
    def get_max_tile(self) -> int:
        """
        Get the maximum tile value on the board.
        
        Returns:
            int: Maximum tile value
        """
        return int(np.max(self.board))
    
    def get_board(self) -> np.ndarray:
        """
        Get a copy of the current board state.
        
        Returns:
            np.ndarray: 4x4 board array
        """
        return self.board.copy()
    
    def get_score(self) -> int:
        """
        Get the current score.
        
        Returns:
            int: Current score
        """
        return self.score
    
    def get_game_state(self) -> dict:
        """
        Get the complete game state.
        
        Returns:
            dict: Game state with board, score, game_over, and max_tile
        """
        return {
            'board': self.get_board(),
            'score': self.score,
            'game_over': self.is_game_over(),
            'max_tile': self.get_max_tile()
        }
    
    def reset(self) -> None:
        """Reset the game to initial state"""
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.move_history = []
        self._add_random_tile()
        self._add_random_tile()


def encode_board_for_cnn(board: np.ndarray) -> np.ndarray:
    """
    Encode a 4x4 board for CNN input using one-hot encoding.
    Each tile value is represented as a channel in a 4x4x16 tensor.
    
    Args:
        board (np.ndarray): 4x4 board with tile values
        
    Returns:
        np.ndarray: 4x4x16 one-hot encoded tensor
    """
    encoded = np.zeros((4, 4, 16), dtype=np.float32)
    
    for i in range(4):
        for j in range(4):
            if board[i, j] > 0:
                # Calculate power of 2 (2^1=2, 2^2=4, 2^3=8, etc.)
                power = int(np.log2(board[i, j]))
                if 1 <= power <= 16:  # Support up to 2^16 = 65536
                    encoded[i, j, power - 1] = 1.0
    
    return encoded


if __name__ == "__main__":
    # Simple test
    game = Game2048Engine()
    print("Initial board:")
    print(game.get_board())
    print(f"Score: {game.get_score()}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print(f"Game over: {game.is_game_over()}")