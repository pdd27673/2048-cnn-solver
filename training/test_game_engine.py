"""
Test script for the Game2048Engine to verify all functionality works correctly.
"""

import numpy as np
from game_engine import Game2048Engine, encode_board_for_cnn


def test_basic_functionality():
    """Test basic game functionality"""
    print("=== Testing Basic Functionality ===")
    
    game = Game2048Engine()
    initial_board = game.get_board()
    
    print(f"Initial board:\n{initial_board}")
    print(f"Initial score: {game.get_score()}")
    print(f"Max tile: {game.get_max_tile()}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print(f"Game over: {game.is_game_over()}")
    
    # Count non-zero tiles
    non_zero_count = np.count_nonzero(initial_board)
    print(f"Non-zero tiles: {non_zero_count}")
    assert non_zero_count == 2, "Should start with exactly 2 tiles"
    
    print("✅ Basic functionality test passed!\n")


def test_moves():
    """Test move functionality"""
    print("=== Testing Move Functionality ===")
    
    # Create a specific board state for testing
    game = Game2048Engine()
    game.board = np.array([
        [2, 2, 0, 0],
        [4, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)
    game.score = 0
    
    print(f"Test board:\n{game.board}")
    
    # Test left move (should merge tiles)
    original_score = game.score
    success = game.move(3)  # Left
    
    print(f"After left move:\n{game.board}")
    print(f"Move successful: {success}")
    print(f"Score change: {game.score - original_score}")
    
    # Should have merged 2+2=4 and 4+4=8, gaining 4+8=12 points
    expected_score_gain = 4 + 8
    actual_score_gain = game.score - original_score
    print(f"Expected score gain: {expected_score_gain}, Actual: {actual_score_gain}")
    
    print("✅ Move functionality test completed!\n")


def test_game_over():
    """Test game over detection"""
    print("=== Testing Game Over Detection ===")
    
    game = Game2048Engine()
    
    # Create a full board with no possible merges
    game.board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2]
    ], dtype=np.int32)
    
    print(f"Full board:\n{game.board}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print(f"Game over: {game.is_game_over()}")
    
    assert game.is_game_over(), "Should detect game over for full board with no merges"
    
    # Test a board with possible merges
    game.board = np.array([
        [2, 2, 4, 8],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 64, 128]
    ], dtype=np.int32)
    
    print(f"Board with merges possible:\n{game.board}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print(f"Game over: {game.is_game_over()}")
    
    assert not game.is_game_over(), "Should not be game over when merges are possible"
    
    print("✅ Game over detection test passed!\n")


def test_board_encoding():
    """Test the CNN board encoding function"""
    print("=== Testing Board Encoding ===")
    
    # Create a test board
    board = np.array([
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)
    
    encoded = encode_board_for_cnn(board)
    
    print(f"Original board:\n{board}")
    print(f"Encoded shape: {encoded.shape}")
    
    # Check specific encodings
    assert encoded[0, 0, 0] == 1.0, "2 should be encoded in channel 0 (2^1)"
    assert encoded[0, 1, 1] == 1.0, "4 should be encoded in channel 1 (2^2)"
    assert encoded[0, 2, 2] == 1.0, "8 should be encoded in channel 2 (2^3)"
    assert encoded[2, 2, 10] == 1.0, "2048 should be encoded in channel 10 (2^11)"
    
    # Check that empty tiles have no encoding
    assert np.sum(encoded[2, 3, :]) == 0.0, "Empty tile should have no encoding"
    
    print("✅ Board encoding test passed!\n")


def test_reset():
    """Test game reset functionality"""
    print("=== Testing Reset Functionality ===")
    
    game = Game2048Engine()
    
    # Make some moves and change the game state
    game.move(0)  # Up
    game.move(1)  # Right
    
    print(f"Board after moves:\n{game.board}")
    print(f"Score after moves: {game.score}")
    
    # Reset the game
    game.reset()
    
    print(f"Board after reset:\n{game.board}")
    print(f"Score after reset: {game.score}")
    
    assert game.score == 0, "Score should be 0 after reset"
    assert np.count_nonzero(game.board) == 2, "Should have exactly 2 tiles after reset"
    assert len(game.move_history) == 0, "Move history should be empty after reset"
    
    print("✅ Reset functionality test passed!\n")


def play_sample_game():
    """Play a short sample game to demonstrate functionality"""
    print("=== Playing Sample Game ===")
    
    game = Game2048Engine()
    move_count = 0
    max_moves = 10
    
    print(f"Starting game:\n{game.board}")
    
    while not game.is_game_over() and move_count < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        
        # Make a random valid move
        import random
        move = random.choice(valid_moves)
        move_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        
        success = game.move(move)
        move_count += 1
        
        print(f"Move {move_count}: {move_names[move]} (success: {success})")
        print(f"Board:\n{game.board}")
        print(f"Score: {game.score}, Max tile: {game.get_max_tile()}")
        print()
    
    print(f"Game ended after {move_count} moves")
    print(f"Final score: {game.score}")
    print(f"Game over: {game.is_game_over()}")
    print("✅ Sample game completed!\n")


if __name__ == "__main__":
    print("Testing Game2048Engine Implementation\n")
    
    test_basic_functionality()
    test_moves()
    test_game_over()
    test_board_encoding()
    test_reset()
    play_sample_game()
    
    print("🎉 All tests passed! Game engine is working correctly.")