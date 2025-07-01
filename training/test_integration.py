"""
Integration test between the Game2048Engine and Game2048CNN model.
This tests the complete pipeline from game state to CNN prediction.
"""

import numpy as np
import tensorflow as tf
from game_engine import Game2048Engine, encode_board_for_cnn
from model import Game2048CNN, create_model, compile_model


def test_game_to_cnn_pipeline():
    """Test the complete pipeline from game state to CNN prediction"""
    print("=== Testing Game to CNN Pipeline ===")
    
    # Create a game and model
    game = Game2048Engine()
    model = create_model()
    model = compile_model(model)
    
    # Get game state
    board = game.get_board()
    print(f"Game board:\n{board}")
    
    # Encode board for CNN
    encoded_board = encode_board_for_cnn(board)
    print(f"Encoded board shape: {encoded_board.shape}")
    
    # Add batch dimension
    batch_input = np.expand_dims(encoded_board, axis=0)
    print(f"Batch input shape: {batch_input.shape}")
    
    # Get CNN predictions
    value_pred, policy_pred = model(batch_input)
    
    print(f"Value prediction: {value_pred.numpy()[0][0]:.4f}")
    print(f"Policy prediction: {policy_pred.numpy()[0]}")
    print(f"Policy sum: {np.sum(policy_pred.numpy()[0]):.4f}")
    
    # Verify predictions
    assert value_pred.shape == (1, 1), f"Value shape should be (1, 1), got {value_pred.shape}"
    assert policy_pred.shape == (1, 4), f"Policy shape should be (1, 4), got {policy_pred.shape}"
    assert abs(np.sum(policy_pred.numpy()[0]) - 1.0) < 1e-6, "Policy should sum to 1"
    
    print("✅ Game to CNN pipeline test passed!\n")


def test_action_selection():
    """Test action selection based on CNN policy output"""
    print("=== Testing Action Selection ===")
    
    game = Game2048Engine()
    model = create_model()
    model = compile_model(model)
    
    # Get valid moves from game engine
    valid_moves = game.get_valid_moves()
    print(f"Valid moves from game engine: {valid_moves}")
    
    # Get CNN policy prediction
    board = game.get_board()
    encoded_board = encode_board_for_cnn(board)
    batch_input = np.expand_dims(encoded_board, axis=0)
    
    value_pred, policy_pred = model(batch_input)
    policy_probs = policy_pred.numpy()[0]
    
    print(f"CNN policy probabilities: {policy_probs}")
    
    # Method 1: Choose action with highest probability
    best_action = np.argmax(policy_probs)
    print(f"Action with highest probability: {best_action}")
    
    # Method 2: Choose best valid action
    valid_policy_probs = np.zeros(4)
    for move in valid_moves:
        valid_policy_probs[move] = policy_probs[move]
    
    if np.sum(valid_policy_probs) > 0:
        # Normalize valid probabilities
        valid_policy_probs = valid_policy_probs / np.sum(valid_policy_probs)
        best_valid_action = np.argmax(valid_policy_probs)
        print(f"Best valid action: {best_valid_action}")
        
        # Method 3: Sample from valid actions
        sampled_action = np.random.choice(valid_moves, p=valid_policy_probs[valid_moves] / np.sum(valid_policy_probs[valid_moves]))
        print(f"Sampled valid action: {sampled_action}")
    
    print("✅ Action selection test passed!\n")


def test_batch_processing():
    """Test processing multiple game states in a batch"""
    print("=== Testing Batch Processing ===")
    
    model = create_model()
    model = compile_model(model)
    
    # Create multiple game states
    batch_size = 4
    games = [Game2048Engine() for _ in range(batch_size)]
    
    # Encode all boards
    encoded_boards = []
    for game in games:
        board = game.get_board()
        encoded_board = encode_board_for_cnn(board)
        encoded_boards.append(encoded_board)
    
    # Stack into batch
    batch_input = np.stack(encoded_boards, axis=0)
    print(f"Batch input shape: {batch_input.shape}")
    
    # Get batch predictions
    value_preds, policy_preds = model(batch_input)
    
    print(f"Batch value predictions shape: {value_preds.shape}")
    print(f"Batch policy predictions shape: {policy_preds.shape}")
    
    # Print individual predictions
    for i in range(batch_size):
        print(f"Game {i+1}: Value={value_preds[i][0]:.4f}, Policy={policy_preds[i].numpy()}")
    
    # Verify shapes
    assert value_preds.shape == (batch_size, 1), f"Value batch shape should be ({batch_size}, 1)"
    assert policy_preds.shape == (batch_size, 4), f"Policy batch shape should be ({batch_size}, 4)"
    
    # Verify all policies sum to 1
    policy_sums = np.sum(policy_preds.numpy(), axis=1)
    assert np.allclose(policy_sums, 1.0), "All policies should sum to 1"
    
    print("✅ Batch processing test passed!\n")


def test_different_board_states():
    """Test CNN predictions on different board states"""
    print("=== Testing Different Board States ===")
    
    model = create_model()
    model = compile_model(model)
    
    # Test case 1: Empty board with 2 tiles
    game1 = Game2048Engine()
    game1.board = np.array([
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4]
    ], dtype=np.int32)
    
    # Test case 2: Board with larger tiles
    game2 = Game2048Engine()
    game2.board = np.array([
        [32, 64, 128, 256],
        [16, 32, 64, 128],
        [8, 16, 32, 64],
        [4, 8, 16, 32]
    ], dtype=np.int32)
    
    # Test case 3: Near game-over board
    game3 = Game2048Engine()
    game3.board = np.array([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 64, 128]
    ], dtype=np.int32)
    
    games = [game1, game2, game3]
    descriptions = ["Sparse board", "High-value board", "Dense board"]
    
    for i, (game, desc) in enumerate(zip(games, descriptions)):
        print(f"\n{desc}:")
        print(f"Board:\n{game.board}")
        print(f"Max tile: {game.get_max_tile()}")
        print(f"Valid moves: {game.get_valid_moves()}")
        
        # Get CNN prediction
        encoded_board = encode_board_for_cnn(game.board)
        batch_input = np.expand_dims(encoded_board, axis=0)
        value_pred, policy_pred = model(batch_input)
        
        print(f"Value prediction: {value_pred.numpy()[0][0]:.4f}")
        print(f"Policy prediction: {policy_pred.numpy()[0]}")
    
    print("\n✅ Different board states test passed!\n")


def test_model_training_compatibility():
    """Test that the model is compatible with training procedures"""
    print("=== Testing Model Training Compatibility ===")
    
    model = create_model()
    model = compile_model(model)
    
    # Create dummy training data
    batch_size = 8
    dummy_inputs = tf.random.normal((batch_size, 4, 4, 16))
    
    # Create dummy targets
    dummy_values = tf.random.normal((batch_size, 1))
    dummy_policies = tf.random.uniform((batch_size, 4))
    dummy_policies = dummy_policies / tf.reduce_sum(dummy_policies, axis=1, keepdims=True)  # Normalize
    
    print(f"Dummy input shape: {dummy_inputs.shape}")
    print(f"Dummy value targets shape: {dummy_values.shape}")
    print(f"Dummy policy targets shape: {dummy_policies.shape}")
    
    # Test forward pass
    predictions = model(dummy_inputs)
    value_preds, policy_preds = predictions
    
    print(f"Value predictions shape: {value_preds.shape}")
    print(f"Policy predictions shape: {policy_preds.shape}")
    
    # Test that we can compute gradients
    with tf.GradientTape() as tape:
        predictions = model(dummy_inputs, training=True)
        value_preds, policy_preds = predictions
        
        # Compute losses inside the gradient tape
        value_loss = tf.keras.losses.mse(dummy_values, value_preds)
        policy_loss = tf.keras.losses.categorical_crossentropy(dummy_policies, policy_preds)
        total_loss = tf.reduce_mean(value_loss) + tf.reduce_mean(policy_loss)
    
    print(f"Value loss: {tf.reduce_mean(value_loss):.4f}")
    print(f"Policy loss: {tf.reduce_mean(policy_loss):.4f}")
    print(f"Total loss: {total_loss:.4f}")
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    print(f"Number of gradient tensors: {len(gradients)}")
    
    # Verify gradients exist and are not None
    non_none_grads = sum(1 for grad in gradients if grad is not None)
    print(f"Non-None gradients: {non_none_grads}/{len(gradients)}")
    assert non_none_grads > 0, "At least some gradients should be non-None"
    
    print("✅ Model training compatibility test passed!\n")


if __name__ == "__main__":
    print("Running Integration Tests for Game2048 CNN")
    print("=" * 60)
    
    test_game_to_cnn_pipeline()
    test_action_selection()
    test_batch_processing()
    test_different_board_states()
    test_model_training_compatibility()
    
    print("🎉 All integration tests passed!")
    print("The Game2048Engine and Game2048CNN are fully compatible and ready for training.")