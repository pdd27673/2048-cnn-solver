"""
Test script for the CNNTrainer class to verify functionality.
"""

import numpy as np
import tensorflow as tf
from trainer import CNNTrainer
from game_engine import Game2048Engine, encode_board_for_cnn


def test_trainer_initialization():
    """Test CNNTrainer initialization"""
    print("=== Testing Trainer Initialization ===")
    
    trainer = CNNTrainer(
        learning_rate=0.001,
        memory_size=1000,
        batch_size=8,
        epsilon=0.2
    )
    
    print(f"Model type: {type(trainer.model)}")
    print(f"Memory size: {len(trainer.memory)}")
    print(f"Batch size: {trainer.batch_size}")
    print(f"Epsilon: {trainer.epsilon}")
    
    # Test model is compiled
    assert trainer.model is not None, "Model should be initialized"
    assert len(trainer.memory) == 0, "Memory should start empty"
    assert trainer.epsilon == 0.2, "Epsilon should be set correctly"
    
    print("✅ Trainer initialization test passed!\n")


def test_experience_storage():
    """Test experience storage and sampling"""
    print("=== Testing Experience Storage ===")
    
    trainer = CNNTrainer(memory_size=100, batch_size=4)
    
    # Create dummy experiences
    for i in range(10):
        state = np.random.random((4, 4, 16))
        action = i % 4
        reward = float(i)
        next_state = np.random.random((4, 4, 16))
        done = (i == 9)
        
        trainer.store_experience(state, action, reward, next_state, done)
    
    print(f"Memory size after storing 10 experiences: {len(trainer.memory)}")
    assert len(trainer.memory) == 10, "Should have 10 experiences"
    
    # Test sampling
    batch = trainer.sample_batch()
    print(f"Sampled batch size: {len(batch) if batch else 0}")
    assert batch is not None, "Should be able to sample batch"
    assert len(batch) == 4, "Batch size should be 4"
    
    print("✅ Experience storage test passed!\n")


def test_action_selection():
    """Test action selection with epsilon-greedy policy"""
    print("=== Testing Action Selection ===")
    
    trainer = CNNTrainer(epsilon=0.0)  # No exploration for testing
    
    # Create a test game state
    game = Game2048Engine()
    state = encode_board_for_cnn(game.get_board())
    valid_moves = game.get_valid_moves()
    
    print(f"Valid moves: {valid_moves}")
    print(f"State shape: {state.shape}")
    
    # Test action selection
    action = trainer.get_action(state, valid_moves)
    print(f"Selected action: {action}")
    
    assert action in valid_moves, f"Action {action} should be in valid moves {valid_moves}"
    
    # Test with high epsilon (should be random)
    trainer.epsilon = 1.0
    actions = [trainer.get_action(state, valid_moves) for _ in range(10)]
    print(f"Random actions: {actions}")
    
    # All actions should be valid
    for action in actions:
        assert action in valid_moves, f"Action {action} should be valid"
    
    print("✅ Action selection test passed!\n")


def test_reward_calculation():
    """Test reward calculation logic"""
    print("=== Testing Reward Calculation ===")
    
    trainer = CNNTrainer()
    
    # Test score improvement reward
    reward1 = trainer.calculate_reward(100, 200, 32, 32, False)
    print(f"Score improvement reward (100→200): {reward1:.4f}")
    
    # Test max tile improvement reward
    reward2 = trainer.calculate_reward(100, 100, 32, 64, False)
    print(f"Max tile improvement reward (32→64): {reward2:.4f}")
    
    # Test game over penalty
    reward3 = trainer.calculate_reward(100, 100, 32, 32, True)
    print(f"Game over penalty: {reward3:.4f}")
    
    # Test combined reward
    reward4 = trainer.calculate_reward(100, 300, 32, 64, False)
    print(f"Combined reward (score+tile improvement): {reward4:.4f}")
    
    assert reward1 > 0, "Score improvement should give positive reward"
    assert reward2 > 0, "Tile improvement should give positive reward"
    assert reward3 < 0, "Game over should give negative reward"
    assert reward4 > reward1, "Combined reward should be higher"
    
    print("✅ Reward calculation test passed!\n")


def test_self_play_episode():
    """Test self-play episode generation"""
    print("=== Testing Self-Play Episode ===")
    
    trainer = CNNTrainer(epsilon=0.5)  # Some exploration
    
    print("Playing one episode...")
    score, max_tile, num_moves = trainer.self_play_episode()
    
    print(f"Episode results:")
    print(f"  Final score: {score}")
    print(f"  Max tile: {max_tile}")
    print(f"  Number of moves: {num_moves}")
    print(f"  Experiences stored: {len(trainer.memory)}")
    
    assert score >= 0, "Score should be non-negative"
    assert max_tile >= 2, "Max tile should be at least 2"
    assert num_moves >= 0, "Number of moves should be non-negative"
    assert len(trainer.memory) == num_moves, "Should store one experience per move"
    
    print("✅ Self-play episode test passed!\n")


def test_batch_training():
    """Test batch training functionality"""
    print("=== Testing Batch Training ===")
    
    trainer = CNNTrainer(batch_size=4, memory_size=100)
    
    # Generate some experiences by playing episodes
    print("Generating experiences...")
    for _ in range(3):
        trainer.self_play_episode()
    
    print(f"Total experiences: {len(trainer.memory)}")
    
    if len(trainer.memory) >= trainer.batch_size:
        print("Training on batch...")
        loss = trainer.train_batch()
        print(f"Training loss: {loss:.4f}")
        
        assert loss is not None, "Should return a loss value"
        assert loss >= 0, "Loss should be non-negative"
    else:
        print("Not enough experiences for batch training")
    
    print("✅ Batch training test passed!\n")


def test_epsilon_decay():
    """Test epsilon decay functionality"""
    print("=== Testing Epsilon Decay ===")
    
    trainer = CNNTrainer(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.1)
    
    print(f"Initial epsilon: {trainer.epsilon}")
    
    # Test multiple updates
    epsilons = [trainer.epsilon]
    for i in range(10):
        trainer.update_epsilon()
        epsilons.append(trainer.epsilon)
    
    print(f"Epsilon values: {[f'{e:.3f}' for e in epsilons]}")
    print(f"Final epsilon: {trainer.epsilon}")
    
    assert trainer.epsilon < 1.0, "Epsilon should decrease"
    assert trainer.epsilon >= trainer.epsilon_min, "Epsilon should not go below minimum"
    
    # Test that it stops at minimum
    for _ in range(20):
        trainer.update_epsilon()
    
    assert trainer.epsilon == trainer.epsilon_min, "Epsilon should reach minimum and stop"
    
    print("✅ Epsilon decay test passed!\n")


if __name__ == "__main__":
    print("Testing CNNTrainer Implementation")
    print("=" * 60)
    
    test_trainer_initialization()
    test_experience_storage()
    test_action_selection()
    test_reward_calculation()
    test_self_play_episode()
    test_batch_training()
    test_epsilon_decay()
    
    print("🎉 All trainer tests passed!")
    print("The CNNTrainer is ready for training.")