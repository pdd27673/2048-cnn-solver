# 2048 AI Game - CNN Implementation PRD

## Project Overview

Build a 2048 game where a Convolutional Neural Network (CNN) learns to play through self-play and reinforcement learning. The AI will run directly in the browser using TensorFlow.js, eliminating backend costs while providing instant, responsive gameplay.

## Architecture Decision: Client-Side AI

### Why Browser-Based AI?

- **Zero backend costs** - No server needed
- **Instant response** - No network latency
- **Privacy** - All computation on user's device
- **Easy deployment** - Just static files on Vercel
- **Impressive for recruiters** - "AI that runs in your browser!"

### Technology Stack

- **Frontend**: Next.js 14 with TypeScript
- **AI Framework**: TensorFlow.js (browser-based)
- **Styling**: Tailwind CSS + Framer Motion (animations)
- **Deployment**: Vercel (free tier)
- **Model Training**: Python + TensorFlow (offline)
- **Model Conversion**: TensorFlow.js converter

## Project Structure

```
2048-ai-cnn/
├── frontend/                    # Next.js app
│   ├── app/
│   │   ├── page.tsx            # Main game page
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── Game.tsx            # Main game controller
│   │   ├── Board.tsx           # Game board display
│   │   ├── Tile.tsx            # Animated tiles
│   │   ├── AIControls.tsx      # AI settings panel
│   │   ├── AIVisualizer.tsx    # CNN visualization
│   │   └── StatsPanel.tsx      # Performance metrics
│   ├── lib/
│   │   ├── game-engine.ts      # Core 2048 logic
│   │   ├── ai/
│   │   │   ├── cnn-model.ts    # CNN inference
│   │   │   ├── model-loader.ts # Load TF.js model
│   │   │   └── preprocessor.ts # Board encoding
│   │   └── types.ts
│   ├── public/
│   │   └── model/              # TensorFlow.js model files
│   │       ├── model.json
│   │       └── weights.bin
│   └── package.json
├── training/                    # Python training scripts
│   ├── train_cnn.py            # Main training script
│   ├── game_engine.py          # Python game implementation
│   ├── generate_data.py        # Self-play data generation
│   ├── model.py                # CNN architecture
│   └── convert_to_tfjs.py      # Model conversion
├── docs/
│   └── training_guide.md
└── README.md
```

## Core Components

### 1. CNN Architecture

```python
# training/model.py
import tensorflow as tf

class Game2048CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Input: 4x4x16 (one-hot encoded board)
        self.conv1 = tf.keras.layers.Conv2D(
            256, kernel_size=2, padding='same', activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            256, kernel_size=2, padding='same', activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            256, kernel_size=2, padding='same', activation='relu'
        )
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        
        # Two heads: value and policy
        self.value_head = tf.keras.layers.Dense(1, activation='tanh')
        self.policy_head = tf.keras.layers.Dense(4, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        features = self.dense2(x)
        
        value = self.value_head(features)
        policy = self.policy_head(features)
        
        return value, policy

def encode_board(board):
    """
    Encode board as 4x4x16 tensor
    Each channel represents a power of 2 (2^1 to 2^16)
    """
    encoded = np.zeros((4, 4, 16))
    for i in range(4):
        for j in range(4):
            if board[i][j] > 0:
                power = int(np.log2(board[i][j])) - 1
                if power < 16:
                    encoded[i, j, power] = 1
    return encoded
```

### 2. Training Strategy

```python
# training/train_cnn.py
import numpy as np
from collections import deque

class CNNTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.memory = deque(maxlen=100000)
        
    def self_play_episode(self, epsilon=0.1):
        """Play one game with epsilon-greedy exploration"""
        game = Game2048()
        episode_data = []
        
        while not game.is_over():
            state = encode_board(game.board)
            
            # Get model prediction
            value, policy = self.model(np.expand_dims(state, 0))
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                valid_moves = game.get_valid_moves()
                action = np.random.choice(valid_moves)
            else:
                # Mask invalid moves
                masked_policy = policy[0].numpy()
                valid_moves = game.get_valid_moves()
                for i in range(4):
                    if i not in valid_moves:
                        masked_policy[i] = 0
                
                # Normalize and sample
                if masked_policy.sum() > 0:
                    masked_policy /= masked_policy.sum()
                    action = np.random.choice(4, p=masked_policy)
                else:
                    action = np.random.choice(valid_moves)
            
            # Make move and record
            old_score = game.score
            game.make_move(action)
            reward = (game.score - old_score) / 1000.0  # Normalize reward
            
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'value': value[0, 0].numpy()
            })
        
        # Calculate returns (discounted rewards)
        returns = []
        G = 0
        for t in reversed(range(len(episode_data))):
            G = episode_data[t]['reward'] + 0.99 * G
            returns.insert(0, G)
        
        # Store in memory
        for t, data in enumerate(episode_data):
            self.memory.append({
                'state': data['state'],
                'action': data['action'],
                'return': returns[t],
                'value': data['value']
            })
        
        return game.score, game.get_max_tile()
    
    def train_batch(self, batch_size=32):
        """Train on a batch from memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = np.array([b['state'] for b in batch])
        actions = np.array([b['action'] for b in batch])
        returns = np.array([b['return'] for b in batch])
        
        with tf.GradientTape() as tape:
            values, policies = self.model(states)
            
            # Value loss (MSE)
            value_loss = tf.reduce_mean(tf.square(returns - values[:, 0]))
            
            # Policy loss (cross-entropy)
            actions_one_hot = tf.one_hot(actions, 4)
            policy_loss = -tf.reduce_mean(
                tf.reduce_sum(actions_one_hot * tf.math.log(policies + 1e-8), axis=1)
            )
            
            # Combined loss
            total_loss = value_loss + policy_loss
        
        # Update weights
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return total_loss.numpy()
```

### 3. Frontend Implementation (Next.js)

#### CNN Model Integration

```typescript
// lib/ai/cnn-model.ts
import * as tf from '@tensorflow/tfjs';

export class CNN2048Model {
    private model: tf.LayersModel | null = null;
    private isLoading: boolean = false;
    
    async loadModel() {
        if (this.model || this.isLoading) return;
        
        this.isLoading = true;
        try {
            // Load model from public directory
            this.model = await tf.loadLayersModel('/model/model.json');
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Failed to load model:', error);
        } finally {
            this.isLoading = false;
        }
    }
    
    encodeBoard(board: number[][]): tf.Tensor {
        // Create 4x4x16 tensor
        const encoded = tf.zeros([1, 4, 4, 16]);
        const encodedArray = encoded.arraySync() as number[][][][];
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (board[i][j] > 0) {
                    const power = Math.log2(board[i][j]) - 1;
                    if (power < 16) {
                        encodedArray[0][i][j][power] = 1;
                    }
                }
            }
        }
        
        return tf.tensor(encodedArray);
    }
    
    async predictMove(board: number[][], validMoves: number[]): Promise<{
        move: number;
        confidence: number;
        moveScores: {direction: string; score: number}[];
        value: number;
    }> {
        if (!this.model) {
            throw new Error('Model not loaded');
        }
        
        // Encode board
        const input = this.encodeBoard(board);
        
        // Get prediction
        const prediction = this.model.predict(input) as tf.Tensor[];
        const [value, policy] = await Promise.all([
            prediction[0].data(),
            prediction[1].data()
        ]);
        
        // Mask invalid moves
        const maskedPolicy = Array.from(policy);
        for (let i = 0; i < 4; i++) {
            if (!validMoves.includes(i)) {
                maskedPolicy[i] = 0;
            }
        }
        
        // Normalize
        const sum = maskedPolicy.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            for (let i = 0; i < 4; i++) {
                maskedPolicy[i] /= sum;
            }
        }
        
        // Get best move
        const bestMove = maskedPolicy.indexOf(Math.max(...maskedPolicy));
        
        // Clean up tensors
        input.dispose();
        prediction.forEach(t => t.dispose());
        
        return {
            move: bestMove,
            confidence: maskedPolicy[bestMove],
            moveScores: [
                {direction: 'Up', score: maskedPolicy[0]},
                {direction: 'Right', score: maskedPolicy[1]},
                {direction: 'Down', score: maskedPolicy[2]},
                {direction: 'Left', score: maskedPolicy[3]}
            ],
            value: value[0]
        };
    }
}
```

#### Main Game Component

```tsx
// components/Game.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GameEngine } from '@/lib/game-engine';
import { CNN2048Model } from '@/lib/ai/cnn-model';
import Board from './Board';
import AIControls from './AIControls';
import AIVisualizer from './AIVisualizer';
import StatsPanel from './StatsPanel';

export default function Game() {
    const [game] = useState(() => new GameEngine());
    const [ai] = useState(() => new CNN2048Model());
    const [board, setBoard] = useState(game.getBoard());
    const [score, setScore] = useState(0);
    const [gameOver, setGameOver] = useState(false);
    
    // AI states
    const [aiEnabled, setAiEnabled] = useState(false);
    const [aiSpeed, setAiSpeed] = useState(500); // ms between moves
    const [aiThinking, setAiThinking] = useState(false);
    const [lastAIPrediction, setLastAIPrediction] = useState(null);
    
    // Stats
    const [stats, setStats] = useState({
        movesPlayed: 0,
        maxTile: 2,
        aiAccuracy: 0,
        totalGames: 0
    });
    
    // Load AI model on mount
    useEffect(() => {
        ai.loadModel();
    }, [ai]);
    
    // AI game loop
    useEffect(() => {
        if (!aiEnabled || gameOver || aiThinking) return;
        
        const timeout = setTimeout(async () => {
            setAiThinking(true);
            
            try {
                const validMoves = game.getValidMoves();
                if (validMoves.length === 0) {
                    setGameOver(true);
                    return;
                }
                
                const prediction = await ai.predictMove(
                    game.getBoard(), 
                    validMoves
                );
                
                setLastAIPrediction(prediction);
                
                // Make the move
                const moved = game.move(prediction.move);
                if (moved) {
                    game.addRandomTile();
                    updateGameState();
                }
            } catch (error) {
                console.error('AI error:', error);
                setAiEnabled(false);
            } finally {
                setAiThinking(false);
            }
        }, aiSpeed);
        
        return () => clearTimeout(timeout);
    }, [aiEnabled, gameOver, aiThinking, board, aiSpeed]);
    
    const updateGameState = () => {
        setBoard([...game.getBoard()]);
        setScore(game.getScore());
        setGameOver(game.isGameOver());
        setStats(prev => ({
            ...prev,
            movesPlayed: prev.movesPlayed + 1,
            maxTile: Math.max(prev.maxTile, game.getMaxTile())
        }));
    };
    
    const handleKeyPress = useCallback((e: KeyboardEvent) => {
        if (aiEnabled || gameOver) return;
        
        const keyMap: Record<string, number> = {
            'ArrowUp': 0,
            'ArrowRight': 1,
            'ArrowDown': 2,
            'ArrowLeft': 3
        };
        
        const move = keyMap[e.key];
        if (move !== undefined) {
            const moved = game.move(move);
            if (moved) {
                game.addRandomTile();
                updateGameState();
            }
        }
    }, [aiEnabled, gameOver]);
    
    useEffect(() => {
        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [handleKeyPress]);
    
    const resetGame = () => {
        game.reset();
        updateGameState();
        setStats(prev => ({
            ...prev,
            totalGames: prev.totalGames + 1,
            movesPlayed: 0
        }));
    };
    
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
            <div className="max-w-7xl mx-auto">
                <motion.h1 
                    className="text-5xl font-bold text-center mb-8"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    2048 AI - CNN Edition
                </motion.h1>
                
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Game Board */}
                    <div className="lg:col-span-2">
                        <Board board={board} score={score} />
                        
                        <div className="mt-4 flex gap-4 justify-center">
                            <button
                                onClick={resetGame}
                                className="px-6 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition"
                            >
                                New Game
                            </button>
                            <button
                                onClick={() => setAiEnabled(!aiEnabled)}
                                className={`px-6 py-2 rounded-lg transition ${
                                    aiEnabled 
                                        ? 'bg-red-600 hover:bg-red-700' 
                                        : 'bg-green-600 hover:bg-green-700'
                                }`}
                            >
                                {aiEnabled ? 'Stop AI' : 'Start AI'}
                            </button>
                        </div>
                        
                        {gameOver && (
                            <motion.div
                                className="mt-4 text-center"
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                            >
                                <h2 className="text-3xl font-bold text-red-400">
                                    Game Over!
                                </h2>
                                <p className="text-xl mt-2">
                                    Final Score: {score}
                                </p>
                                <p className="text-lg">
                                    Max Tile: {stats.maxTile}
                                </p>
                            </motion.div>
                        )}
                    </div>
                    
                    {/* AI Controls and Visualization */}
                    <div className="space-y-6">
                        <AIControls 
                            aiEnabled={aiEnabled}
                            aiSpeed={aiSpeed}
                            onSpeedChange={setAiSpeed}
                            onToggleAI={() => setAiEnabled(!aiEnabled)}
                        />
                        
                        <AIVisualizer 
                            prediction={lastAIPrediction}
                            thinking={aiThinking}
                        />
                        
                        <StatsPanel stats={stats} />
                    </div>
                </div>
            </div>
        </div>
    );
}
```

#### AI Visualizer Component

```tsx
// components/AIVisualizer.tsx
import { motion } from 'framer-motion';

interface Props {
    prediction: {
        move: number;
        confidence: number;
        moveScores: {direction: string; score: number}[];
        value: number;
    } | null;
    thinking: boolean;
}

export default function AIVisualizer({ prediction, thinking }: Props) {
    const getBarColor = (score: number) => {
        if (score > 0.7) return 'bg-green-500';
        if (score > 0.4) return 'bg-yellow-500';
        return 'bg-red-500';
    };
    
    return (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                AI Analysis
                {thinking && (
                    <motion.div
                        className="w-3 h-3 bg-blue-500 rounded-full"
                        animate={{ scale: [1, 1.5, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                    />
                )}
            </h3>
            
            {prediction && (
                <div className="space-y-4">
                    {/* Move Probabilities */}
                    <div className="space-y-2">
                        {prediction.moveScores.map((move, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                                <span className="w-16 text-sm">{move.direction}</span>
                                <div className="flex-1 bg-gray-700 rounded-full h-6 relative overflow-hidden">
                                    <motion.div
                                        className={`h-full ${getBarColor(move.score)}`}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${move.score * 100}%` }}
                                        transition={{ duration: 0.3 }}
                                    />
                                    <span className="absolute inset-0 flex items-center justify-center text-xs">
                                        {(move.score * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                    
                    {/* Confidence & Value */}
                    <div className="pt-4 border-t border-gray-700">
                        <div className="flex justify-between mb-2">
                            <span>Confidence</span>
                            <span className="font-mono">
                                {(prediction.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span>Position Value</span>
                            <span className="font-mono">
                                {prediction.value.toFixed(3)}
                            </span>
                        </div>
                    </div>
                </div>
            )}
            
            {!prediction && !thinking && (
                <p className="text-gray-500 text-center">
                    Enable AI to see analysis
                </p>
            )}
        </div>
    );
}
```

### 4. Model Conversion for Browser

```python
# training/convert_to_tfjs.py
import tensorflowjs as tfjs

# After training your model
model = tf.keras.models.load_model('trained_model.h5')

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(
    model,
    '../frontend/public/model'
)

print("Model converted successfully!")
print("Files created:")
print("- model.json (model architecture)")
print("- group1-shard1of1.bin (weights)")
```

### 5. Game Engine Implementation

```typescript
// lib/game-engine.ts
export class GameEngine {
    private board: number[][];
    private score: number;
    private moveHistory: number[];
    
    constructor() {
        this.board = Array(4).fill(null).map(() => Array(4).fill(0));
        this.score = 0;
        this.moveHistory = [];
        this.addRandomTile();
        this.addRandomTile();
    }
    
    private addRandomTile(): boolean {
        const emptyCells = [];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                if (this.board[i][j] === 0) {
                    emptyCells.push([i, j]);
                }
            }
        }
        
        if (emptyCells.length === 0) return false;
        
        const [row, col] = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        this.board[row][col] = Math.random() < 0.9 ? 2 : 4;
        return true;
    }
    
    move(direction: number): boolean {
        // 0: up, 1: right, 2: down, 3: left
        const previousBoard = this.board.map(row => [...row]);
        const previousScore = this.score;
        
        switch (direction) {
            case 0: this.moveUp(); break;
            case 1: this.moveRight(); break;
            case 2: this.moveDown(); break;
            case 3: this.moveLeft(); break;
        }
        
        // Check if board changed
        const changed = !this.boardsEqual(previousBoard, this.board);
        
        if (changed) {
            this.moveHistory.push(direction);
            return true;
        }
        
        this.score = previousScore;
        return false;
    }
    
    private moveLeft(): void {
        for (let row = 0; row < 4; row++) {
            // Move tiles
            const newRow = this.board[row].filter(val => val !== 0);
            
            // Merge tiles
            for (let i = 0; i < newRow.length - 1; i++) {
                if (newRow[i] === newRow[i + 1]) {
                    newRow[i] *= 2;
                    this.score += newRow[i];
                    newRow.splice(i + 1, 1);
                }
            }
            
            // Pad with zeros
            while (newRow.length < 4) {
                newRow.push(0);
            }
            
            this.board[row] = newRow;
        }
    }
    
    // Implement moveRight, moveUp, moveDown similarly...
    
    getValidMoves(): number[] {
        const valid = [];
        for (let i = 0; i < 4; i++) {
            const testBoard = this.board.map(row => [...row]);
            const testScore = this.score;
            
            if (this.move(i)) {
                valid.push(i);
            }
            
            this.board = testBoard;
            this.score = testScore;
        }
        return valid;
    }
    
    isGameOver(): boolean {
        return this.getValidMoves().length === 0;
    }
    
    getMaxTile(): number {
        return Math.max(...this.board.flat());
    }
    
    getBoard(): number[][] {
        return this.board.map(row => [...row]);
    }
    
    getScore(): number {
        return this.score;
    }
    
    reset(): void {
        this.board = Array(4).fill(null).map(() => Array(4).fill(0));
        this.score = 0;
        this.moveHistory = [];
        this.addRandomTile();
        this.addRandomTile();
    }
}
```

## Deployment Strategy

### 1. Frontend Deployment (Vercel - Free)

```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Build and deploy
cd frontend
npm run build
vercel

# 3. Follow prompts - your app will be live in minutes!
```

#### Vercel Configuration

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "regions": ["iad1"], // US East for fast loading
  "functions": {
    "app/api/*.ts": {
      "maxDuration": 10
    }
  }
}
```

### 2. Model Hosting Options

#### Option A: Bundle with Frontend (Recommended)

- Model files served from `/public/model/`
- Cached by browser after first load
- No additional hosting needed

#### Option B: CDN Hosting (For larger models)

```typescript
// Use a CDN like jsdelivr for model files
const MODEL_URL = 'https://cdn.jsdelivr.net/gh/yourusername/2048-ai-models@latest/model.json';
this.model = await tf.loadLayersModel(MODEL_URL);
```

### 3. Training Pipeline

```bash
# Local training script
cd training

# 1. Generate training data
python generate_data.py --games 10000

# 2. Train the model
python train_cnn.py --epochs 100 --batch-size 32

# 3. Convert to TensorFlow.js
python convert_to_tfjs.py

# 4. Copy model files to frontend
cp -r tfjs_model/* ../frontend/public/model/
```

## Performance Optimizations

### 1. Model Optimization

```javascript
// Quantize model for faster inference
const quantizedModel = await tf.loadLayersModel('/model/model.json', {
    weightPathPrefix: '/model/',
    requestInit: {
        cache: 'force-cache'
    }
});

// Use WebGL backend for GPU acceleration
await tf.setBackend('webgl');
```

### 2. Lazy Loading

```typescript
// Load model only when AI is first enabled
const loadModelLazy = async () => {
    if (!modelLoaded) {
        setLoading(true);
        await ai.loadModel();
        setModelLoaded(true);
        setLoading(false);
    }
};
```

### 3. Web Workers (Optional)

```typescript
// Run AI in web worker to prevent UI blocking
const worker = new Worker('/ai-worker.js');
worker.postMessage({ board, validMoves });
worker.onmessage = (e) => {
    const { move, confidence } = e.data;
    // Update UI
};
```

## Development Timeline

### Phase 1: Game Engine (3-4 hours)

- ✅ Implement core 2048 logic
- ✅ Create React components
- ✅ Add animations with Framer Motion
- ✅ Basic UI with Tailwind

### Phase 2: CNN Training (4-6 hours)

- ✅ Implement game in Python
- ✅ Create CNN architecture
- ✅ Generate training data
- ✅ Train model with self-play
- ✅ Convert to TensorFlow.js

### Phase 3: AI Integration (3-4 hours)

- ✅ Load model in browser
- ✅ Implement board encoding
- ✅ Add prediction logic
- ✅ Create AI visualizations

### Phase 4: Polish & Deploy (2-3 hours)

- ✅ Optimize performance
- ✅ Add stats tracking
- ✅ Write documentation
- ✅ Deploy to Vercel

**Total: 12-17 hours**

## Monitoring & Analytics

```typescript
// Track AI performance
const trackGameStats = () => {
    if (typeof window !== 'undefined' && window.gtag) {
        window.gtag('event', 'game_complete', {
            'event_category': 'AI_Performance',
            'final_score': score,
            'max_tile': maxTile,
            'moves_played': moveCount,
            'ai_enabled': aiEnabled
        });
    }
};
```

## Demo Features for Recruiters

- **Live AI Visualization**: See CNN confidence scores in real-time
- **Performance Stats**: Track AI vs human performance
- **Speed Control**: Adjust AI thinking speed
- **Learning Curve**: Show improvement over games
- **Mobile Responsive**: Works on all devices

## Cost Analysis

- **Frontend Hosting**: $0 (Vercel free tier)
- **Model Serving**: $0 (served as static files)
- **Domain (optional)**: $10-15/year
- **Total Monthly Cost**: $0