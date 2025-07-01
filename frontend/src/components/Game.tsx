'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Board from './Board';
import AIControls from './AIControls';
import AIVisualizer from './AIVisualizer';
import StatsPanel from './StatsPanel';
import { GameEngine } from '../lib/game-engine';
import { aiModel, ModelState } from '../lib/ai/cnn-model';

interface GameStats {
  movesPlayed: number;
  maxTile: number;
  aiAccuracy: number;
  totalGames: number;
}

interface AIModelStatus {
  state: ModelState;
  error: string | null;
  lastPrediction: any;
}

export default function Game() {
  const gameEngineRef = useRef<GameEngine | null>(null);
  const [board, setBoard] = useState<number[][]>([]);
  const [score, setScore] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [aiSpeed, setAiSpeed] = useState(500);
  const [stats, setStats] = useState<GameStats>({
    movesPlayed: 0,
    maxTile: 2,
    aiAccuracy: 0,
    totalGames: 0
  });
  
  const [modelStatus, setModelStatus] = useState<AIModelStatus>({
    state: ModelState.UNLOADED,
    error: null,
    lastPrediction: null
  });

  const updateGameState = useCallback(() => {
    if (!gameEngineRef.current) return;
    
    const gameState = gameEngineRef.current.getGameState();
    setBoard(gameState.board);
    setScore(gameState.score);
    setGameOver(gameState.gameOver);
    setStats(prev => ({
      ...prev,
      maxTile: Math.max(prev.maxTile, gameState.maxTile)
    }));
  }, []);

  // Initialize game engine and board
  useEffect(() => {
    gameEngineRef.current = new GameEngine();
    updateGameState();
  }, [updateGameState]);

  // Load AI model on component mount
  useEffect(() => {
    const loadModel = async () => {
      console.log('🤖 Attempting to load AI model...');
      setModelStatus(prev => ({ ...prev, state: ModelState.LOADING }));
      
      try {
        const result = await aiModel.loadModel();
        
        if (result.success) {
          console.log('✅ AI model loaded successfully:', result.modelInfo);
          setModelStatus({
            state: ModelState.LOADED,
            error: null,
            lastPrediction: null
          });
        } else {
          console.log('❌ Failed to load AI model:', result.error);
          setModelStatus({
            state: ModelState.ERROR,
            error: result.error || 'Unknown error',
            lastPrediction: null
          });
        }
      } catch (error) {
        console.error('❌ Error loading AI model:', error);
        setModelStatus({
          state: ModelState.ERROR,
          error: error instanceof Error ? error.message : 'Unknown error',
          lastPrediction: null
        });
      }
    };

    loadModel();
  }, []);

  const resetGame = () => {
    if (!gameEngineRef.current) return;
    
    gameEngineRef.current.reset();
    updateGameState();
    setStats(prev => ({
      ...prev,
      totalGames: prev.totalGames + 1,
      movesPlayed: 0
    }));
  };

  // Test AI prediction with current board
  const testAIPrediction = async () => {
    if (!gameEngineRef.current || modelStatus.state !== ModelState.LOADED) {
      console.log('❌ Cannot test AI: Model not loaded or game not ready');
      return;
    }

    try {
      const gameState = gameEngineRef.current.getGameState();
      const validMoves = gameEngineRef.current.getValidMoves();
      
      console.log('🧠 Testing AI prediction...');
      console.log('Current board:', gameState.board);
      console.log('Valid moves:', validMoves);
      
      const prediction = await aiModel.predictMove(gameState.board, validMoves);
      
      console.log('🎯 AI Prediction:', {
        move: prediction.move,
        moveName: ['Up', 'Right', 'Down', 'Left'][prediction.move],
        confidence: `${(prediction.confidence * 100).toFixed(1)}%`,
        value: prediction.value.toFixed(3),
        policy: prediction.policy.map(p => `${(p * 100).toFixed(1)}%`)
      });

      setModelStatus(prev => ({
        ...prev,
        lastPrediction: prediction
      }));

    } catch (error) {
      console.error('❌ AI prediction error:', error);
      setModelStatus(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Prediction failed'
      }));
    }
  };

  const makeMove = useCallback((direction: number) => {
    if (!gameEngineRef.current || gameOver) return;
    
    const moved = gameEngineRef.current.move(direction);
    if (moved) {
      updateGameState();
      setStats(prev => ({
        ...prev,
        movesPlayed: prev.movesPlayed + 1
      }));
    }
  }, [gameOver, updateGameState]);

  // Handle keyboard input for manual play
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (aiEnabled) return; // Don't handle keys when AI is playing
      
      event.preventDefault();
      
      switch (event.key) {
        case 'ArrowUp':
          makeMove(0);
          break;
        case 'ArrowRight':
          makeMove(1);
          break;
        case 'ArrowDown':
          makeMove(2);
          break;
        case 'ArrowLeft':
          makeMove(3);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [aiEnabled, gameOver, makeMove]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-5xl font-bold text-center mb-8">
          2048 AI - CNN Edition
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Game Board */}
          <div className="lg:col-span-2">
            <Board board={board} score={score} />
            
            <div className="mt-4 flex gap-4 justify-center flex-wrap">
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
              <button
                onClick={testAIPrediction}
                disabled={modelStatus.state !== ModelState.LOADED}
                className={`px-6 py-2 rounded-lg transition ${
                  modelStatus.state === ModelState.LOADED
                    ? 'bg-purple-600 hover:bg-purple-700'
                    : 'bg-gray-600 cursor-not-allowed'
                }`}
              >
                Test AI Prediction
              </button>
            </div>

            {/* AI Model Status */}
            <div className="mt-4 text-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-700">
                <span className="text-sm font-medium">AI Model:</span>
                <span className={`text-sm font-bold ${
                  modelStatus.state === ModelState.LOADED ? 'text-green-400' :
                  modelStatus.state === ModelState.LOADING ? 'text-yellow-400' :
                  modelStatus.state === ModelState.ERROR ? 'text-red-400' :
                  'text-gray-400'
                }`}>
                  {modelStatus.state.toUpperCase()}
                </span>
                {modelStatus.error && (
                  <span className="text-xs text-red-300">({modelStatus.error})</span>
                )}
              </div>
            </div>

            {gameOver && (
              <div className="mt-4 text-center">
                <h2 className="text-3xl font-bold text-red-400">
                  Game Over!
                </h2>
                <p className="text-xl mt-2">Final Score: {score}</p>
                <p className="text-lg">Max Tile: {stats.maxTile}</p>
              </div>
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
            
            <AIVisualizer />
            
            <StatsPanel stats={stats} />
          </div>
        </div>
      </div>
    </div>
  );
} 