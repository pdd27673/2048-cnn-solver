'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Board from './Board';
import AIControls from './AIControls';
import AIVisualizer from './AIVisualizer';
import StatsPanel from './StatsPanel';
import { GameEngine } from '../lib/game-engine';

interface GameStats {
  movesPlayed: number;
  maxTile: number;
  aiAccuracy: number;
  totalGames: number;
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