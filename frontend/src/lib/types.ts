// Core game types
export interface Board {
  tiles: number[][];
  score: number;
}

export interface Tile {
  value: number;
  position: Position;
  isNew?: boolean;
  isMerged?: boolean;
}

export interface Position {
  row: number;
  col: number;
}

export interface GameState {
  board: number[][];
  score: number;
  gameOver: boolean;
  maxTile: number;
  movesPlayed: number;
}

// AI related types
export interface AIPrediction {
  move: number;
  confidence: number;
  moveScores: MoveScore[];
  value: number;
}

export interface MoveScore {
  direction: string;
  score: number;
}

// Game stats
export interface GameStats {
  movesPlayed: number;
  maxTile: number;
  aiAccuracy: number;
  totalGames: number;
  averageScore: number;
  bestScore: number;
}

// Move directions
export enum Direction {
  UP = 0,
  RIGHT = 1,
  DOWN = 2,
  LEFT = 3
}

export const DIRECTION_NAMES = {
  [Direction.UP]: 'Up',
  [Direction.RIGHT]: 'Right',
  [Direction.DOWN]: 'Down',
  [Direction.LEFT]: 'Left'
} as const;

// Animation types
export interface TileAnimation {
  from: Position;
  to: Position;
  duration: number;
}

// AI configuration
export interface AIConfig {
  enabled: boolean;
  speed: number; // milliseconds between moves
  showAnalysis: boolean;
  autoRestart: boolean;
} 