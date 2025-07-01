// Types and interfaces
export interface GameState {
  board: number[][];
  score: number;
  gameOver: boolean;
  maxTile: number;
}

export interface Position {
  row: number;
  col: number;
}

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
    const emptyCells: Position[] = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (this.board[i][j] === 0) {
          emptyCells.push({ row: i, col: j });
        }
      }
    }

    if (emptyCells.length === 0) return false;

    const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
    this.board[randomCell.row][randomCell.col] = Math.random() < 0.9 ? 2 : 4;
    return true;
  }

  private boardsEqual(board1: number[][], board2: number[][]): boolean {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (board1[i][j] !== board2[i][j]) return false;
      }
    }
    return true;
  }

  move(direction: number): boolean {
    // 0: up, 1: right, 2: down, 3: left
    const previousBoard = this.board.map(row => [...row]);
    const previousScore = this.score;

    switch (direction) {
      case 0:
        this.moveUp();
        break;
      case 1:
        this.moveRight();
        break;
      case 2:
        this.moveDown();
        break;
      case 3:
        this.moveLeft();
        break;
      default:
        return false;
    }

    // Check if board changed
    const changed = !this.boardsEqual(previousBoard, this.board);

    if (changed) {
      this.addRandomTile();
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

  private moveRight(): void {
    for (let row = 0; row < 4; row++) {
      // Move tiles
      const newRow = this.board[row].filter(val => val !== 0);

      // Merge tiles (from right to left)
      for (let i = newRow.length - 1; i > 0; i--) {
        if (newRow[i] === newRow[i - 1]) {
          newRow[i] *= 2;
          this.score += newRow[i];
          newRow.splice(i - 1, 1);
          i--; // Skip next iteration since we removed an element
        }
      }

      // Pad with zeros at the beginning
      while (newRow.length < 4) {
        newRow.unshift(0);
      }

      this.board[row] = newRow;
    }
  }

  private moveUp(): void {
    for (let col = 0; col < 4; col++) {
      // Extract column
      const column = [];
      for (let row = 0; row < 4; row++) {
        column.push(this.board[row][col]);
      }

      // Move tiles
      const newColumn = column.filter(val => val !== 0);

      // Merge tiles
      for (let i = 0; i < newColumn.length - 1; i++) {
        if (newColumn[i] === newColumn[i + 1]) {
          newColumn[i] *= 2;
          this.score += newColumn[i];
          newColumn.splice(i + 1, 1);
        }
      }

      // Pad with zeros
      while (newColumn.length < 4) {
        newColumn.push(0);
      }

      // Put column back
      for (let row = 0; row < 4; row++) {
        this.board[row][col] = newColumn[row];
      }
    }
  }

  private moveDown(): void {
    for (let col = 0; col < 4; col++) {
      // Extract column
      const column = [];
      for (let row = 0; row < 4; row++) {
        column.push(this.board[row][col]);
      }

      // Move tiles
      const newColumn = column.filter(val => val !== 0);

      // Merge tiles (from bottom to top)
      for (let i = newColumn.length - 1; i > 0; i--) {
        if (newColumn[i] === newColumn[i - 1]) {
          newColumn[i] *= 2;
          this.score += newColumn[i];
          newColumn.splice(i - 1, 1);
          i--; // Skip next iteration
        }
      }

      // Pad with zeros at the beginning
      while (newColumn.length < 4) {
        newColumn.unshift(0);
      }

      // Put column back
      for (let row = 0; row < 4; row++) {
        this.board[row][col] = newColumn[row];
      }
    }
  }

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

  getGameState(): GameState {
    return {
      board: this.getBoard(),
      score: this.score,
      gameOver: this.isGameOver(),
      maxTile: this.getMaxTile()
    };
  }

  reset(): void {
    this.board = Array(4).fill(null).map(() => Array(4).fill(0));
    this.score = 0;
    this.moveHistory = [];
    this.addRandomTile();
    this.addRandomTile();
  }
} 