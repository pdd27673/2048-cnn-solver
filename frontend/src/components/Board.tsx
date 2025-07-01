import Tile from './Tile';

interface BoardProps {
  board: number[][];
  score: number;
}

export default function Board({ board, score }: BoardProps) {
  return (
    <div className="bg-gray-700 rounded-lg p-6">
      {/* Score Display */}
      <div className="mb-4 text-center">
        <h2 className="text-2xl font-bold">Score: {score}</h2>
      </div>

      {/* Game Grid */}
      <div className="grid grid-cols-4 gap-2 bg-gray-600 p-2 rounded-lg max-w-lg mx-auto">
        {board.map((row, rowIndex) =>
          row.map((value, colIndex) => (
            <Tile 
              key={`${rowIndex}-${colIndex}`}
              value={value}
              position={{row: rowIndex, col: colIndex}}
            />
          ))
        )}
      </div>
    </div>
  );
} 