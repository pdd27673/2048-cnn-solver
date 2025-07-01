interface GameStats {
  movesPlayed: number;
  maxTile: number;
  aiAccuracy: number;
  totalGames: number;
}

interface StatsPanelProps {
  stats: GameStats;
}

export default function StatsPanel({ stats }: StatsPanelProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-xl font-semibold mb-4">Statistics</h3>
      
      <div className="space-y-3">
        <div className="flex justify-between">
          <span className="text-gray-300">Moves Played</span>
          <span className="font-mono">{stats.movesPlayed}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-300">Max Tile</span>
          <span className="font-mono">{stats.maxTile}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-300">AI Accuracy</span>
          <span className="font-mono">{stats.aiAccuracy.toFixed(1)}%</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-300">Total Games</span>
          <span className="font-mono">{stats.totalGames}</span>
        </div>
        
        {/* Progress Bar for AI Accuracy */}
        <div className="pt-2">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">AI Performance</span>
            <span className="text-gray-400">{stats.aiAccuracy.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(stats.aiAccuracy, 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
} 