interface AIControlsProps {
  aiEnabled: boolean;
  aiSpeed: number;
  onSpeedChange: (speed: number) => void;
  onToggleAI: () => void;
}

export default function AIControls({ 
  aiEnabled, 
  aiSpeed, 
  onSpeedChange, 
  onToggleAI 
}: AIControlsProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-xl font-semibold mb-4">AI Controls</h3>
      
      <div className="space-y-4">
        {/* AI Toggle */}
        <div className="flex items-center justify-between">
          <span>AI Enabled</span>
          <button
            onClick={onToggleAI}
            className={`px-4 py-2 rounded-lg transition ${
              aiEnabled 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {aiEnabled ? 'Stop' : 'Start'}
          </button>
        </div>

        {/* Speed Control */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <span>Speed</span>
            <span className="text-sm text-gray-400">
              {aiSpeed}ms per move
            </span>
          </div>
          <input
            type="range"
            min="100"
            max="2000"
            step="100"
            value={aiSpeed}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            disabled={!aiEnabled}
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>Fast</span>
            <span>Slow</span>
          </div>
        </div>
      </div>
    </div>
  );
} 