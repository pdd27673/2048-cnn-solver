'use client';

import { motion } from 'framer-motion';

interface AIVisualizerProps {
  prediction?: {
    move: number;
    confidence: number;
    moveScores: {direction: string; score: number}[];
    value: number;
  } | null;
  thinking?: boolean;
}

export default function AIVisualizer({ prediction, thinking = false }: AIVisualizerProps) {
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