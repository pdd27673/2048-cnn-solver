'use client';

import { motion } from 'framer-motion';

interface TileProps {
  value: number;
  position: {row: number; col: number};
}

export default function Tile({ value, position }: TileProps) {
  const getTileStyle = (value: number) => {
    if (value === 0) return 'bg-gray-600 text-gray-600';
    if (value <= 4) return 'bg-gray-300 text-gray-800';
    if (value <= 16) return 'bg-orange-300 text-gray-800';
    if (value <= 64) return 'bg-orange-400 text-white';
    if (value <= 256) return 'bg-orange-500 text-white';
    if (value <= 1024) return 'bg-orange-600 text-white';
    return 'bg-yellow-400 text-white';
  };

  const getFontSize = (value: number) => {
    if (value >= 1000) return 'text-sm';
    if (value >= 100) return 'text-base';
    return 'text-lg';
  };

  return (
    <motion.div
      className={`
        w-16 h-16 rounded-lg flex items-center justify-center font-bold
        ${getTileStyle(value)} ${getFontSize(value)}
      `}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ duration: 0.2 }}
      key={`${position.row}-${position.col}-${value}`}
    >
      {value > 0 ? value : ''}
    </motion.div>
  );
} 