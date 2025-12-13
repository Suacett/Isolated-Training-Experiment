'use client';
import { useState } from 'react';
import { Info } from 'lucide-react';

interface InfoMarkerProps {
  title: string;
  simple: string;      // "How sure the AI is about this trade."
  technical: string;   // "Derived from the softmax probability output."
  className?: string;
}

export default function InfoMarker({ title, simple, technical, className = "" }: InfoMarkerProps) {
  const [show, setShow] = useState(false);

  return (
    <span
      className={`relative inline-flex items-center cursor-help ${className}`}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <Info
        size={14}
        className="text-zinc-500 hover:text-amber-400 transition-colors ml-1"
      />

      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-[#1A1A24] border border-white/10 rounded-lg shadow-xl z-[999] animate-fade-in pointer-events-none">
          <div className="text-amber-400 font-semibold text-xs mb-1">{title}</div>
          <div className="text-white text-xs mb-2 leading-relaxed">{simple}</div>
          <div className="text-zinc-400 text-[10px] italic border-t border-white/10 pt-2">
            📐 {technical}
          </div>
          {/* Arrow pointing down */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-[1px]">
            <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-[#1A1A24]"></div>
          </div>
        </div>
      )}
    </span>
  );
}
