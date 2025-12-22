import { Maximize2 } from 'lucide-react';

export default function ChartCard({ title, action, onExpand, children, className = '' }) {
  return (
    <div className={`bg-white rounded-2xl p-6 shadow-soft border border-gray-100/50 hover:shadow-lg transition-all duration-300 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-gray-900 font-semibold">{title}</h3>
        <div className="flex items-center gap-2">
          {action}
          {onExpand && (
            <button
              onClick={onExpand}
              className="p-2 hover:bg-gray-100 rounded-lg transition-all duration-200 hover:scale-110"
            >
              <Maximize2 className="w-4 h-4 text-gray-600" />
            </button>
          )}
        </div>
      </div>
      {children}
    </div>
  );
}