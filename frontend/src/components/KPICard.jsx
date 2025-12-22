import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function KPICard({ title, value, unit, change, changeType, icon: Icon, onClick }) {
  const getTrendIcon = () => {
    if (changeType === 'up') return <TrendingUp className="w-4 h-4" />;
    if (changeType === 'down') return <TrendingDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  const getTrendColor = () => {
    if (changeType === 'up') return 'text-green-600 bg-green-50 border border-green-100';
    if (changeType === 'down') return 'text-red-600 bg-red-50 border border-red-100';
    return 'text-gray-600 bg-gray-50 border border-gray-100';
  };

  return (
    <div
      onClick={onClick}
      className={`group bg-white rounded-2xl p-6 shadow-soft border border-gray-100/50 hover-lift ${
        onClick ? 'cursor-pointer' : ''
      } relative overflow-hidden`}
    >
      {/* Gradient accent on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      
      <div className="relative">
        <div className="flex items-start justify-between mb-4">
          <div className="p-3 bg-gradient-to-br from-blue-50 to-blue-100/50 rounded-xl group-hover:shadow-glow transition-all duration-300">
            {Icon && <Icon className="w-5 h-5 text-blue-600" />}
          </div>
          {change && (
            <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg ${getTrendColor()} backdrop-blur-sm`}>
              {getTrendIcon()}
              <span className="text-xs font-medium">{change}</span>
            </div>
          )}
        </div>
        <div className="space-y-1">
          <p className="text-gray-600 text-sm font-medium">{title}</p>
          <div className="flex items-baseline gap-2">
            <span className="text-gray-900 font-semibold text-3xl tracking-tight">{value}</span>
            {unit && <span className="text-gray-500 text-sm font-medium">{unit}</span>}
          </div>
        </div>
      </div>
    </div>
  );
}