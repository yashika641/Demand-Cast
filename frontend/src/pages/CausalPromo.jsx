import { useState } from 'react';
import { Target, TrendingUp, DollarSign } from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import Modal from '../components/Modal';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area } from 'recharts';

const elasticityData = [
  { price: 80, demand: 1800 },
  { price: 85, demand: 1650 },
  { price: 90, demand: 1500 },
  { price: 95, demand: 1380 },
  { price: 100, demand: 1200 },
  { price: 105, demand: 1100 },
  { price: 110, demand: 980 },
  { price: 115, demand: 850 },
  { price: 120, demand: 750 },
];

const promoUpliftData = [
  { promo: 'Black Friday', baseline: 1200, actual: 2880, uplift: 140 },
  { promo: 'Spring Sale', baseline: 950, actual: 1805, uplift: 90 },
  { promo: 'Summer Promo', baseline: 1100, actual: 1980, uplift: 80 },
  { promo: 'Back to School', baseline: 1050, actual: 1785, uplift: 70 },
  { promo: 'Holiday Sale', baseline: 1300, actual: 2470, uplift: 90 },
];

const upliftTimelineData = [
  { day: 'D-3', baseline: 1200, withPromo: 1250 },
  { day: 'D-2', baseline: 1200, withPromo: 1320 },
  { day: 'D-1', baseline: 1200, withPromo: 1480 },
  { day: 'D0', baseline: 1200, withPromo: 2880 },
  { day: 'D+1', baseline: 1200, withPromo: 1980 },
  { day: 'D+2', baseline: 1200, withPromo: 1560 },
  { day: 'D+3', baseline: 1200, withPromo: 1320 },
];

export default function CausalPromo() {
  const [activeTab, setActiveTab] = useState('elasticity');
  const [showSimulator, setShowSimulator] = useState(false);
  const [priceChange, setPriceChange] = useState(0);
  const [promoDiscount, setPromoDiscount] = useState(15);

  const tabs = [
    { id: 'elasticity', label: 'Price Elasticity', icon: DollarSign },
    { id: 'promo', label: 'Promotional Uplift', icon: TrendingUp },
  ];

  const calculateImpact = () => {
    const baseRevenue = 120000;
    const baseDemand = 1200;
    const elasticity = -1.2;
    
    const demandChange = (priceChange / 100) * elasticity;
    const newDemand = baseDemand * (1 + demandChange);
    const newPrice = 100 * (1 + priceChange / 100);
    const newRevenue = newDemand * newPrice;
    const revenueChange = ((newRevenue - baseRevenue) / baseRevenue * 100).toFixed(1);
    
    return {
      demand: Math.round(newDemand),
      revenue: Math.round(newRevenue),
      revenueChange
    };
  };

  const impact = calculateImpact();

  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-gray-900 text-3xl mb-2">Causal & Promotional Insights</h1>
          <p className="text-gray-600">Understand drivers and optimize pricing strategies</p>
        </div>
        <button
          onClick={() => setShowSimulator(true)}
          className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors"
        >
          Run Simulation
        </button>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-2xl p-2 border border-gray-200 inline-flex gap-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2.5 rounded-xl transition-all flex items-center gap-2 ${
                activeTab === tab.id
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Price Elasticity */}
      {activeTab === 'elasticity' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
            <KPICard
              title="Price Elasticity"
              value="-1.2"
              icon={Target}
            />
            <KPICard
              title="Optimal Price Point"
              value="$98"
              change="2% below current"
              changeType="up"
              icon={DollarSign}
            />
            <KPICard
              title="Revenue Maximizing Price"
              value="$105"
              icon={DollarSign}
            />
            <KPICard
              title="Cross-Elasticity"
              value="0.8"
              icon={Target}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Price-Demand Elasticity Curve">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="price" stroke="#6b7280" label={{ value: 'Price ($)', position: 'bottom' }} />
                  <YAxis dataKey="demand" stroke="#6b7280" label={{ value: 'Demand', angle: -90, position: 'left' }} />
                  <Tooltip />
                  <Scatter data={elasticityData} fill="#3B82F6" line={{ stroke: '#3B82F6', strokeWidth: 2 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Revenue Optimization">
              <div className="space-y-6 pt-4">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-gray-700">Current Price</span>
                    <span className="text-gray-900">$100</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div className="bg-blue-500 h-3 rounded-full" style={{ width: '50%' }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-gray-700">Optimal Price</span>
                    <span className="text-gray-900">$98</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div className="bg-green-500 h-3 rounded-full" style={{ width: '49%' }} />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mt-6">
                  <div className="p-4 bg-blue-50 rounded-xl">
                    <p className="text-sm text-blue-600 mb-1">Current Revenue</p>
                    <p className="text-2xl text-blue-900">$120K</p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-xl">
                    <p className="text-sm text-green-600 mb-1">Optimal Revenue</p>
                    <p className="text-2xl text-green-900">$127K</p>
                  </div>
                </div>

                <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
                  <p className="text-amber-900 text-sm">
                    💡 Reducing price by 2% could increase revenue by $7K (5.8%)
                  </p>
                </div>
              </div>
            </ChartCard>
          </div>

          {/* Counterfactual Analysis */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Counterfactual Scenario Analysis</h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div>
                <label className="block text-gray-700 text-sm mb-2">Price Change (%)</label>
                <input
                  type="range"
                  min="-20"
                  max="20"
                  value={priceChange}
                  onChange={(e) => setPriceChange(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-gray-600 mt-1">
                  <span>-20%</span>
                  <span className="text-gray-900">{priceChange}%</span>
                  <span>+20%</span>
                </div>
              </div>

              <div className="space-y-3">
                <div className="p-4 bg-gray-50 rounded-xl">
                  <p className="text-sm text-gray-600 mb-1">Predicted Demand</p>
                  <p className="text-2xl text-gray-900">{impact.demand} units</p>
                </div>
              </div>

              <div className="space-y-3">
                <div className="p-4 bg-gray-50 rounded-xl">
                  <p className="text-sm text-gray-600 mb-1">Revenue Impact</p>
                  <p className="text-2xl text-gray-900">${(impact.revenue / 1000).toFixed(0)}K</p>
                  <p className={`text-sm mt-1 ${
                    Number(impact.revenueChange) > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {impact.revenueChange > 0 ? '+' : ''}{impact.revenueChange}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Promotional Uplift */}
      {activeTab === 'promo' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
            <KPICard
              title="Avg Uplift Rate"
              value="94"
              unit="%"
              icon={TrendingUp}
            />
            <KPICard
              title="Promo ROI"
              value="3.2x"
              change="0.4x better"
              changeType="up"
              icon={DollarSign}
            />
            <KPICard
              title="Incremental Revenue"
              value="$480K"
              icon={DollarSign}
            />
            <KPICard
              title="Active Promotions"
              value="12"
              icon={Target}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Promotional Uplift by Campaign">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={promoUpliftData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="promo" stroke="#6b7280" angle={-20} textAnchor="end" height={80} />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="baseline" fill="#9CA3AF" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="actual" fill="#3B82F6" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Uplift Timeline (Black Friday)">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={upliftTimelineData}>
                  <defs>
                    <linearGradient id="colorPromo" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="baseline" stroke="#9CA3AF" strokeWidth={2} strokeDasharray="5 5" />
                  <Area type="monotone" dataKey="withPromo" stroke="#3B82F6" fillOpacity={1} fill="url(#colorPromo)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Promo Performance Table */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Recent Promotional Performance</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-gray-600">Campaign</th>
                    <th className="text-left py-3 px-4 text-gray-600">Baseline</th>
                    <th className="text-left py-3 px-4 text-gray-600">Actual</th>
                    <th className="text-left py-3 px-4 text-gray-600">Uplift %</th>
                    <th className="text-left py-3 px-4 text-gray-600">Incremental</th>
                    <th className="text-left py-3 px-4 text-gray-600">ROI</th>
                  </tr>
                </thead>
                <tbody>
                  {promoUpliftData.map((row, i) => (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-gray-900">{row.promo}</td>
                      <td className="py-3 px-4 text-gray-700">{row.baseline}</td>
                      <td className="py-3 px-4 text-gray-700">{row.actual}</td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-lg text-sm">
                          +{row.uplift}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-gray-700">{row.actual - row.baseline}</td>
                      <td className="py-3 px-4 text-gray-900">{(2.5 + Math.random()).toFixed(1)}x</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Simulation Modal */}
      <Modal
        isOpen={showSimulator}
        onClose={() => setShowSimulator(false)}
        title="Promotional Impact Simulator"
        size="lg"
      >
        <div className="space-y-6">
          <div>
            <label className="block text-gray-700 mb-2">Promotion Type</label>
            <select className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option>Percentage Discount</option>
              <option>BOGO (Buy One Get One)</option>
              <option>Bundle Offer</option>
              <option>Flash Sale</option>
            </select>
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Discount Amount (%)</label>
            <input
              type="range"
              min="5"
              max="50"
              value={promoDiscount}
              onChange={(e) => setPromoDiscount(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>5%</span>
              <span className="text-gray-900">{promoDiscount}%</span>
              <span>50%</span>
            </div>
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Duration (days)</label>
            <input
              type="number"
              defaultValue="7"
              className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="p-6 bg-blue-50 rounded-xl space-y-3">
            <h4 className="text-blue-900">Predicted Impact</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-blue-600">Expected Uplift</p>
                <p className="text-2xl text-blue-900">+{Math.round(promoDiscount * 2.8)}%</p>
              </div>
              <div>
                <p className="text-sm text-blue-600">Incremental Units</p>
                <p className="text-2xl text-blue-900">{Math.round(1200 * (promoDiscount * 0.028))}</p>
              </div>
              <div>
                <p className="text-sm text-blue-600">Revenue Impact</p>
                <p className="text-2xl text-blue-900">+${Math.round((120 * promoDiscount * 0.028) / 10) * 10}K</p>
              </div>
              <div>
                <p className="text-sm text-blue-600">Estimated ROI</p>
                <p className="text-2xl text-blue-900">{(2.1 + (50 - promoDiscount) * 0.05).toFixed(1)}x</p>
              </div>
            </div>
          </div>

          <div className="flex gap-3 justify-end">
            <button
              onClick={() => setShowSimulator(false)}
              className="px-6 py-2.5 border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button className="px-6 py-2.5 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors">
              Run Simulation
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
