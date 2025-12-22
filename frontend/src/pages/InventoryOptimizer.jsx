import { useState } from 'react';
import { Package, TrendingUp, Truck, DollarSign } from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import Modal from '../components/Modal';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const eoqData = [
  { quantity: 100, cost: 3200 },
  { quantity: 200, cost: 2400 },
  { quantity: 300, cost: 2100 },
  { quantity: 400, cost: 2000 },
  { quantity: 500, cost: 1950 },
  { quantity: 600, cost: 2000 },
  { quantity: 700, cost: 2100 },
  { quantity: 800, cost: 2300 },
];

const allocationData = [
  { location: 'NY-001', current: 850, optimal: 1200, capacity: 2000 },
  { location: 'LA-003', current: 1200, optimal: 980, capacity: 1800 },
  { location: 'CHI-002', current: 650, optimal: 890, capacity: 1500 },
  { location: 'SF-001', current: 920, optimal: 1100, capacity: 1600 },
  { location: 'SEA-004', current: 780, optimal: 830, capacity: 1400 },
];

const inventoryTrendData = [
  { date: 'W1', level: 4200, reorderPoint: 3000, safetyStock: 2000 },
  { date: 'W2', level: 3800, reorderPoint: 3000, safetyStock: 2000 },
  { date: 'W3', level: 3400, reorderPoint: 3000, safetyStock: 2000 },
  { date: 'W4', level: 2900, reorderPoint: 3000, safetyStock: 2000 },
  { date: 'W5', level: 5200, reorderPoint: 3000, safetyStock: 2000 },
  { date: 'W6', level: 4800, reorderPoint: 3000, safetyStock: 2000 },
  { date: 'W7', level: 4400, reorderPoint: 3000, safetyStock: 2000 },
];

export default function InventoryOptimizer() {
  const [showScenario, setShowScenario] = useState(false);
  const [leadTime, setLeadTime] = useState(7);
  const [serviceLevel, setServiceLevel] = useState(95);

  const calculateMetrics = () => {
    const safetyStock = Math.round((serviceLevel / 100) * 300 * Math.sqrt(leadTime));
    const reorderPoint = Math.round(150 * leadTime + safetyStock);
    const eoq = 450;
    
    return { safetyStock, reorderPoint, eoq };
  };

  const metrics = calculateMetrics();

  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-gray-900 text-3xl mb-2">Inventory Optimizer</h1>
          <p className="text-gray-600">AI-driven reorder and allocation recommendations</p>
        </div>
        <button
          onClick={() => setShowScenario(true)}
          className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors"
        >
          Simulate Scenario
        </button>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
        <KPICard
          title="Total Inventory Value"
          value="$8.4M"
          change="$200K optimized"
          changeType="up"
          icon={DollarSign}
        />
        <KPICard
          title="Days of Inventory"
          value="42"
          unit="days"
          change="3 days lower"
          changeType="up"
          icon={Package}
        />
        <KPICard
          title="Inventory Turns"
          value="8.4"
          change="0.6 higher"
          changeType="up"
          icon={TrendingUp}
        />
        <KPICard
          title="Carrying Cost"
          value="$156K"
          change="$18K saved"
          changeType="up"
          icon={DollarSign}
        />
      </div>

      {/* Recommendation Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-2xl p-6 shadow-sm">
          <div className="flex items-start justify-between mb-4">
            <Package className="w-8 h-8" />
            <span className="px-3 py-1 bg-white bg-opacity-20 rounded-lg text-sm">Optimal</span>
          </div>
          <h3 className="mb-2">Reorder Quantity</h3>
          <p className="text-4xl mb-2">450</p>
          <p className="text-sm text-blue-100">units per order</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
          <div className="flex items-start justify-between mb-4">
            <Truck className="w-8 h-8 text-blue-600" />
            <span className="px-3 py-1 bg-blue-50 text-blue-600 rounded-lg text-sm">Recommended</span>
          </div>
          <h3 className="text-gray-900 mb-2">Preferred Supplier</h3>
          <p className="text-4xl text-gray-900 mb-2">Supplier A</p>
          <p className="text-sm text-gray-600">96% reliability • 7 days lead time</p>
        </div>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
          <div className="flex items-start justify-between mb-4">
            <TrendingUp className="w-8 h-8 text-blue-600" />
            <span className="px-3 py-1 bg-green-50 text-green-600 rounded-lg text-sm">Optimized</span>
          </div>
          <h3 className="text-gray-900 mb-2">Safety Stock</h3>
          <p className="text-4xl text-gray-900 mb-2">780</p>
          <p className="text-sm text-gray-600">units across all locations</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Economic Order Quantity (EOQ) Analysis">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={eoqData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="quantity" stroke="#6b7280" label={{ value: 'Order Quantity', position: 'bottom' }} />
              <YAxis stroke="#6b7280" label={{ value: 'Total Cost ($)', angle: -90, position: 'left' }} />
              <Tooltip />
              <Line type="monotone" dataKey="cost" stroke="#3B82F6" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4 p-4 bg-blue-50 rounded-xl">
            <p className="text-blue-900 text-sm">
              💡 Optimal order quantity: 450 units minimizes total holding and ordering costs
            </p>
          </div>
        </ChartCard>

        <ChartCard title="Inventory Level Tracking">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={inventoryTrendData}>
              <defs>
                <linearGradient id="colorLevel" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="level" stroke="#3B82F6" fillOpacity={1} fill="url(#colorLevel)" strokeWidth={2} />
              <Line type="monotone" dataKey="reorderPoint" stroke="#F59E0B" strokeWidth={2} strokeDasharray="5 5" />
              <Line type="monotone" dataKey="safetyStock" stroke="#EF4444" strokeWidth={2} strokeDasharray="5 5" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Allocation Plan */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-gray-900">Multi-Location Allocation Plan</h3>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm">
            Apply Recommendations
          </button>
        </div>
        
        <div className="space-y-4">
          {allocationData.map((loc, i) => (
            <div key={i} className="p-4 bg-gray-50 rounded-xl">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-gray-900">{loc.location}</p>
                  <p className="text-sm text-gray-600">
                    Current: {loc.current} → Optimal: {loc.optimal}
                  </p>
                </div>
                <div className="text-right">
                  <span className={`px-3 py-1 rounded-lg text-sm ${
                    loc.optimal > loc.current
                      ? 'bg-green-100 text-green-700'
                      : loc.optimal < loc.current
                      ? 'bg-red-100 text-red-700'
                      : 'bg-gray-100 text-gray-700'
                  }`}>
                    {loc.optimal > loc.current ? '+' : ''}{loc.optimal - loc.current}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${(loc.current / loc.capacity) * 100}%` }}
                  />
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${(loc.optimal / loc.capacity) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reorder Schedule */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-gray-900 mb-4">Upcoming Reorder Schedule</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-gray-600">SKU</th>
                <th className="text-left py-3 px-4 text-gray-600">Current Stock</th>
                <th className="text-left py-3 px-4 text-gray-600">Reorder Point</th>
                <th className="text-left py-3 px-4 text-gray-600">Quantity</th>
                <th className="text-left py-3 px-4 text-gray-600">Est. Reorder Date</th>
                <th className="text-left py-3 px-4 text-gray-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {[
                { sku: 'SKU-001', current: 120, reorder: 200, qty: 450, date: '2024-05-18', status: 'Urgent' },
                { sku: 'SKU-045', current: 340, reorder: 350, qty: 420, date: '2024-05-20', status: 'Soon' },
                { sku: 'SKU-129', current: 580, reorder: 500, qty: 480, date: '2024-05-25', status: 'Planned' },
                { sku: 'SKU-234', current: 780, reorder: 600, qty: 500, date: '2024-05-30', status: 'Planned' },
              ].map((row, i) => (
                <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 text-gray-900">{row.sku}</td>
                  <td className="py-3 px-4 text-gray-700">{row.current}</td>
                  <td className="py-3 px-4 text-gray-700">{row.reorder}</td>
                  <td className="py-3 px-4 text-gray-700">{row.qty}</td>
                  <td className="py-3 px-4 text-gray-700">{row.date}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded-lg text-sm ${
                      row.status === 'Urgent' ? 'bg-red-100 text-red-700' :
                      row.status === 'Soon' ? 'bg-amber-100 text-amber-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scenario Simulator Modal */}
      <Modal
        isOpen={showScenario}
        onClose={() => setShowScenario(false)}
        title="Inventory Scenario Simulator"
        size="lg"
      >
        <div className="space-y-6">
          <div>
            <label className="block text-gray-700 mb-2">Lead Time (days)</label>
            <input
              type="range"
              min="3"
              max="14"
              value={leadTime}
              onChange={(e) => setLeadTime(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>3 days</span>
              <span className="text-gray-900">{leadTime} days</span>
              <span>14 days</span>
            </div>
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Service Level (%)</label>
            <input
              type="range"
              min="90"
              max="99"
              value={serviceLevel}
              onChange={(e) => setServiceLevel(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>90%</span>
              <span className="text-gray-900">{serviceLevel}%</span>
              <span>99%</span>
            </div>
          </div>

          <div className="p-6 bg-blue-50 rounded-xl space-y-4">
            <h4 className="text-blue-900">Calculated Recommendations</h4>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-blue-600">Safety Stock</p>
                <p className="text-2xl text-blue-900">{metrics.safetyStock}</p>
              </div>
              <div>
                <p className="text-sm text-blue-600">Reorder Point</p>
                <p className="text-2xl text-blue-900">{metrics.reorderPoint}</p>
              </div>
              <div>
                <p className="text-sm text-blue-600">Order Quantity</p>
                <p className="text-2xl text-blue-900">{metrics.eoq}</p>
              </div>
            </div>
          </div>

          <div className="flex gap-3 justify-end">
            <button
              onClick={() => setShowScenario(false)}
              className="px-6 py-2.5 border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button className="px-6 py-2.5 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors">
              Apply Scenario
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
