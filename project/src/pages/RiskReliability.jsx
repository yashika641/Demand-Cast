import { useState } from 'react';
import { AlertTriangle, TrendingUp, Shield, Activity } from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import Drawer from '../components/Drawer';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area } from 'recharts';

const stockoutData = [
  { date: 'W1', probability: 15 },
  { date: 'W2', probability: 22 },
  { date: 'W3', probability: 35 },
  { date: 'W4', probability: 52 },
  { date: 'W5', probability: 68 },
  { date: 'W6', probability: 78 },
  { date: 'W7', probability: 85 },
];

const monteCarloData = Array.from({ length: 100 }, (_, i) => ({
  simulation: i,
  demand: Math.floor(Math.random() * 2000) + 4000 + (i * 10)
}));

const supplierData = [
  { supplier: 'Supplier A', reliability: 96, leadTime: 7, variance: 1.2 },
  { supplier: 'Supplier B', reliability: 92, leadTime: 9, variance: 2.1 },
  { supplier: 'Supplier C', reliability: 88, leadTime: 11, variance: 3.4 },
  { supplier: 'Supplier D', reliability: 94, leadTime: 8, variance: 1.8 },
];

const anomalyData = [
  { date: '2024-01-01', demand: 4200, anomaly: null },
  { date: '2024-01-02', demand: 4350, anomaly: null },
  { date: '2024-01-03', demand: 4100, anomaly: null },
  { date: '2024-01-04', demand: 6800, anomaly: 6800 },
  { date: '2024-01-05', demand: 4250, anomaly: null },
  { date: '2024-01-06', demand: 4400, anomaly: null },
  { date: '2024-01-07', demand: 2100, anomaly: 2100 },
];

export default function RiskReliability() {
  const [activeTab, setActiveTab] = useState('stockout');
  const [showRecommendations, setShowRecommendations] = useState(false);

  const tabs = [
    { id: 'stockout', label: 'Stockout Probability', icon: AlertTriangle },
    { id: 'supplier', label: 'Supplier Reliability', icon: Shield },
    { id: 'anomaly', label: 'Anomaly Detection', icon: Activity },
  ];

  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-gray-900 text-3xl mb-2">Risk & Reliability Analysis</h1>
          <p className="text-gray-600">Proactive risk management and supply chain intelligence</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-2xl p-2 border border-gray-200 inline-flex gap-2 overflow-x-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2.5 rounded-xl transition-all flex items-center gap-2 whitespace-nowrap ${
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

      {/* Stockout Probability */}
      {activeTab === 'stockout' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
            <KPICard
              title="Overall Stockout Risk"
              value="2.1"
              unit="%"
              change="0.5% lower"
              changeType="up"
              icon={AlertTriangle}
            />
            <KPICard
              title="High Risk SKUs"
              value="47"
              change="8 less"
              changeType="up"
              icon={AlertTriangle}
            />
            <KPICard
              title="Critical Alerts"
              value="12"
              icon={AlertTriangle}
            />
            <KPICard
              title="Avg Days to Stockout"
              value="18"
              unit="days"
              icon={TrendingUp}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard 
              title="7-Day Stockout Probability Trend"
              action={
                <button
                  onClick={() => setShowRecommendations(true)}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
                >
                  View Recommendations
                </button>
              }
            >
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={stockoutData}>
                  <defs>
                    <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Area type="monotone" dataKey="probability" stroke="#EF4444" fillOpacity={1} fill="url(#colorProb)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Monte Carlo Demand Simulation">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="simulation" stroke="#6b7280" />
                  <YAxis dataKey="demand" stroke="#6b7280" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter data={monteCarloData} fill="#3B82F6" opacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Risk Table */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">High-Risk SKUs Requiring Attention</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-gray-600">SKU</th>
                    <th className="text-left py-3 px-4 text-gray-600">Current Stock</th>
                    <th className="text-left py-3 px-4 text-gray-600">Safety Stock</th>
                    <th className="text-left py-3 px-4 text-gray-600">Stockout Risk</th>
                    <th className="text-left py-3 px-4 text-gray-600">Days Until Stockout</th>
                    <th className="text-left py-3 px-4 text-gray-600">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { sku: 'SKU-001', current: 120, safety: 200, risk: '92%', days: 3, level: 'critical' },
                    { sku: 'SKU-045', current: 340, safety: 400, risk: '78%', days: 5, level: 'high' },
                    { sku: 'SKU-129', current: 580, safety: 650, risk: '65%', days: 8, level: 'medium' },
                    { sku: 'SKU-234', current: 780, safety: 800, risk: '45%', days: 12, level: 'medium' },
                  ].map((row, i) => (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-gray-900">{row.sku}</td>
                      <td className="py-3 px-4 text-gray-700">{row.current}</td>
                      <td className="py-3 px-4 text-gray-700">{row.safety}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-lg text-sm ${
                          row.level === 'critical' ? 'bg-red-100 text-red-700' :
                          row.level === 'high' ? 'bg-amber-100 text-amber-700' :
                          'bg-yellow-100 text-yellow-700'
                        }`}>
                          {row.risk}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-gray-700">{row.days} days</td>
                      <td className="py-3 px-4">
                        <button className="text-blue-600 hover:text-blue-700 text-sm">
                          Reorder
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Supplier Reliability */}
      {activeTab === 'supplier' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
            <KPICard
              title="Avg Supplier Reliability"
              value="92.5"
              unit="%"
              change="1.2% better"
              changeType="up"
              icon={Shield}
            />
            <KPICard
              title="Avg Lead Time"
              value="8.8"
              unit="days"
              change="0.5 days faster"
              changeType="up"
              icon={TrendingUp}
            />
            <KPICard
              title="On-Time Delivery"
              value="94.2"
              unit="%"
              icon={Shield}
            />
          </div>

          <ChartCard title="Supplier Performance Comparison">
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={supplierData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="supplier" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip />
                <Legend />
                <Bar dataKey="reliability" fill="#3B82F6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Detailed Supplier Metrics</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-gray-600">Supplier</th>
                    <th className="text-left py-3 px-4 text-gray-600">Reliability Score</th>
                    <th className="text-left py-3 px-4 text-gray-600">Avg Lead Time</th>
                    <th className="text-left py-3 px-4 text-gray-600">Lead Time Variance</th>
                    <th className="text-left py-3 px-4 text-gray-600">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {supplierData.map((supplier, i) => (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-gray-900">{supplier.supplier}</td>
                      <td className="py-3 px-4 text-gray-700">{supplier.reliability}%</td>
                      <td className="py-3 px-4 text-gray-700">{supplier.leadTime} days</td>
                      <td className="py-3 px-4 text-gray-700">±{supplier.variance} days</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-lg text-sm ${
                          supplier.reliability >= 95 ? 'bg-green-100 text-green-700' :
                          supplier.reliability >= 90 ? 'bg-blue-100 text-blue-700' :
                          'bg-amber-100 text-amber-700'
                        }`}>
                          {supplier.reliability >= 95 ? 'Excellent' : supplier.reliability >= 90 ? 'Good' : 'Fair'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Anomaly Detection */}
      {activeTab === 'anomaly' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
            <KPICard
              title="Anomalies Detected"
              value="8"
              change="3 this week"
              icon={Activity}
            />
            <KPICard
              title="Data Quality Score"
              value="96.2"
              unit="%"
              change="1.2% better"
              changeType="up"
              icon={Activity}
            />
            <KPICard
              title="Missing Data Points"
              value="24"
              change="12 resolved"
              changeType="up"
              icon={Activity}
            />
          </div>

          <ChartCard title="Demand Anomaly Timeline">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={anomalyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="demand" stroke="#3B82F6" strokeWidth={2} dot={false} />
                <Scatter dataKey="anomaly" fill="#EF4444" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Recent Data Quality Issues</h3>
            <div className="space-y-3">
              {[
                { date: '2024-05-14', type: 'Spike', value: '+162%', sku: 'SKU-045', action: 'Investigated' },
                { date: '2024-05-12', type: 'Drop', value: '-78%', sku: 'SKU-129', action: 'Resolved' },
                { date: '2024-05-09', type: 'Missing', value: '12 hours', sku: 'Multiple', action: 'Imputed' },
                { date: '2024-05-06', type: 'Outlier', value: '+245%', sku: 'SKU-234', action: 'Validated' },
              ].map((issue, i) => (
                <div key={i} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                  <div className="flex items-center gap-4">
                    <div className={`w-2 h-2 rounded-full ${
                      issue.action === 'Investigated' ? 'bg-amber-500' : 'bg-green-500'
                    }`} />
                    <div>
                      <p className="text-gray-900">{issue.type} Anomaly</p>
                      <p className="text-sm text-gray-500">{issue.date} • {issue.sku}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-gray-900">{issue.value}</p>
                    <p className="text-sm text-gray-600">{issue.action}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Recommendations Drawer */}
      <Drawer
        isOpen={showRecommendations}
        onClose={() => setShowRecommendations(false)}
        title="Stockout Mitigation Recommendations"
      >
        <div className="space-y-4">
          {[
            {
              sku: 'SKU-001',
              action: 'Expedite Reorder',
              priority: 'Critical',
              impact: 'Prevent $45K revenue loss',
              timeline: 'Next 48 hours'
            },
            {
              sku: 'SKU-045',
              action: 'Increase Safety Stock',
              priority: 'High',
              impact: 'Reduce stockout risk by 40%',
              timeline: 'Next week'
            },
            {
              sku: 'SKU-129',
              action: 'Alternative Supplier',
              priority: 'Medium',
              impact: 'Faster lead time by 3 days',
              timeline: 'Next 2 weeks'
            },
          ].map((rec, i) => (
            <div key={i} className="p-4 border border-gray-200 rounded-xl space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-gray-900">{rec.sku}</p>
                  <p className="text-gray-600">{rec.action}</p>
                </div>
                <span className={`px-2 py-1 rounded-lg text-sm ${
                  rec.priority === 'Critical' ? 'bg-red-100 text-red-700' :
                  rec.priority === 'High' ? 'bg-amber-100 text-amber-700' :
                  'bg-yellow-100 text-yellow-700'
                }`}>
                  {rec.priority}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                <p>💡 {rec.impact}</p>
                <p>⏰ {rec.timeline}</p>
              </div>
              <button className="w-full py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                Execute Action
              </button>
            </div>
          ))}
        </div>
      </Drawer>
    </div>
  );
}
