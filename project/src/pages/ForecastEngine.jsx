import { useState } from 'react';
import { TrendingUp, GitBranch, Zap, Info } from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import Modal from '../components/Modal';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

const forecastData = [
  { date: '2024-01', actual: 4200, tft: 4100, nbeats: 4150 },
  { date: '2024-02', actual: 4500, tft: 4400, nbeats: 4450 },
  { date: '2024-03', actual: 4800, tft: 4900, nbeats: 4850 },
  { date: '2024-04', actual: 5200, tft: 5100, nbeats: 5150 },
  { date: '2024-05', actual: 5500, tft: 5600, nbeats: 5550 },
  { date: '2024-06', actual: null, tft: 6000, nbeats: 5950 },
  { date: '2024-07', actual: null, tft: 6200, nbeats: 6100 },
];

const shapData = [
  { feature: 'Price', importance: 0.35 },
  { feature: 'Seasonality', importance: 0.28 },
  { feature: 'Promotions', importance: 0.18 },
  { feature: 'Day of Week', importance: 0.12 },
  { feature: 'Weather', importance: 0.07 },
];

const intermittentData = [
  { date: 'W1', demand: 0, forecast: 0 },
  { date: 'W2', demand: 45, forecast: 42 },
  { date: 'W3', demand: 0, forecast: 0 },
  { date: 'W4', demand: 0, forecast: 0 },
  { date: 'W5', demand: 52, forecast: 48 },
  { date: 'W6', demand: 0, forecast: 0 },
  { date: 'W7', demand: 38, forecast: 40 },
];

const adjustmentData = [
  { date: '12:00', baseline: 850, adjusted: 920 },
  { date: '13:00', baseline: 870, adjusted: 940 },
  { date: '14:00', baseline: 890, adjusted: 950 },
  { date: '15:00', baseline: 910, adjusted: 980 },
  { date: '16:00', baseline: 920, adjusted: 1000 },
];

const modelVersions = [
  { version: 'v2.4.1', date: '2024-05-15', mape: 4.2, rmse: 156, status: 'production' },
  { version: 'v2.4.0', date: '2024-04-28', mape: 4.5, rmse: 168, status: 'archived' },
  { version: 'v2.3.9', date: '2024-04-10', mape: 4.8, rmse: 175, status: 'archived' },
];

export default function ForecastEngine() {
  const [activeTab, setActiveTab] = useState('global');
  const [showCompare, setShowCompare] = useState(false);

  const tabs = [
    { id: 'global', label: 'Global Forecaster', icon: TrendingUp },
    { id: 'intermittent', label: 'Intermittent Model', icon: GitBranch },
    { id: 'sensing', label: 'Short-Term Sensing', icon: Zap },
  ];

  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-gray-900 text-3xl mb-2">Forecast Engine</h1>
          <p className="text-gray-600">AI-powered demand prediction models</p>
        </div>
        <button
          onClick={() => setShowCompare(true)}
          className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors"
        >
          Compare Versions
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
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="sm:hidden">{tab.label.split(' ')[0]}</span>
            </button>
          );
        })}
      </div>

      {/* Global Forecaster */}
      {activeTab === 'global' && (
        <>
          {/* KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
            <KPICard
              title="Model MAPE"
              value="4.2"
              unit="%"
              change="0.3% better"
              changeType="up"
              icon={TrendingUp}
            />
            <KPICard
              title="RMSE"
              value="156"
              unit="units"
              change="12 lower"
              changeType="up"
              icon={TrendingUp}
            />
            <KPICard
              title="7-Day Horizon"
              value="3.8"
              unit="% MAPE"
              icon={TrendingUp}
            />
            <KPICard
              title="30-Day Horizon"
              value="5.1"
              unit="% MAPE"
              icon={TrendingUp}
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Multi-Model Forecast Comparison">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="tft" stroke="#3B82F6" strokeWidth={2} strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="nbeats" stroke="#8B5CF6" strokeWidth={2} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="SHAP Feature Importance">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={shapData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis type="number" stroke="#6b7280" />
                  <YAxis dataKey="feature" type="category" stroke="#6b7280" width={100} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#3B82F6" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Horizon-wise Accuracy Table */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Horizon-wise Accuracy Metrics</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-gray-600">Horizon</th>
                    <th className="text-left py-3 px-4 text-gray-600">MAPE</th>
                    <th className="text-left py-3 px-4 text-gray-600">RMSE</th>
                    <th className="text-left py-3 px-4 text-gray-600">Coverage</th>
                    <th className="text-left py-3 px-4 text-gray-600">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { horizon: '1 Day', mape: '2.8%', rmse: '98', coverage: '97.2%', status: 'Excellent' },
                    { horizon: '7 Days', mape: '3.8%', rmse: '142', coverage: '95.8%', status: 'Good' },
                    { horizon: '14 Days', mape: '4.5%', rmse: '168', coverage: '94.1%', status: 'Good' },
                    { horizon: '30 Days', mape: '5.1%', rmse: '189', coverage: '92.5%', status: 'Acceptable' },
                  ].map((row, i) => (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-gray-900">{row.horizon}</td>
                      <td className="py-3 px-4 text-gray-700">{row.mape}</td>
                      <td className="py-3 px-4 text-gray-700">{row.rmse}</td>
                      <td className="py-3 px-4 text-gray-700">{row.coverage}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-lg text-sm ${
                          row.status === 'Excellent' ? 'bg-green-100 text-green-700' :
                          row.status === 'Good' ? 'bg-blue-100 text-blue-700' :
                          'bg-amber-100 text-amber-700'
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
        </>
      )}

      {/* Intermittent Model */}
      {activeTab === 'intermittent' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
            <KPICard
              title="Zero-Inflation Rate"
              value="68"
              unit="%"
              icon={GitBranch}
            />
            <KPICard
              title="Croston MAPE"
              value="12.4"
              unit="%"
              change="1.2% better"
              changeType="up"
              icon={TrendingUp}
            />
            <KPICard
              title="Non-Zero Accuracy"
              value="8.6"
              unit="% MAPE"
              icon={TrendingUp}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ChartCard title="Intermittent Demand Pattern">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={intermittentData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Legend />
                  <Line type="stepAfter" dataKey="demand" stroke="#10B981" strokeWidth={2} />
                  <Line type="stepAfter" dataKey="forecast" stroke="#3B82F6" strokeWidth={2} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Zero vs Non-Zero Distribution">
              <div className="space-y-4 pt-8">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600">Zero Demand Periods</span>
                    <span className="text-gray-900">68%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div className="bg-gray-500 h-3 rounded-full" style={{ width: '68%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600">Non-Zero Demand</span>
                    <span className="text-gray-900">32%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div className="bg-blue-500 h-3 rounded-full" style={{ width: '32%' }} />
                  </div>
                </div>
                <div className="mt-8 p-4 bg-blue-50 rounded-xl">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-blue-900 mb-1">Croston Method Applied</p>
                      <p className="text-blue-700 text-sm">Optimized for sporadic demand patterns with long zero-periods</p>
                    </div>
                  </div>
                </div>
              </div>
            </ChartCard>
          </div>
        </>
      )}

      {/* Short-Term Demand Sensing */}
      {activeTab === 'sensing' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
            <KPICard
              title="Real-time Adjustment"
              value="+8.2"
              unit="%"
              icon={Zap}
            />
            <KPICard
              title="Latency"
              value="45"
              unit="sec"
              icon={Zap}
            />
            <KPICard
              title="Data Freshness"
              value="5"
              unit="min"
              icon={Zap}
            />
          </div>

          <ChartCard title="Baseline vs Real-Time Adjusted Forecast">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={adjustmentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="baseline" stroke="#9CA3AF" strokeWidth={2} />
                <Line type="monotone" dataKey="adjusted" stroke="#3B82F6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Real-Time Data Ingestion Timeline</h3>
            <div className="space-y-3">
              {[
                { time: '16:45:23', source: 'POS System', status: 'Ingested', latency: '2s' },
                { time: '16:45:18', source: 'Web Analytics', status: 'Ingested', latency: '3s' },
                { time: '16:45:12', source: 'Inventory Feed', status: 'Ingested', latency: '1s' },
                { time: '16:45:05', source: 'Weather API', status: 'Ingested', latency: '4s' },
              ].map((item, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <div>
                      <p className="text-gray-900">{item.source}</p>
                      <p className="text-sm text-gray-500">{item.time}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-gray-600">{item.latency}</span>
                    <span className="px-2 py-1 bg-green-100 text-green-700 text-sm rounded-lg">
                      {item.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Model Version Compare Modal */}
      <Modal
        isOpen={showCompare}
        onClose={() => setShowCompare(false)}
        title="Compare Model Versions"
        size="xl"
      >
        <div className="space-y-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-gray-600">Version</th>
                  <th className="text-left py-3 px-4 text-gray-600">Date</th>
                  <th className="text-left py-3 px-4 text-gray-600">MAPE</th>
                  <th className="text-left py-3 px-4 text-gray-600">RMSE</th>
                  <th className="text-left py-3 px-4 text-gray-600">Status</th>
                </tr>
              </thead>
              <tbody>
                {modelVersions.map((version, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-gray-900">{version.version}</td>
                    <td className="py-3 px-4 text-gray-700">{version.date}</td>
                    <td className="py-3 px-4 text-gray-700">{version.mape}%</td>
                    <td className="py-3 px-4 text-gray-700">{version.rmse}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-lg text-sm ${
                        version.status === 'production'
                          ? 'bg-green-100 text-green-700'
                          : 'bg-gray-100 text-gray-700'
                      }`}>
                        {version.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex gap-3 justify-end">
            <button
              onClick={() => setShowCompare(false)}
              className="px-6 py-2.5 border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors"
            >
              Close
            </button>
            <button className="px-6 py-2.5 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors">
              Deploy Selected
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
