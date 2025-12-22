import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  Package, 
  Target,
  CheckCircle2,
  Database,
  DollarSign,
  ChevronRight
} from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import Modal from '../components/Modal';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

const forecastData = [
  { date: 'Jan', actual: 4200, forecast: 4100, lower: 3900, upper: 4300 },
  { date: 'Feb', actual: 4500, forecast: 4400, lower: 4200, upper: 4600 },
  { date: 'Mar', actual: 4800, forecast: 4900, lower: 4700, upper: 5100 },
  { date: 'Apr', actual: 5200, forecast: 5100, lower: 4900, upper: 5300 },
  { date: 'May', actual: 5500, forecast: 5600, lower: 5400, upper: 5800 },
  { date: 'Jun', actual: null, forecast: 6000, lower: 5700, upper: 6300 },
  { date: 'Jul', actual: null, forecast: 6200, lower: 5900, upper: 6500 },
];

const demandTrendData = [
  { date: 'W1', demand: 850 },
  { date: 'W2', demand: 920 },
  { date: 'W3', demand: 880 },
  { date: 'W4', demand: 1100 },
  { date: 'W5', demand: 1050 },
  { date: 'W6', demand: 1200 },
  { date: 'W7', demand: 1180 },
  { date: 'W8', demand: 1350 },
];

const promoUpliftData = [
  { promo: 'Black Friday', uplift: 240 },
  { promo: 'Spring Sale', uplift: 180 },
  { promo: 'Summer Promo', uplift: 160 },
  { promo: 'Back to School', uplift: 140 },
  { promo: 'Holiday Sale', uplift: 200 },
];

const highRiskSKUs = [
  { id: 'SKU-001', name: 'Premium Widget Pro', risk: 92, stockout: '78%', location: 'NY-001' },
  { id: 'SKU-045', name: 'Deluxe Component X', risk: 88, stockout: '72%', location: 'LA-003' },
  { id: 'SKU-129', name: 'Ultra Module Z', risk: 85, stockout: '68%', location: 'CHI-002' },
  { id: 'SKU-234', name: 'Advanced Kit Alpha', risk: 82, stockout: '65%', location: 'SF-001' },
  { id: 'SKU-456', name: 'Elite Assembly Beta', risk: 79, stockout: '61%', location: 'SEA-004' },
];

export default function HomePage() {
  const navigate = useNavigate();
  const [expandedChart, setExpandedChart] = useState(null);
  const [selectedSKU, setSelectedSKU] = useState(null);

  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-gray-900 text-3xl mb-2">Demand Intelligence Dashboard</h1>
          <p className="text-gray-600">Real-time forecasting and inventory insights</p>
        </div>
        <div className="flex items-center gap-3">
          <select className="px-4 py-2 bg-white border border-gray-200 rounded-xl text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option>Last 30 Days</option>
            <option>Last 90 Days</option>
            <option>Last 6 Months</option>
            <option>Last Year</option>
          </select>
        </div>
      </div>

      {/* Alerts Banner */}
      <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-amber-900">
            <span>5 SKUs predicted to stock out within 7 days. </span>
            <button 
              onClick={() => navigate('/risk')}
              className="text-amber-700 hover:text-amber-800 underline"
            >
              Review now
            </button>
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
        <KPICard
          title="MAPE (Accuracy)"
          value="4.2"
          unit="%"
          change="0.3% better"
          changeType="up"
          icon={Activity}
        />
        <KPICard
          title="Stockout Rate"
          value="2.1"
          unit="%"
          change="0.5% lower"
          changeType="up"
          icon={AlertTriangle}
        />
        <KPICard
          title="Inventory Turns"
          value="8.4"
          change="0.6 higher"
          changeType="up"
          icon={Package}
        />
        <KPICard
          title="Revenue Impact"
          value="$2.4M"
          change="$340K increase"
          changeType="up"
          icon={DollarSign}
        />
      </div>

      {/* Additional KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
        <KPICard
          title="RMSE"
          value="156"
          unit="units"
          change="12 lower"
          changeType="up"
          icon={Target}
        />
        <KPICard
          title="Interval Coverage"
          value="94.8"
          unit="%"
          change="stable"
          changeType="neutral"
          icon={CheckCircle2}
        />
        <KPICard
          title="Data Quality Score"
          value="96.2"
          unit="%"
          change="1.2% better"
          changeType="up"
          icon={Database}
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Forecast vs Actual */}
        <ChartCard
          title="Forecast vs Actual"
          onExpand={() => setExpandedChart('forecast')}
        >
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={forecastData}>
              <defs>
                <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.1}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="upper" stroke="none" fill="#E0E7FF" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="#fff" />
              <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={2} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="forecast" stroke="#3B82F6" strokeWidth={2} strokeDasharray="5 5" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Demand Trend */}
        <ChartCard
          title="30-Day Demand Trend"
          onExpand={() => setExpandedChart('demand')}
        >
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={demandTrendData}>
              <defs>
                <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Area type="monotone" dataKey="demand" stroke="#3B82F6" fillOpacity={1} fill="url(#colorDemand)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Promo Uplift */}
        <ChartCard
          title="Promotional Uplift Impact"
          onExpand={() => setExpandedChart('promo')}
        >
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={promoUpliftData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="promo" stroke="#6b7280" angle={-20} textAnchor="end" height={80} />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Bar dataKey="uplift" fill="#3B82F6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* High Risk SKUs */}
        <ChartCard
          title="Top High-Risk SKUs"
          action={
            <button
              onClick={() => navigate('/risk')}
              className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
            >
              View all
              <ChevronRight className="w-4 h-4" />
            </button>
          }
        >
          <div className="space-y-3">
            {highRiskSKUs.map((sku) => (
              <div
                key={sku.id}
                onClick={() => {
                  setSelectedSKU(sku);
                  navigate(`/sku/${sku.id}`);
                }}
                className="p-4 bg-gray-50 rounded-xl hover:bg-gray-100 cursor-pointer transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex-1">
                    <p className="text-gray-900">{sku.name}</p>
                    <p className="text-sm text-gray-500">{sku.id} • {sku.location}</p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center gap-2">
                      <div className={`px-2 py-1 rounded-lg text-sm ${
                        sku.risk >= 85 ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'
                      }`}>
                        {sku.risk}% risk
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{sku.stockout} stockout</p>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      sku.risk >= 85 ? 'bg-red-500' : 'bg-amber-500'
                    }`}
                    style={{ width: `${sku.risk}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <button
          onClick={() => navigate('/insights')}
          className="p-6 bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-2xl hover:shadow-lg transition-all text-left"
        >
          <TrendingUp className="w-8 h-8 mb-3" />
          <h3 className="mb-1">AI Insights</h3>
          <p className="text-sm text-blue-100">Get intelligent recommendations</p>
        </button>

        <button
          onClick={() => navigate('/forecast')}
          className="p-6 bg-white border border-gray-200 rounded-2xl hover:shadow-md transition-all text-left"
        >
          <Activity className="w-8 h-8 text-blue-600 mb-3" />
          <h3 className="text-gray-900 mb-1">Run Forecast</h3>
          <p className="text-sm text-gray-600">Generate new predictions</p>
        </button>

        <button
          onClick={() => navigate('/optimizer')}
          className="p-6 bg-white border border-gray-200 rounded-2xl hover:shadow-md transition-all text-left"
        >
          <Package className="w-8 h-8 text-blue-600 mb-3" />
          <h3 className="text-gray-900 mb-1">Optimize Inventory</h3>
          <p className="text-sm text-gray-600">Balance stock levels</p>
        </button>

        <button
          onClick={() => navigate('/retraining')}
          className="p-6 bg-white border border-gray-200 rounded-2xl hover:shadow-md transition-all text-left"
        >
          <CheckCircle2 className="w-8 h-8 text-blue-600 mb-3" />
          <h3 className="text-gray-900 mb-1">Retrain Models</h3>
          <p className="text-sm text-gray-600">Update with new data</p>
        </button>
      </div>

      {/* Expanded Chart Modal */}
      <Modal
        isOpen={expandedChart !== null}
        onClose={() => setExpandedChart(null)}
        title={
          expandedChart === 'forecast' ? 'Forecast vs Actual' :
          expandedChart === 'demand' ? '30-Day Demand Trend' :
          'Promotional Uplift Impact'
        }
        size="xl"
      >
        <ResponsiveContainer width="100%" height={500}>
          {expandedChart === 'forecast' ? (
            <AreaChart data={forecastData}>
              <defs>
                <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.1}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="upper" stroke="none" fill="#E0E7FF" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="#fff" />
              <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={3} dot={{ r: 5 }} />
              <Line type="monotone" dataKey="forecast" stroke="#3B82F6" strokeWidth={3} strokeDasharray="5 5" />
            </AreaChart>
          ) : expandedChart === 'demand' ? (
            <AreaChart data={demandTrendData}>
              <defs>
                <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Area type="monotone" dataKey="demand" stroke="#3B82F6" fillOpacity={1} fill="url(#colorDemand)" strokeWidth={3} />
            </AreaChart>
          ) : (
            <BarChart data={promoUpliftData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="promo" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Bar dataKey="uplift" fill="#3B82F6" radius={[8, 8, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </Modal>
    </div>
  );
}
