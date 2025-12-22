import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingUp, AlertTriangle, Package, Target, Truck } from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const skuForecastData = [
  { date: 'Jan', actual: 420, forecast: 410, lower: 390, upper: 430 },
  { date: 'Feb', actual: 450, forecast: 440, lower: 420, upper: 460 },
  { date: 'Mar', actual: 480, forecast: 490, lower: 470, upper: 510 },
  { date: 'Apr', actual: 520, forecast: 510, lower: 490, upper: 530 },
  { date: 'May', actual: 550, forecast: 560, lower: 540, upper: 580 },
  { date: 'Jun', actual: null, forecast: 600, lower: 570, upper: 630 },
];

const riskTimelineData = [
  { date: 'W1', risk: 15 },
  { date: 'W2', risk: 22 },
  { date: 'W3', risk: 35 },
  { date: 'W4', risk: 52 },
  { date: 'W5', risk: 68 },
  { date: 'W6', risk: 78 },
];

const promoImpactData = [
  { event: 'Black Friday', uplift: 180 },
  { event: 'Spring Sale', uplift: 95 },
  { event: 'Summer Promo', uplift: 110 },
];

const supplierLeadTimeData = [
  { supplier: 'Supplier A', leadTime: 7, variance: 1.2, reliability: 96 },
  { supplier: 'Supplier B', leadTime: 9, variance: 2.1, reliability: 92 },
  { supplier: 'Supplier C', leadTime: 11, variance: 3.4, reliability: 88 },
];

export default function SKUDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'forecast', label: 'Forecast' },
    { id: 'risk', label: 'Risk' },
    { id: 'promotions', label: 'Promotions' },
    { id: 'supply', label: 'Supply Chain' },
  ];

  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={() => navigate('/')}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </button>
        <div className="flex-1">
          <h1 className="text-gray-900 text-3xl mb-1">Premium Widget Pro</h1>
          <p className="text-gray-600">{id} • NY-001 Warehouse</p>
        </div>
        <button className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors">
          Create Order
        </button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <KPICard
          title="Current Stock"
          value="120"
          unit="units"
          icon={Package}
        />
        <KPICard
          title="Forecast (30d)"
          value="1,680"
          unit="units"
          icon={TrendingUp}
        />
        <KPICard
          title="Stockout Risk"
          value="92"
          unit="%"
          icon={AlertTriangle}
        />
        <KPICard
          title="Avg Lead Time"
          value="7"
          unit="days"
          icon={Truck}
        />
        <KPICard
          title="Unit Price"
          value="$98"
          icon={Target}
        />
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-2xl p-2 border border-gray-200">
        <div className="flex gap-2 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2.5 rounded-xl transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="SKU Demand Forecast">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={skuForecastData}>
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

          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Key Information</h3>
            <div className="space-y-4">
              <div className="flex justify-between py-3 border-b border-gray-100">
                <span className="text-gray-600">Category</span>
                <span className="text-gray-900">Premium Electronics</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-100">
                <span className="text-gray-600">Supplier</span>
                <span className="text-gray-900">Supplier A</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-100">
                <span className="text-gray-600">Min Order Qty</span>
                <span className="text-gray-900">200 units</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-100">
                <span className="text-gray-600">Safety Stock</span>
                <span className="text-gray-900">150 units</span>
              </div>
              <div className="flex justify-between py-3 border-b border-gray-100">
                <span className="text-gray-600">Reorder Point</span>
                <span className="text-gray-900">200 units</span>
              </div>
              <div className="flex justify-between py-3">
                <span className="text-gray-600">Last Restock</span>
                <span className="text-gray-900">2024-04-28</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Forecast Tab */}
      {activeTab === 'forecast' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <KPICard
              title="7-Day Forecast"
              value="392"
              unit="units"
              icon={TrendingUp}
            />
            <KPICard
              title="30-Day Forecast"
              value="1,680"
              unit="units"
              icon={TrendingUp}
            />
            <KPICard
              title="Forecast MAPE"
              value="3.8"
              unit="%"
              icon={Target}
            />
            <KPICard
              title="Confidence"
              value="94"
              unit="%"
              icon={Target}
            />
          </div>

          <ChartCard title="Detailed Forecast with Confidence Intervals">
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={skuForecastData}>
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
                <Area type="monotone" dataKey="upper" stroke="none" fill="#E0E7FF" name="Upper Bound" />
                <Area type="monotone" dataKey="lower" stroke="none" fill="#fff" name="Lower Bound" />
                <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={2} dot={{ r: 4 }} name="Actual" />
                <Line type="monotone" dataKey="forecast" stroke="#3B82F6" strokeWidth={2} strokeDasharray="5 5" name="Forecast" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      {/* Risk Tab */}
      {activeTab === 'risk' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <KPICard
              title="Stockout Probability"
              value="92"
              unit="%"
              icon={AlertTriangle}
            />
            <KPICard
              title="Days Until Stockout"
              value="3"
              unit="days"
              icon={AlertTriangle}
            />
            <KPICard
              title="Risk Level"
              value="Critical"
              icon={AlertTriangle}
            />
          </div>

          <ChartCard title="Risk Progression Timeline">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={riskTimelineData}>
                <defs>
                  <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip />
                <Area type="monotone" dataKey="risk" stroke="#EF4444" fillOpacity={1} fill="url(#colorRisk)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="text-amber-900 mb-2">Immediate Action Required</h4>
                <p className="text-amber-800 mb-4">
                  This SKU is at critical risk of stockout. Current inventory levels are below safety stock threshold.
                </p>
                <div className="flex gap-3">
                  <button className="px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition-colors">
                    Expedite Reorder
                  </button>
                  <button className="px-4 py-2 bg-white text-amber-900 border border-amber-300 rounded-lg hover:bg-amber-50 transition-colors">
                    View Alternatives
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Promotions Tab */}
      {activeTab === 'promotions' && (
        <div className="space-y-6">
          <ChartCard title="Historical Promotional Impact">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={promoImpactData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="event" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip />
                <Bar dataKey="uplift" fill="#3B82F6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Upcoming Promotions</h3>
            <div className="space-y-3">
              {[
                { name: 'Summer Flash Sale', date: '2024-06-15', discount: '20%', forecast: '+165%' },
                { name: 'Mid-Year Clearance', date: '2024-07-01', discount: '30%', forecast: '+220%' },
              ].map((promo, i) => (
                <div key={i} className="p-4 bg-gray-50 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-900">{promo.name}</p>
                      <p className="text-sm text-gray-600">{promo.date} • {promo.discount} discount</p>
                    </div>
                    <div className="text-right">
                      <span className="px-3 py-1 bg-green-100 text-green-700 rounded-lg text-sm">
                        {promo.forecast} uplift
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Supply Chain Tab */}
      {activeTab === 'supply' && (
        <div className="space-y-6">
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Supplier Performance</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-gray-600">Supplier</th>
                    <th className="text-left py-3 px-4 text-gray-600">Lead Time</th>
                    <th className="text-left py-3 px-4 text-gray-600">Variance</th>
                    <th className="text-left py-3 px-4 text-gray-600">Reliability</th>
                    <th className="text-left py-3 px-4 text-gray-600">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {supplierLeadTimeData.map((supplier, i) => (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-gray-900">{supplier.supplier}</td>
                      <td className="py-3 px-4 text-gray-700">{supplier.leadTime} days</td>
                      <td className="py-3 px-4 text-gray-700">±{supplier.variance} days</td>
                      <td className="py-3 px-4 text-gray-700">{supplier.reliability}%</td>
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

          <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-gray-900 mb-4">Recent Orders</h3>
            <div className="space-y-3">
              {[
                { date: '2024-04-28', qty: 450, supplier: 'Supplier A', status: 'Delivered', leadTime: 6 },
                { date: '2024-04-15', qty: 420, supplier: 'Supplier A', status: 'Delivered', leadTime: 7 },
                { date: '2024-04-01', qty: 480, supplier: 'Supplier B', status: 'Delivered', leadTime: 9 },
              ].map((order, i) => (
                <div key={i} className="p-4 bg-gray-50 rounded-xl flex items-center justify-between">
                  <div>
                    <p className="text-gray-900">{order.qty} units from {order.supplier}</p>
                    <p className="text-sm text-gray-600">{order.date} • {order.leadTime} days lead time</p>
                  </div>
                  <span className="px-3 py-1 bg-green-100 text-green-700 rounded-lg text-sm">
                    {order.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Recommendations Section */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-gray-900 mb-4">AI Recommendations</h3>
        <div className="space-y-3">
          <div className="p-4 bg-blue-50 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Package className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="text-blue-900 mb-1">Reorder 450 units immediately</p>
                <p className="text-blue-700 text-sm">
                  Based on forecast and current risk level, reordering now will prevent stockout and maintain optimal inventory levels.
                </p>
              </div>
            </div>
          </div>

          <div className="p-4 bg-amber-50 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-amber-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Truck className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="text-amber-900 mb-1">Consider expedited shipping</p>
                <p className="text-amber-700 text-sm">
                  Standard lead time of 7 days may result in stockout. Expedited shipping (3 days) recommended.
                </p>
              </div>
            </div>
          </div>

          <div className="p-4 bg-green-50 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Target className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <p className="text-green-900 mb-1">Increase safety stock to 200 units</p>
                <p className="text-green-700 text-sm">
                  Higher demand variability detected. Increasing safety stock will reduce future stockout risk by 40%.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
