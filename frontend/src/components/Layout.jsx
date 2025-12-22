import { useState } from 'react';
import { Outlet, useLocation, Link } from 'react-router-dom';
import { 
  LayoutDashboard, 
  TrendingUp, 
  AlertTriangle, 
  Target, 
  Package, 
  Brain, 
  RefreshCw, 
  Upload,
  Menu,
  X,
  Home,
  BarChart3
} from 'lucide-react';
import path from 'path';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/forecast', icon: TrendingUp, label: 'Forecast Engine' },
  { path: '/risk', icon: AlertTriangle, label: 'Risk & Reliability' },
  { path: '/causal', icon: Target, label: 'Causal & Promo' },
  { path: '/optimizer', icon: Package, label: 'Inventory Optimizer' },
  { path: '/insights', icon: Brain, label: 'AI Insights' },
  { path: '/retraining', icon: RefreshCw, label: 'Retraining Center' },
  { path: '/model-training', icon: Upload, label: 'Model Training'},
  { path: '/data-upload', icon: Upload, label: 'Data Upload' },
];

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-50 via-blue-50/20 to-gray-50">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col bg-white/80 backdrop-blur-xl border-r border-gray-200/50 shadow-lg">
        <div className="flex flex-col flex-1 min-h-0">
          <div className="flex items-center h-16 px-6 border-b border-gray-200/50 bg-linear-to-r from-blue-50/50 to-transparent">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-linear-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <span className="text-gray-900 font-semibold">DemandCast</span>
            </div>
          </div>
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`sidebar-nav-item flex items-center px-3 py-2.5 rounded-xl transition-all duration-200 ${
                    isActive
                      ? 'bg-linear-to-r from-blue-50 to-blue-50/50 text-blue-600 shadow-sm active'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  <span className="font-medium">{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-white/90 backdrop-blur-xl border-b border-gray-200/50 shadow-sm">
        <div className="flex items-center justify-between h-16 px-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-linear-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <span className="text-gray-900 font-semibold">DemandCast</span>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-xl hover:bg-gray-100 transition-colors"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-30 bg-black/20 backdrop-blur-sm" onClick={() => setMobileMenuOpen(false)}>
          <div 
            className="fixed inset-y-0 left-0 w-64 bg-white shadow-2xl transform transition-transform"
            onClick={(e) => e.stopPropagation()}
          >
            <nav className="px-3 py-4 mt-16 space-y-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center px-3 py-2.5 rounded-xl transition-all duration-200 ${
                      isActive
                        ? 'bg-linear-to-r from-blue-50 to-blue-50/50 text-blue-600 shadow-sm'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-5 h-5 mr-3" />
                    <span className="font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      )}

      {/* Mobile Bottom Navigation */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 z-30 bg-white/90 backdrop-blur-xl border-t border-gray-200/50 shadow-lg">
        <div className="grid grid-cols-4 gap-1 px-2 py-2">
          <Link
            to="/"
            className={`flex flex-col items-center py-2 px-3 rounded-xl transition-all ${
              location.pathname === '/' ? 'text-blue-600 bg-blue-50' : 'text-gray-600'
            }`}
          >
            <Home className="w-5 h-5" />
            <span className="text-xs mt-1 font-medium">Home</span>
          </Link>
          <Link
            to="/forecast"
            className={`flex flex-col items-center py-2 px-3 rounded-xl transition-all ${
              location.pathname === '/forecast' ? 'text-blue-600 bg-blue-50' : 'text-gray-600'
            }`}
          >
            <TrendingUp className="w-5 h-5" />
            <span className="text-xs mt-1 font-medium">Forecast</span>
          </Link>
          <Link
            to="/insights"
            className={`flex flex-col items-center py-2 px-3 rounded-xl transition-all ${
              location.pathname === '/insights' ? 'text-blue-600 bg-blue-50' : 'text-gray-600'
            }`}
          >
            <Brain className="w-5 h-5" />
            <span className="text-xs mt-1 font-medium">Insights</span>
          </Link>
          <Link
            to="/optimizer"
            className={`flex flex-col items-center py-2 px-3 rounded-xl transition-all ${
              location.pathname === '/optimizer' ? 'text-blue-600 bg-blue-50' : 'text-gray-600'
            }`}
          >
            <Package className="w-5 h-5" />
            <span className="text-xs mt-1 font-medium">Optimize</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <main className="lg:pl-64 pt-16 lg:pt-0 pb-20 lg:pb-0">
        <div className="min-h-screen">
          <Outlet />
        </div>
      </main>
    </div>
  );
}