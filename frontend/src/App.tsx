import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import ForecastEngine from './pages/ForecastEngine';
import RiskReliability from './pages/RiskReliability';
import CausalPromo from './pages/CausalPromo';
import InventoryOptimizer from './pages/InventoryOptimizer';
import SKUDetail from './pages/SKUDetail';
import Insights from './pages/Insights';
import RetrainingCenter from './pages/RetrainingCenter';
import DataUpload from './pages/DataUpload';
import Login from './pages/Login';
import Signup from './pages/signup';
import AuthCallback from './pages/authcallback';
import ModelTrainingOrchestration from './pages/model_training';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!sessionStorage.getItem("token")
  );

  // When token changes, update login state
  useEffect(() => {
    setIsAuthenticated(!!sessionStorage.getItem("token"));
  }, []);

  return (
    <Router>
      <Routes>

        {/* OAuth redirect */}
        <Route path="/auth/callback" element={<AuthCallback />} />

        {/* Public pages */}
        <Route
          path="/login"
          element={<Login onLogin={() => setIsAuthenticated(true)} />}
        />
        <Route path="/signup" element={<Signup />} />

        {/* Protected routes */}
        <Route
          path="/"
          element={
            isAuthenticated ? (
              <Layout />
            ) : (
              <Navigate to="/login" replace />
            )
          }
        >
          <Route index element={<HomePage />} />
          <Route path="forecast" element={<ForecastEngine />} />
          <Route path="risk" element={<RiskReliability />} />
          <Route path="causal" element={<CausalPromo />} />
          <Route path="optimizer" element={<InventoryOptimizer />} />
          <Route path="sku/:id" element={<SKUDetail />} />
          <Route path="insights" element={<Insights />} />
          <Route path="retraining" element={<RetrainingCenter />} />
          <Route path="data-upload" element={<DataUpload />} />
          <Route path="model-training" element={<ModelTrainingOrchestration />} />
        </Route>

      </Routes>
    </Router>
  );
}
