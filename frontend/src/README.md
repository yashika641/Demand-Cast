# DemandCast - AI-Powered Demand Forecasting Platform

A full-featured, enterprise-grade clickable prototype for demand forecasting and inventory intelligence.

## 🎯 Features

### Core Pages

1. **Home Dashboard**
   - Real-time KPI cards (MAPE, RMSE, Stockout Rate, Inventory Turns, etc.)
   - Interactive charts (Forecast vs Actual, Risk Heatmap, Demand Trends)
   - High-risk SKU alerts and monitoring
   - Quick action cards

2. **Forecast Engine**
   - Global Forecaster (TFT/N-BEATS models)
   - Intermittent demand modeling
   - Short-term demand sensing
   - Model version comparison
   - SHAP feature importance visualization

3. **Risk & Reliability**
   - Stockout probability analysis
   - Monte Carlo simulations
   - Supplier reliability metrics
   - Anomaly detection
   - Data quality monitoring

4. **Causal & Promo Insights**
   - Price elasticity analysis
   - Promotional uplift measurement
   - ROI calculations
   - Counterfactual scenario simulator

5. **Inventory Optimizer**
   - Economic Order Quantity (EOQ) analysis
   - Multi-location allocation planning
   - Safety stock recommendations
   - Reorder schedule management

6. **SKU Detail Page**
   - Individual SKU analytics
   - Tabbed views (Overview, Forecast, Risk, Promotions, Supply Chain)
   - AI-powered recommendations
   - Supplier performance tracking

7. **AI Insights Engine**
   - RAG-powered conversational AI
   - Evidence-based recommendations
   - SHAP explanations
   - Confidence scores
   - Chat history and quick prompts

8. **Retraining Center**
   - ML pipeline visualization
   - Training history and metrics
   - Model version management
   - Automated retraining triggers
   - Real-time logs viewer

9. **Data Upload**
   - Drag & drop file upload
   - Auto-schema detection
   - Field mapping interface
   - Data validation
   - Error reporting

10. **Authentication**
    - Login page with role-based access UI
    - Remember me functionality
    - Password recovery flow

## 🎨 Design System

- **Primary Color**: Blue (#3B82F6)
- **Typography**: Inter font family with optimized weight hierarchy
- **Border Radius**: 12-20px rounded cards
- **Shadows**: Multi-layered soft shadows with glow effects
- **Layout**: Clean, modern, minimalistic with glassmorphism
- **Animations**: Spring-based transitions, hover lifts, smooth fades

### Modern UI Features
- ✨ Glassmorphism effects with backdrop blur
- 🎨 Gradient accents and animated backgrounds
- 💫 Smooth spring animations for modals and drawers
- 🌈 Gradient borders and shadow glows
- 🎯 Hover lift effects on interactive cards
- 🔮 Frosted glass sidebar with backdrop blur
- ⚡ Hardware-accelerated transitions
- 📱 Touch-optimized mobile interactions

## 📱 Responsive Design

### Desktop
- Full sidebar navigation
- Multi-column grid layouts
- Expanded charts and tables
- Rich data visualizations

### Mobile
- Collapsible hamburger menu
- Single-column stacked layout
- Bottom navigation bar with quick access
- Touch-optimized interactions
- Drawer/modal overlays

## 🔄 Interactive Features

### Clickable Elements
- ✅ Sidebar navigation between pages
- ✅ Chart expand/fullscreen views
- ✅ SKU cards → SKU detail pages
- ✅ Modal and drawer triggers
- ✅ Tab navigation
- ✅ "Trigger Retrain" workflow
- ✅ "Run Simulation" modals
- ✅ Evidence drawers in AI Insights
- ✅ Version comparison
- ✅ File upload wizard

### Animations
- Smooth page transitions
- Slide-in drawers from right/bottom
- Fade-in modals
- Hover effects on cards
- Progress animations
- Loading states

## 🛠️ Tech Stack

- **Framework**: React 18 with Hooks
- **Routing**: React Router v6
- **Styling**: Tailwind CSS v4
- **Charts**: Recharts
- **Icons**: Lucide React
- **Animations**: Motion (Framer Motion)
- **Build Tool**: Vite

## 📂 Project Structure

```
/
├── components/
│   ├── Layout.jsx          # Main layout with sidebar & mobile nav
│   ├── KPICard.jsx         # Reusable KPI card component
│   ├── ChartCard.jsx       # Chart wrapper component
│   ├── Modal.jsx           # Modal dialog component
│   └── Drawer.jsx          # Slide-in drawer component
├── pages/
│   ├── HomePage.jsx        # Main dashboard
│   ├── ForecastEngine.jsx  # Forecast models
│   ├── RiskReliability.jsx # Risk analysis
│   ├── CausalPromo.jsx     # Promotional insights
│   ├── InventoryOptimizer.jsx
│   ├── SKUDetail.jsx       # Individual SKU view
│   ├── Insights.jsx        # AI chat interface
│   ├── RetrainingCenter.jsx
│   ├── DataUpload.jsx
│   └── Login.jsx
└── App.tsx                 # Main app component
```

## 🚀 Getting Started

1. Click "Sign in" on the login page (no credentials required for demo)
2. Explore the dashboard with interactive KPI cards and charts
3. Navigate using the sidebar or mobile bottom navigation
4. Click on SKU cards to view detailed analytics
5. Try the AI Insights chat with quick prompts
6. Trigger model retraining from the Retraining Center
7. Upload data files with the Data Upload wizard

## 💡 Key Interactions

### From Dashboard
- Click high-risk SKUs → Navigate to SKU Detail page
- Click charts → Expand to fullscreen modal
- Click "AI Insights" → Open chat interface
- Click "Run Forecast" → Navigate to Forecast Engine

### From Forecast Engine
- Switch between Global/Intermittent/Sensing tabs
- Click "Compare Versions" → Open version comparison modal
- View SHAP importance and accuracy metrics

### From Risk & Reliability
- Switch between Stockout/Supplier/Anomaly tabs
- Click "View Recommendations" → Open recommendations drawer
- Review real-time risk progression

### From Insights
- Type queries or use quick prompts
- View evidence citations
- Check SHAP explanations
- Browse previous insights

### From Retraining Center
- View ML pipeline steps
- Click "Trigger Retrain" → Configure and start retraining
- Monitor progress in real-time
- Compare model versions

## 📊 Sample Data

The prototype includes realistic mock data for:
- 150,000 demand records
- 1,247 unique SKUs
- 5 warehouse locations
- 4 supplier relationships
- Historical promotional campaigns
- Forecast accuracy metrics
- Risk assessments

## 🎯 Use Cases Demonstrated

1. **Demand Forecasting**: Multi-horizon predictions with confidence intervals
2. **Inventory Optimization**: EOQ, safety stock, reorder point calculations
3. **Risk Management**: Stockout probability and supplier reliability
4. **Promotional Planning**: Uplift measurement and ROI optimization
5. **Supply Chain Intelligence**: Lead time analysis and allocation
6. **AI-Driven Insights**: Conversational analytics with evidence
7. **Model Operations**: Automated ML pipeline and monitoring

## 📱 Mobile Features

- Hamburger menu for navigation
- Bottom quick-access bar (Home, Forecast, Insights, Optimize)
- Responsive tables and charts
- Touch-friendly drawers and modals
- Single-column layouts
- Optimized input controls

## 🎨 Component Library

Reusable components for:
- KPI cards with trend indicators
- Line, bar, area, and scatter charts
- Heatmaps and distribution plots
- Data tables with sorting/filtering
- Modal dialogs
- Side/bottom drawers
- Progress bars
- Loading states
- Alert banners
- Badge components

## 🔐 Security Note

This is a prototype for demonstration purposes. In production:
- Implement real authentication
- Secure API endpoints
- Encrypt sensitive data
- Add role-based access control
- Validate all user inputs
- Use environment variables for configuration

---

Built with ❤️ for enterprise demand forecasting