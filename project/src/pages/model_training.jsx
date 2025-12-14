import React, { useState } from "react";

export default function ModelTrainingOrchestration() {
  const [settings, setSettings] = useState({
    demand: { horizon: 30, model: "xgboost", seasonality: "auto" },
    leadtime: { smoothing: 7, variability: false },
    stockout: { balance: "auto", model: "xgboost" },
    promo: { lookback: 180, discount: true },
    pricing: { method: "loglog", cross: true },
    rag: { embedding: "bge", dataset: "supply_docs" },
    // token: sessionStorage.getItem("token")
  });

  const [estimatedTime, setEstimatedTime] = useState(32);
  const [started, setStarted] = useState(false);
  // const token = sessionStorage.getItem("token");

  // ----------------------------------------------
  // 🔥 NEW: Start Training → Send POST to backend
  // ----------------------------------------------
  const startTraining = async () => {
    setStarted(true);

    const token = sessionStorage.getItem("token");

    try {
      const response = await fetch("http://localhost:8000/model_training/start/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ ...settings, token })
      });


      const data = await response.json();
      console.log("Training started:", data);

      if (data.estimated_time) {
        setEstimatedTime(data.estimated_time);
      }
    } catch (err) {
      console.error("Training API error:", err);
    }
  };

  // Update nested settings
  const updateSetting = (section, field, value) => {
    setSettings((prev) => ({
      ...prev,
      [section]: { ...prev[section], [field]: value }
    }));
  };

  return (
    <div className="min-h-screen p-10 bg-[#0d0f15] text-white space-y-10">
      <h1 className="text-3xl font-bold">Model Training Orchestration</h1>

      {/* Demand Forecast */}
      <Section title="Demand Forecast Model">
        <div className="grid grid-cols-3 gap-5">
          <Select
            label="Forecast Horizon"
            value={settings.demand.horizon}
            options={[7, 14, 30, 90]}
            onChange={(v) => updateSetting("demand", "horizon", Number(v))}
          />
          <Select
            label="Seasonality"
            value={settings.demand.seasonality}
            options={["auto", "manual"]}
            onChange={(v) => updateSetting("demand", "seasonality", v)}
          />
          <Select
            label="Model Type"
            value={settings.demand.model}
            options={["prophet", "xgboost", "lstm"]}
            onChange={(v) => updateSetting("demand", "model", v)}
          />
        </div>
      </Section>

      {/* Lead Time Model */}
      <Section title="Lead Time Model">
        <div className="grid grid-cols-2 gap-5">
          <Input
            label="Smoothing Window (days)"
            value={settings.leadtime.smoothing}
            onChange={(v) => updateSetting("leadtime", "smoothing", Number(v))}
          />
          <Toggle
            label="Supplier Variability Handling"
            value={settings.leadtime.variability}
            onChange={(v) => updateSetting("leadtime", "variability", v)}
          />
        </div>
      </Section>

      {/* Stockout Model */}
      <Section title="Stockout Model">
        <div className="grid grid-cols-2 gap-5">
          <Select
            label="Class Balancing"
            value={settings.stockout.balance}
            options={["auto", "manual"]}
            onChange={(v) => updateSetting("stockout", "balance", v)}
          />
          <Select
            label="Model Type"
            value={settings.stockout.model}
            options={["xgboost", "random_forest", "lstm"]}
            onChange={(v) => updateSetting("stockout", "model", v)}
          />
        </div>
      </Section>

      {/* Promotions */}
      <Section title="Promotional Effectiveness Model">
        <div className="grid grid-cols-2 gap-5">
          <Select
            label="Lookback Period"
            value={settings.promo.lookback}
            options={[90, 180, 365]}
            onChange={(v) => updateSetting("promo", "lookback", Number(v))}
          />
          <Toggle
            label="Discount Sensitivity"
            value={settings.promo.discount}
            onChange={(v) => updateSetting("promo", "discount", v)}
          />
        </div>
      </Section>

      {/* Price Elasticity */}
      <Section title="Price Elasticity Model">
        <div className="grid grid-cols-2 gap-5">
          <Select
            label="Elasticity Method"
            value={settings.pricing.method}
            options={["loglog", "regression"]}
            onChange={(v) => updateSetting("pricing", "method", v)}
          />
          <Toggle
            label="Cross Elasticity"
            value={settings.pricing.cross}
            onChange={(v) => updateSetting("pricing", "cross", v)}
          />
        </div>
      </Section>

      {/* RAG */}
      <Section title="RAG Model (AI Retrieval)">
        <div className="grid grid-cols-2 gap-5">
          <Select
            label="Embedding Model"
            value={settings.rag.embedding}
            options={["bge", "gte", "minilm"]}
            onChange={(v) => updateSetting("rag", "embedding", v)}
          />
          <Select
            label="Document Set"
            value={settings.rag.dataset}
            options={["supply_docs", "all_docs"]}
            onChange={(v) => updateSetting("rag", "dataset", v)}
          />
        </div>
      </Section>

      {/* Estimated Time */}
      <div className="p-6 bg-[#141821] rounded-xl border border-gray-700 flex justify-between items-center">
        <div>
          <p className="text-lg font-semibold">Estimated Training Time</p>
          <p className="text-3xl font-bold text-green-400">{estimatedTime} minutes</p>
        </div>
      </div>

      {/* Start Training Button */}
      {!started ? (
        <button
          onClick={startTraining}
          className="w-full py-5 text-xl font-semibold bg-blue-600 hover:bg-blue-700 rounded-xl transition"
        >
          🚀 Start Training
        </button>
      ) : (
        <div className="p-6 mt-5 bg-[#141821] rounded-xl border border-gray-700 text-center text-green-400 text-xl">
          Your models are now training... Check back in {estimatedTime} minutes for insights.
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------
// UI Components
// ----------------------------------------------

function Section({ title, children }) {
  return (
    <div className="p-6 bg-[#141821] rounded-xl border border-gray-700 space-y-5">
      <h2 className="text-xl font-semibold">{title}</h2>
      {children}
    </div>
  );
}

function Select({ label, value, options, onChange }) {
  return (
    <div className="space-y-2">
      <p className="text-sm text-gray-400">{label}</p>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-3 bg-[#1c1f27] border border-gray-700 rounded-lg"
      >
        {options.map((opt) => (
          <option key={opt}>{opt}</option>
        ))}
      </select>
    </div>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <div className="flex items-center gap-3">
      <p className="text-sm text-gray-400 grow">{label}</p>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
        className="scale-125"
      />
    </div>
  );
}

function Input({ label, value, onChange }) {
  return (
    <div className="space-y-2">
      <p className="text-sm text-gray-400">{label}</p>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-3 bg-[#1c1f27] border border-gray-700 rounded-lg"
      />
    </div>
  );
}
