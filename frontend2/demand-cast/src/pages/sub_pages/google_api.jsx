import React, { useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";
import { useNavigate } from "react-router-dom";

export default function GoogleTrends() {
  const [product, setProduct] = useState("");
  const [timeframe, setTimeframe] = useState("today 12-m");
  const [region, setRegion] = useState("");
  const [loading, setLoading] = useState(false);
  const [trendData, setTrendData] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();

  const apiUrl = "http://127.0.0.1:8000/google-trends";

  const fetchTrends = async () => {
    if (!product.trim()) {
      setErrorMsg("Please enter a product name.");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setTrendData([]);

    try {
      const response = await axios.get(apiUrl, {
        params: {
          product_name: product,
          timeframe: timeframe,
          geo: region || "",
        },
      });

      const data = response.data;
      console.log("Full API Response:", data);

      if (!data.trend_data || data.trend_data.length === 0) {
        throw new Error("No data found for this product.");
      }

      // Prepare Plotly chart
      const xData = data.trend_data.map((d) => d.date);
      const yData = data.trend_data.map((d) => d[product]);

      setTrendData([
        {
          x: xData,
          y: yData,
          type: "scatter",
          mode: "lines+markers",
          name: product,
          line: { color: "#38bdf8", width: 3 },
          marker: { color: "#fbbf24", size: 6 },
        },
      ]);
    } catch (error) {
      console.error("Error fetching trends:", error);
      setErrorMsg(
        error?.response?.data?.detail || error.message || "Failed to fetch data."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center py-10 px-6">
      {/* Header */}
      <div className="w-full max-w-6xl bg-gray-900 rounded-2xl shadow-xl p-6 mb-6 flex justify-between items-center">
        <h1 className="text-3xl font-bold text-amber-400">
          📈 Google Trends Analytics
        </h1>
        <button
          onClick={() => navigate("/services")}
          className="bg-amber-700 hover:bg-amber-800 text-white px-4 py-2 rounded-lg transition"
        >
          Back to Hub
        </button>
      </div>

      {/* Input Section */}
      <div className="w-full max-w-6xl bg-gray-800 rounded-xl p-6 mb-10 shadow-lg">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Product Input */}
          <div>
            <label className="block text-sm text-gray-300 mb-2">
              Product / Keyword
            </label>
            <input
              type="text"
              value={product}
              onChange={(e) => setProduct(e.target.value)}
              placeholder="e.g. iPhone, Netflix, AirPods"
              className="w-full px-3 py-2 rounded-md bg-gray-900 text-gray-100 border border-gray-700 focus:ring-2 focus:ring-amber-500 outline-none"
            />
          </div>

          {/* Timeframe Dropdown */}
          <div>
            <label className="block text-sm text-gray-300 mb-2">Timeframe</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="w-full px-3 py-2 rounded-md bg-gray-900 text-gray-100 border border-gray-700 focus:ring-2 focus:ring-amber-500 outline-none"
            >
              <option value="today 5-y">Last 5 Years</option>
              <option value="today 12-m">Last 12 Months</option>
              <option value="today 3-m">Last 3 Months</option>
              <option value="now 7-d">Last 7 Days</option>
              <option value="now 1-d">Last 1 Day</option>
            </select>
          </div>

          {/* Region Input */}
          <div>
            <label className="block text-sm text-gray-300 mb-2">
              Region (optional)
            </label>
            <input
              type="text"
              value={region}
              onChange={(e) => setRegion(e.target.value.toUpperCase())}
              placeholder="e.g. IN, US, GB"
              className="w-full px-3 py-2 rounded-md bg-gray-900 text-gray-100 border border-gray-700 focus:ring-2 focus:ring-amber-500 outline-none"
            />
          </div>
        </div>

        {/* Fetch Button */}
        <div className="flex justify-center mt-8">
          <button
            onClick={fetchTrends}
            disabled={loading}
            className={`px-6 py-2 rounded-lg font-semibold transition ${
              loading
                ? "bg-gray-600 cursor-not-allowed"
                : "bg-amber-600 hover:bg-amber-700 text-white"
            }`}
          >
            {loading ? "Fetching..." : "Fetch Trends"}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {errorMsg && (
        <div className="bg-red-900 bg-opacity-30 border border-red-500 text-red-300 p-4 rounded-xl w-full max-w-4xl text-center mb-6">
          <p className="font-semibold">{errorMsg}</p>
        </div>
      )}

      {/* Chart Display */}
      {trendData.length > 0 && (
        <div className="w-full max-w-6xl bg-gray-900 rounded-xl shadow-xl p-6">
          <h2 className="text-xl font-semibold text-center text-amber-400 mb-4">
            {`Google Search Interest for "${product}"`}
          </h2>

          <Plot
            data={trendData}
            layout={{
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              font: { color: "#f3f4f6" },
              margin: { l: 60, r: 30, t: 50, b: 50 },
              xaxis: {
                title: "Date",
                gridcolor: "#1f2937",
              },
              yaxis: {
                title: "Search Interest (0–100)",
                gridcolor: "#1f2937",
                range: [0, 100],
              },
            }}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
            }}
            style={{ width: "100%", height: "400px" }}
          />
        </div>
      )}
    </div>
  );
}
