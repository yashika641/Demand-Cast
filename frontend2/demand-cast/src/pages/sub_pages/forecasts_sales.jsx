import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function Forecasts() {
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [horizon, setHorizon] = useState(30);
  const [unit, setUnit] = useState("days"); // or 'months'

  const navigate = useNavigate();
  const apiUrl = "http://127.0.0.1:8000/forecasts";

  const fetchForecast = async () => {
    try {
      setLoading(true);
      setErr("");
      setResp(null);

      const token = sessionStorage.getItem("firebaseIdToken");
      if (!token) throw new Error("No auth token found.");

      const { data } = await axios.get(apiUrl, {
        headers: { Authorization: `Bearer ${token}` },
        params: { horizon, unit },
      });

      if (data.status !== "success") {
        throw new Error(data.message || "Forecast failed.");
      }
      console.log("Forecast API:", JSON.stringify(data, null, 2));
      setResp(data);
    } catch (e) {
      setErr(e?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchForecast();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ChartCard = ({ title, children }) => (
    <div className="bg-gray-800 rounded-xl shadow-lg p-5 mb-8">
      <h2 className="text-lg font-semibold mb-3 text-center text-amber-400">{title}</h2>
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 py-10 px-6 flex flex-col items-center">
      <div className="w-full max-w-7xl bg-gray-900 rounded-2xl shadow-2xl p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-amber-400">🔮 Sales Forecasts</h1>
          <button
            onClick={() => navigate("/services")}
            className="bg-amber-700 hover:bg-amber-800 text-white px-4 py-2 rounded-lg transition"
          >
            Back to Hub
          </button>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-4">
            <label className="block text-sm text-gray-300 mb-2">
              Horizon: <span className="font-semibold">{horizon} {unit}</span>
            </label>
            <input
              type="range"
              min={unit === "days" ? 7 : 1}
              max={unit === "days" ? 180 : 24}
              value={horizon}
              onChange={(e) => setHorizon(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="bg-gray-800 rounded-xl p-4">
            <label className="block text-sm text-gray-300 mb-2">Unit</label>
            <select
              value={unit}
              onChange={(e) => {
                setUnit(e.target.value);
                // reset horizon to a sensible default when unit changes
                setHorizon(e.target.value === "days" ? 30 : 6);
              }}
              className="w-full bg-gray-700 rounded px-3 py-2"
            >
              <option value="days">Days</option>
              <option value="months">Months</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={fetchForecast}
              className="w-full bg-amber-600 hover:bg-amber-700 text-white font-semibold px-6 py-3 rounded-lg transition"
            >
              Run Forecast
            </button>
          </div>
        </div>

        {/* Status */}
        {loading && <p className="text-center text-gray-400 py-10">Computing forecast…</p>}
        {err && (
          <div className="bg-red-800/20 border border-red-500 rounded-xl p-4 mb-6">
            <p className="text-red-400 font-semibold">Error</p>
            <pre className="text-gray-300 text-sm mt-2 whitespace-pre-wrap">{err}</pre>
          </div>
        )}

        {/* Charts */}
        {!loading && resp && resp.status === "success" && (
          <>
            <ChartCard title={`📈 ${resp.meta.freq === "monthly" ? "Monthly" : "Daily"} Forecast (${resp.meta.horizon} ${resp.meta.unit})`}>
              <Plot
                data={[
                  {
                    x: resp.plot.history.x,
                    y: resp.plot.history.y,
                    type: "scatter",
                    mode: "lines",
                    name: "History",
                  },
                  {
                    x: resp.plot.forecast.prophet.x,
                    y: resp.plot.forecast.prophet.y,
                    type: "scatter",
                    mode: "lines",
                    name: "Prophet",
                  },
                  {
                    x: resp.plot.forecast.hybrid.x,
                    y: resp.plot.forecast.hybrid.y,
                    type: "scatter",
                    mode: "lines",
                    name: "Hybrid",
                  },
                  ...(resp.plot.forecast.ci?.x?.length
                    ? [
                        {
                          x: [
                            ...resp.plot.forecast.ci.x,
                            ...[...resp.plot.forecast.ci.x].reverse(),
                          ],
                          y: [
                            ...resp.plot.forecast.ci.upper,
                            ...[...resp.plot.forecast.ci.lower].reverse(),
                          ],
                          fill: "toself",
                          fillcolor: "rgba(59,130,246,0.15)",
                          line: { width: 0 },
                          name: "Prophet CI",
                          type: "scatter",
                          hoverinfo: "skip",
                          showlegend: true,
                        },
                      ]
                    : []),
                ]}
                layout={{
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  font: { color: "#f3f4f6" },
                  margin: { t: 30, b: 40, l: 50, r: 30 },
                  xaxis: { title: "Date" },
                  yaxis: { title: "Sales" },
                  legend: { orientation: "h" },
                }}
                config={{ responsive: true, displaylogo: false }}
                useResizeHandler
                style={{ width: "100%", height: "460px" }}
              />
            </ChartCard>

            {/* Table preview */}
            <div className="bg-gray-900 text-gray-200 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-2 text-amber-400">🔍 Forecast Preview (last 10)</h3>
              <pre className="text-xs overflow-x-auto">{JSON.stringify(resp.table, null, 2)}</pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
