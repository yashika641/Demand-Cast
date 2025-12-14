import React, { useEffect, useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";
import { useNavigate } from "react-router-dom";

export default function QuarterlyPerformance() {
  const [plotData, setPlotData] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [errorDetails, setErrorDetails] = useState("");
  const [aggFreq, setAggFreq] = useState("Quarterly");

  const navigate = useNavigate();
  const apiUrl = "http://127.0.0.1:8000/user-sales-analytics";

  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true);
      setErrorMsg("");
      setErrorDetails("");
      setPlotData(null);
      setStats(null);

      try {
        const token = sessionStorage.getItem("firebaseIdToken");
        if (!token) {
          setErrorMsg("No authorization token found.");
          return;
        }

        const response = await axios.get(`${apiUrl}?frequency=${aggFreq.toLowerCase()}`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        const resp = response.data;
        console.log("Full API Response:", resp);

        if (resp.status !== "success" || !resp.files?.length) {
          throw new Error("Unexpected API format or empty response.");
        }

        const file = resp.files[0];
        const plot = file.plot || {};
        if (!plot.labels || !plot.values) {
          throw new Error("Plot data missing or malformed.");
        }

        // Prepare Plotly data
        const chartData = [
          {
            x: plot.labels,
            y: plot.values,
            type: "scatter",
            mode: "lines+markers",
            name: plot.title || "Sales Performance",
            line: { color: "#38bdf8", width: 3 },
            marker: { color: "#38bdf8", size: 6 },
            hovertemplate: "₹%{y}<br>%{x}<extra></extra>",
          },
        ];

        const chartLayout = {
          title: {
            text: plot.title || "Sales Performance",
            font: { size: 18, color: "#fff" },
          },
          paper_bgcolor: "#0f172a",
          plot_bgcolor: "#0f172a",
          font: { color: "#f8fafc" },
          xaxis: {
            title: { text: plot.xlabel || "Time", font: { color: "#f8fafc" } },
            gridcolor: "#1f2937",
          },
          yaxis: {
            title: { text: plot.ylabel || "Value", font: { color: "#f8fafc" } },
            gridcolor: "#1f2937",
            zeroline: false,
          },
          margin: { l: 60, r: 30, t: 60, b: 50 },
        };

        setPlotData({ chartData, chartLayout });
        setStats(file.stats || null);
      } catch (error) {
        console.error("Error fetching analytics:", error);
        setErrorMsg("Failed to fetch sales analytics.");
        setErrorDetails(error?.message || "Unknown error occurred.");
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, [apiUrl, aggFreq]);

  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border-amber-700 border-2 shadow-[0_0_20px_3px_rgba(217,119,6,0.6)] w-full mt-10 mb-10 gap-4 p-4">
      {/* Header Section */}
      <div className="flex flex-row justify-between items-center w-full px-4">
        <p className="text-3xl font-bold mt-3 text-white">
          Sales Performance ({aggFreq})
        </p>
        <div className="flex flex-row items-center justify-center gap-3">
          {["Yearly", "Quarterly", "Monthly", "Weekly", "Daily"].map((freq) => (
            <button
              key={freq}
              onClick={() => setAggFreq(freq)}
              className={`px-3 py-1 rounded-md border ${aggFreq === freq
                  ? "bg-amber-600 text-white"
                  : "bg-gray-700 text-gray-200 hover:bg-gray-600"
                } transition`}
            >
              {freq}
            </button>
          ))}
          <button
            onClick={() => navigate("/services")}
            className="ml-4 px-4 py-1 rounded-md bg-amber-700 text-white hover:bg-amber-800 transition"
          >
            Back to Hub
          </button>
        </div>
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="w-full bg-gray-800 text-gray-100 rounded-xl p-4 mt-2 flex flex-row justify-around text-center shadow-md">
          <div>
            <p className="text-lg font-semibold">Total Sales</p>
            <p className="text-xl font-bold text-green-400">
              ₹ {Number(stats.total_sales).toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-lg font-semibold">Avg Order Value</p>
            <p className="text-xl font-bold text-blue-400">
              ₹ {Number(stats.avg_order_value).toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-lg font-semibold">Total Orders</p>
            <p className="text-xl font-bold text-yellow-400">
              {stats.row_count}
            </p>
          </div>
        </div>
      )}

      {/* Chart or Loading/Error */}
      <div className="w-full px-6" style={{ minHeight: 400 }}>
        {loading && <p className="text-gray-400">Loading chart...</p>}

        {errorMsg && (
          <div className="text-red-500 bg-gray-900 p-4 rounded-xl mt-4">
            <p className="font-semibold">{errorMsg}</p>
            <pre
              className="text-sm overflow-x-auto text-gray-300"
              style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
            >
              {typeof errorDetails === "object"
                ? JSON.stringify(errorDetails, null, 2)
                : errorDetails}
            </pre>
          </div>
        )}

        {!loading && !errorMsg && plotData && (
          <div className="h-[400px] mt-4">
            <Plot
              data={plotData.chartData}
              layout={plotData.chartLayout}
              config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["lasso2d", "select2d"],
              }}
              style={{ width: "100%", height: "100%" }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
