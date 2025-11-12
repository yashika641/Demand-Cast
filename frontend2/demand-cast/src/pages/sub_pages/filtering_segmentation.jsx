import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function FilteringSegmentation() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [errorDetails, setErrorDetails] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const navigate = useNavigate();
  const apiUrl = "http://127.0.0.1:8000/filtering-segmentation";

  const fetchData = async (start = "", end = "") => {
    try {
      setLoading(true);
      setErrorMsg("");
      setErrorDetails("");
      setData(null);

      // 🔐 Retrieve Firebase ID token from sessionStorage
      const token = sessionStorage.getItem("firebaseIdToken");
      if (!token) {
        throw new Error("No authorization token found.");
      }

      // 🔗 API Request with Authorization header
      const response = await axios.get(apiUrl, {
        headers: { Authorization: `Bearer ${token}` },
        params: {
          start_date: start || undefined,
          end_date: end || undefined,
        },
      });

      const resp = response.data;
      console.log("Full API Response:", resp);

      if (resp.status !== "success") {
        throw new Error(resp.message || "Failed to fetch segmentation data.");
      }

      setData(resp);
    } catch (error) {
      console.error("Error fetching segmentation data:", error);
      setErrorMsg("Failed to fetch segmentation data.");
      setErrorDetails(error?.message || "Unknown error occurred.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 py-10 px-6 flex flex-col items-center">
      {/* 🔹 Header */}
      <div className="w-full max-w-7xl bg-gray-900 rounded-2xl shadow-2xl p-8">
        <div className="flex flex-row justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-amber-400">
            🔍 Filter & Segment Data
          </h1>
          <button
            onClick={() => navigate("/services")}
            className="bg-amber-700 hover:bg-amber-800 text-white px-4 py-2 rounded-lg transition"
          >
            Back to Hub
          </button>
        </div>

        {/* 🔸 Date Filters */}
        <div className="flex flex-wrap justify-center gap-4 mb-10">
          <div className="flex flex-col">
            <label className="text-gray-300 text-sm mb-1">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="border border-gray-600 bg-gray-800 rounded-md px-3 py-2 text-gray-100 focus:ring-2 focus:ring-amber-500 outline-none"
            />
          </div>
          <div className="flex flex-col">
            <label className="text-gray-300 text-sm mb-1">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="border border-gray-600 bg-gray-800 rounded-md px-3 py-2 text-gray-100 focus:ring-2 focus:ring-amber-500 outline-none"
            />
          </div>
          <button
            onClick={() => fetchData(startDate, endDate)}
            className="bg-amber-600 hover:bg-amber-700 text-white font-semibold px-6 py-2 rounded-lg self-end transition"
          >
            Apply Filter
          </button>
        </div>

        {/* 🧠 Content */}
        {loading && (
          <p className="text-center text-gray-400 py-10">Loading data...</p>
        )}

        {errorMsg && (
          <div className="bg-red-800 bg-opacity-20 border border-red-500 rounded-xl p-4 mb-6">
            <p className="text-red-400 font-semibold">{errorMsg}</p>
            <pre className="text-gray-300 text-sm mt-2 whitespace-pre-wrap">
              {errorDetails}
            </pre>
          </div>
        )}

        {!loading && data && data.status === "success" && (
          <>
            {/* 📈 Sales Trend */}
            <div className="bg-gray-800 rounded-xl shadow-lg p-4 mb-8">
              <h2 className="text-xl font-semibold mb-3 text-center text-amber-400">
                📈 Sales Over Time
              </h2>
              <Plot
                data={[
                  {
                    x: data.line_chart.x,
                    y: data.line_chart.y,
                    mode: "lines+markers",
                    type: "scatter",
                    line: { color: "#fbbf24", width: 3 },
                    marker: { color: "#f59e0b", size: 6 },
                  },
                ]}
                layout={{
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  font: { color: "#f3f4f6" },
                  margin: { t: 30, b: 40, l: 50, r: 30 },
                }}
                useResizeHandler
                style={{ width: "100%", height: "400px" }}
              />
            </div>

            {/* 🧩 Side-by-Side Pie Charts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Product Share */}
              <div className="bg-gray-800 rounded-xl shadow-lg p-4">
                <h2 className="text-lg font-semibold mb-3 text-center text-amber-400">
                  📊 Top 50% Product Sales Share
                </h2>
                <Plot
                  data={[
                    {
                      labels: data.product_chart.map(
                        (d) => d[data.meta.product_col]
                      ),
                      values: data.product_chart.map(
                        (d) => d[data.meta.sales_col]
                      ),
                      type: "pie",
                      hole: 0.4,
                      textinfo: "percent+label",
                      marker: { colors: ["#f59e0b", "#84cc16", "#22d3ee", "#a78bfa"] },
                    },
                  ]}
                  layout={{
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    font: { color: "#f3f4f6" },
                  }}
                  useResizeHandler
                  style={{ width: "100%", height: "400px" }}
                />
              </div>

              {/* Category Share */}
              <div className="bg-gray-800 rounded-xl shadow-lg p-4">
                <h2 className="text-lg font-semibold mb-3 text-center text-amber-400">
                  📊 Top 50% Category Sales Share
                </h2>
                <Plot
                  data={[
                    {
                      labels: data.category_chart.map(
                        (d) => d[data.meta.category_col]
                      ),
                      values: data.category_chart.map(
                        (d) => d[data.meta.sales_col]
                      ),
                      type: "pie",
                      hole: 0.4,
                      textinfo: "percent+label",
                      marker: { colors: ["#10b981", "#6366f1", "#f97316", "#f43f5e"] },
                    },
                  ]}
                  layout={{
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    font: { color: "#f3f4f6" },
                  }}
                  useResizeHandler
                  style={{ width: "100%", height: "400px" }}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
