// frontend2/src/pages/CustomerDashboard.jsx
import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import Plot from "react-plotly.js";
import StatCard from "./sub_pages/stat_card";
import Loader from "./sub_pages/loader";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

export default function CustomerDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = sessionStorage.getItem("firebaseIdToken");
        const res = await axios.get(`${BACKEND_URL}/customer-dashboard`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setData(res.data);
      } catch (err) {
        console.error("Error fetching data:", err);
        setError("Failed to load customer analytics data.",err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <Loader text="Fetching Customer Insights..." />;
  if (error) return <p className="text-red-500 text-center mt-8">{error}</p>;

  const demographics = data?.demographics || {};
  const clv = data?.clv_summary || {};
  const churn = data?.churn_summary || {};
  const segmentation = data?.segmentation || {};
  const loyalty = data?.loyalty_summary || {};

  return (
    <motion.div
      className="min-h-screen bg-linear-to-br from-indigo-50 to-purple-50 p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.7 }}
    >
      <h1 className="text-4xl font-bold text-center mb-8 text-indigo-700">
        👥 Customer Analytics Dashboard
      </h1>

      {/* ======== STAT CARDS ======== */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <StatCard title="Total Customers" value={churn.total_customers} />
        <StatCard title="Retention Rate" value={`${churn.retention_rate}%`} />
        <StatCard title="Churned Customers" value={churn.churned_customers} />
        <StatCard title="Avg CLV" value={`₹${clv.distribution?.mean?.toFixed(2)}`} />
      </div>

      {/* ======== DEMOGRAPHICS ======== */}
      <motion.div
        className="bg-white p-6 rounded-2xl shadow-md mb-10"
        whileHover={{ scale: 1.01 }}
        transition={{ type: "spring", stiffness: 200 }}
      >
        <h2 className="text-2xl font-semibold mb-4 text-indigo-600">🧍 Demographics Overview</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {demographics.gender_distribution && (
            <Plot
              data={[
                {
                  labels: Object.keys(demographics.gender_distribution),
                  values: Object.values(demographics.gender_distribution),
                  type: "pie",
                  marker: { colors: ["#4b5fff", "#b24cff"] },
                },
              ]}
              layout={{
                title: "Gender Split",
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              }}
              style={{ width: "100%", height: "300px" }}
            />
          )}

          {demographics.region_distribution && (
            <Plot
              data={[
                {
                  x: Object.keys(demographics.region_distribution),
                  y: Object.values(demographics.region_distribution),
                  type: "bar",
                  marker: { color: "#7f5aff" },
                },
              ]}
              layout={{
                title: "Customers by Region",
                paper_bgcolor: "rgba(0,0,0,0)",
              }}
              style={{ width: "100%", height: "300px" }}
            />
          )}

          {demographics.membership_tiers && (
            <Plot
              data={[
                {
                  labels: Object.keys(demographics.membership_tiers),
                  values: Object.values(demographics.membership_tiers),
                  type: "pie",
                  marker: { colors: ["#4b5fff", "#b24cff", "#7f5aff"] },
                },
              ]}
              layout={{
                title: "Membership Tiers",
                paper_bgcolor: "rgba(0,0,0,0)",
              }}
              style={{ width: "100%", height: "300px" }}
            />
          )}
        </div>
      </motion.div>

      {/* ======== CLV ANALYSIS ======== */}
      <motion.div className="bg-white p-6 rounded-2xl shadow-md mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-indigo-600">💰 Customer Lifetime Value</h2>

        <Plot
          data={[
            {
              x: clv.top_customers?.map((c) => c.customer_id),
              y: clv.top_customers?.map((c) => c.CLV),
              type: "bar",
              marker: { color: "#4b5fff" },
            },
          ]}
          layout={{
            title: "Top 10 Customers by CLV",
            paper_bgcolor: "rgba(0,0,0,0)",
          }}
          style={{ width: "100%", height: "400px" }}
        />
      </motion.div>

      {/* ======== SEGMENTATION ======== */}
      <motion.div className="bg-white p-6 rounded-2xl shadow-md mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-indigo-600">🧩 Customer Segmentation</h2>
        <Plot
          data={[
            {
              labels: Object.keys(segmentation.segments || {}),
              values: Object.values(segmentation.segments || {}),
              type: "pie",
              marker: { colors: ["#4b5fff", "#b24cff", "#7f5aff"] },
            },
          ]}
          layout={{
            title: "Customer Segments (CLV-based)",
            paper_bgcolor: "rgba(0,0,0,0)",
          }}
          style={{ width: "100%", height: "400px" }}
        />
      </motion.div>

      {/* ======== LOYALTY LEADERBOARD ======== */}
      <motion.div className="bg-white p-6 rounded-2xl shadow-md mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-indigo-600">🏆 Loyalty Leaderboard</h2>
        <table className="w-full text-left border-collapse">
          <thead className="bg-indigo-100">
            <tr>
              <th className="p-3">Customer ID</th>
              <th className="p-3">Loyalty Points</th>
            </tr>
          </thead>
          <tbody>
            {loyalty.top_loyal_customers?.map((c, idx) => (
              <tr
                key={idx}
                className="border-b hover:bg-indigo-50 transition-all"
              >
                <td className="p-3">{c.customer_id}</td>
                <td className="p-3">{c.loyalty_points}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </motion.div>
  );
}
