import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchCustomerData } from "../utils/api";

export default function CustomerCLV() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCustomerData("clv", setData, setError, setLoading);
  }, []);

  if (loading) return <p className="loading">Loading CLV metrics...</p>;
  if (error) return <p className="error">{error}</p>;

  const topCustomers = data?.top_customers || [];

  return (
    <div className="tab-section">
      <h2>Customer Lifetime Value (CLV)</h2>
      {topCustomers.length > 0 ? (
        <Plot
          data={[{
            type: "bar",
            x: topCustomers.map(d => d.customer_id),
            y: topCustomers.map(d => d.CLV),
            marker: { color: "#b24cff" }
          }]}
          layout={{
            title: "Top 10 Customers by CLV",
            paper_bgcolor: "transparent",
            font: { color: "#fff" }
          }}
          className="plot"
        />
      ) : <p>No CLV data available.</p>}
    </div>
  );
}
