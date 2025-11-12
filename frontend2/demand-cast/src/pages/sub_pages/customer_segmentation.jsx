import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchCustomerData } from "../utils/api";

export default function CustomerSegmentation() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCustomerData("segmentation", setData, setError, setLoading);
  }, []);

  if (loading) return <p className="loading">Loading segmentation...</p>;
  if (error) return <p className="error">{error}</p>;

  const segData = data?.segment_summary || [];

  return (
    <div className="tab-section">
      <h2>Customer Segmentation & Personas</h2>
      {segData.length > 0 ? (
        <Plot
          data={[{
            type: "bar",
            x: segData.map(d => d.segment),
            y: segData.map(d => d.CLV),
            marker: { color: ["#4b5fff", "#b24cff", "#7f5aff"] }
          }]}
          layout={{
            title: "Average CLV by Segment",
            paper_bgcolor: "transparent",
            font: { color: "#fff" }
          }}
          className="plot"
        />
      ) : <p>No segmentation data available.</p>}
    </div>
  );
}
