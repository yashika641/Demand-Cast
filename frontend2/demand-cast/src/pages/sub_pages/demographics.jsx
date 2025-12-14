import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchCustomerData } from "../utils/api";

export default function CustomerDemographics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCustomerData("demographics", setData, setError, setLoading);
  }, []);

  if (loading) return <p className="loading">Loading demographics...</p>;
  if (error) return <p className="error">{error}</p>;

  const genderData = data?.data?.gender || [];
  const ageData = data?.data?.age_distribution || [];
  const regionData = data?.data?.regions || [];

  return (
    <div className="tab-section">
      <h2>Customer Demographics</h2>
      <div className="charts-grid">
        {genderData.length > 0 && (
          <Plot
            data={[{ type: "pie", labels: genderData.map(d => d.gender), values: genderData.map(d => d.count), marker: { colors: ["#4b5fff", "#b24cff"] } }]}
            layout={{ title: "Gender Distribution", paper_bgcolor: "transparent", font: { color: "#fff" } }}
            className="plot"
          />
        )}
        {ageData.length > 0 && (
          <Plot
            data={[{ type: "bar", x: ageData.map(d => d.age_group), y: ageData.map(d => d.count), marker: { color: "#7f5aff" } }]}
            layout={{ title: "Age Groups", paper_bgcolor: "transparent", font: { color: "#fff" } }}
            className="plot"
          />
        )}
        {regionData.length > 0 && (
          <Plot
            data={[{ type: "bar", x: regionData.map(d => d.region), y: regionData.map(d => d.count), marker: { color: "#b24cff" } }]}
            layout={{ title: "Regions", paper_bgcolor: "transparent", font: { color: "#fff" } }}
            className="plot"
          />
        )}
      </div>
    </div>
  );
}
