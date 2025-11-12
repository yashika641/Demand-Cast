import React, { useEffect, useState } from "react";
import { fetchCustomerData } from "../utils/api";

export default function CustomerChurn() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCustomerData("churn", setData, setError, setLoading);
  }, []);

  if (loading) return <p className="loading">Analyzing churn...</p>;
  if (error) return <p className="error">{error}</p>;

  return (
    <div className="tab-section">
      <h2>Churn & Retention</h2>
      <div className="metrics-row">
        <div className="metric-card">
          <p>Total Customers</p>
          <h3>{data.total_customers}</h3>
        </div>
        <div className="metric-card">
          <p>Churned (Inactive 6+ months)</p>
          <h3>{data.churned_customers}</h3>
        </div>
        <div className="metric-card">
          <p>Retention Rate</p>
          <h3>{data.retention_rate}%</h3>
        </div>
      </div>
    </div>
  );
}
