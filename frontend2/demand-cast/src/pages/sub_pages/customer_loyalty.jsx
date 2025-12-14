import React, { useEffect, useState } from "react";
import { fetchCustomerData } from "../utils/api";

export default function CustomerLoyalty() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchCustomerData("loyalty", setData, setError, setLoading);
  }, []);

  if (loading) return <p className="loading">Loading loyalty data...</p>;
  if (error) return <p className="error">{error}</p>;

  const topLoyal = data?.top_loyal_customers || [];

  return (
    <div className="tab-section">
      <h2>Loyalty & Retention</h2>
      <table className="loyalty-table">
        <thead>
          <tr><th>Rank</th><th>Loyalty Points</th></tr>
        </thead>
        <tbody>
          {topLoyal.map((row, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{Object.values(row)[0]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
