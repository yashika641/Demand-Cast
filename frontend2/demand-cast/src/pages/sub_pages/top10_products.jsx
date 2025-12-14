import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

export default function Top10ProductsChart() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const apiUrl = "http://127.0.0.1:8000/product-dashboard";

  useEffect(() => {
    const fetchTop10 = async () => {
      try {
        const token = sessionStorage.getItem("firebaseIdToken");
        if (!token) throw new Error("Unauthorized");

        const res = await axios.get(apiUrl, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.data.status !== "success") throw new Error(res.data.message);
        setData(res.data.top_10_products);
      } catch (e) {
        setErr(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchTop10();
  }, []);

  if (loading) return <p className="text-gray-400">Loading...</p>;
  if (err) return <p className="text-red-400">{err}</p>;

  const productNames = data.map((d) => d.product);
  const salesValues = data.map((d) => d.total_sales);

  return (
    <Plot
      data={[
        {
          x: salesValues,
          y: productNames,
          type: "bar",
          orientation: "h", // horizontal bar chart
          marker: {
            color: salesValues.map(
              (v, i) =>
                `rgba(${50 + i * 10}, ${180 - i * 8}, ${255 - i * 12}, 0.8)`
            ),
          },
          hovertemplate:
            "<b>%{y}</b><br>Sales: ₹%{x:,.0f}<extra></extra>", // rich tooltip
        },
      ]}
      layout={{
        title: "",
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#cbd5e1" },
        margin: { l: 150, r: 30, t: 20, b: 40 },
        xaxis: {
          title: "Total Sales (₹)",
          gridcolor: "rgba(255,255,255,0.1)",
          zeroline: false,
        },
        yaxis: {
          automargin: true,
          tickfont: { size: 12 },
        },
        hovermode: "closest",
        showlegend: false,
      }}
      config={{
        responsive: true,
        displayModeBar: true, // shows zoom/download toolbar
        modeBarButtonsToRemove: [
          "lasso2d",
          "select2d",
          "autoScale2d",
          "toggleSpikelines",
        ],
      }}
      style={{ width: "100%", height: "500px" }}
      useResizeHandler
    />
  );
}
