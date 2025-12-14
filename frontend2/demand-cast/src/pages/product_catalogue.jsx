import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { X, Sparkles } from "lucide-react";
import { Swiper, SwiperSlide } from "swiper/react";
import { Navigation, Autoplay } from "swiper/modules";
import "swiper/css";
import "swiper/css/navigation";

export default function ProductDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [modalData, setModalData] = useState(null);
  const [modalProduct, setModalProduct] = useState(null);
  const navigate = useNavigate();
  const apiUrl = "http://127.0.0.1:8000/product-dashboard";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = sessionStorage.getItem("firebaseIdToken");
        if (!token) throw new Error("Unauthorized");
        const res = await axios.get(apiUrl, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.data.status !== "success") throw new Error(res.data.message || "Failed");
        setData(res.data);
      } catch (e) {
        setErr(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleCardClick = (product) => {
    const key = product.name || product.product;
    const metadata = data?.product_metadata?.[key] || {};
    setModalData(metadata);
    setModalProduct(product);
  };

  const closeModal = () => {
    setModalData(null);
    setModalProduct(null);
  };

  const formatINR = (v) =>
    typeof v === "number" ? `₹${v.toLocaleString("en-IN")}` : "—";

  return (
    <div className="w-380 -ml-30 min-h-screen bg-linear-to-b from-[#030914] to-[#010410] text-gray-200  flex flex-col items-center font-sans">
      <div className="w-full max-w-7xl bg-[#0a1225]/80 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-cyan-800/40">
        {/* === HEADER === */}
        <div className="flex justify-between items-center mb-10">
          <div>
            <h1 className="text-4xl font-extrabold text-emerald-400 tracking-tight flex items-center gap-2">
              <Sparkles className="w-7 h-7 text-cyan-400" /> PRODUCT DASHBOARD
            </h1>
            <p className="text-gray-400 mt-2 text-lg">
              Strategize. Optimize. Accelerate.
            </p>
          </div>
          <button
            onClick={() => navigate("/services")}
            className="bg-linear-to-r from-cyan-500 to-emerald-500 px-5 py-2 rounded-xl text-white hover:shadow-lg transition"
          >
            Back to Hub
          </button>
        </div>

        {loading && <p className="text-center text-gray-400 py-6">Loading...</p>}
        {err && <p className="text-red-400 text-center">{err}</p>}

        {!loading && data && (
          <>
            {/* === KPI STRIP === */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <KPI label="Total Revenue" value={formatINR(data.meta?.total_revenue ?? 0)} />
              <KPI label="Products" value={data.meta?.total_products ?? 0} />
              <KPI label="Turnover" value={`${data.meta?.turnover ?? "—"}×`} />
              <KPI label="Inventory Value" value={formatINR(data.meta?.inventory_value ?? 0)} />
            </div>

            {/* === SWIPER CAROUSEL (AUTO + CLICKABLE) === */}
            <div className="mb-12">
              <div className="flex justify-between items-center mb-4 px-2">
                <h2 className="text-xl font-semibold text-emerald-400">
                  Featured Products
                </h2>
                <p className="text-sm text-gray-400 italic">Auto-scroll & swipe</p>
              </div>
              <Swiper
                modules={[Navigation, Autoplay]}
                spaceBetween={25}
                slidesPerView={4}
                loop={true}
                navigation
                autoplay={{
                  delay: 2500,
                  disableOnInteraction: false,
                }}
                breakpoints={{
                  320: { slidesPerView: 1 },
                  640: { slidesPerView: 2 },
                  1024: { slidesPerView: 4 },
                  1280: { slidesPerView: 5 },
                }}
              >
                {data.products && data.products.length > 0 ? (
                  data.products.map((p, idx) => {
                    const key = p.name || p.product || `Product ${idx + 1}`;
                    const revenue = data.product_metadata?.[key]?.Revenue;
                    const cardValue =
                      typeof revenue === "number"
                        ? `${formatINR(revenue)}`
                        : p.price
                        ? `${formatINR(Number(p.price))}`
                        : "View Details";
                    return (
                      <SwiperSlide key={idx}>
                        <MetricCard
                          title={key}
                          subtitle={p.category || "Category"}
                          value={cardValue}
                          image={p.image}
                          onClick={() => handleCardClick(p)}
                        />
                      </SwiperSlide>
                    );
                  })
                ) : (
                  <p className="text-gray-400 text-center">No products found.</p>
                )}
              </Swiper>
            </div>

            {/* === CHARTS === */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
              <ChartCard title="Top 5 Products by Revenue">
                <Plot
                  data={[
                    {
                      x: (data.top5_products || []).map(
                        (p) => p.product || p.Product || "Unknown"
                      ),
                      y: (data.top5_products || []).map(
                        (p) => p.sales || p.revenue || 0
                      ),
                      type: "bar",
                      marker: { color: "rgba(34,197,94,0.8)" },
                    },
                  ]}
                  layout={{
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    font: { color: "#cbd5e1" },
                    margin: { t: 30, b: 50, l: 50, r: 30 },
                  }}
                  style={{ width: "100%", height: "400px" }}
                  useResizeHandler
                />
              </ChartCard>

              <ChartCard title="Category Distribution">
                <Plot
                  data={[
                    {
                      labels: (data.category_distribution || []).map(
                        (d) => d.category || d.Category || "Unknown"
                      ),
                      values: (data.category_distribution || []).map(
                        (d) => d.sales || d.revenue || 0
                      ),
                      type: "pie",
                      hole: 0.4,
                      textinfo: "percent+label",
                      marker: { colors: ["#34d399","#22d3ee","#a78bfa","#facc15","#f43f5e"] },
                    },
                  ]}
                  layout={{
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    font: { color: "#cbd5e1" },
                  }}
                  style={{ width: "100%", height: "400px" }}
                  useResizeHandler
                />
              </ChartCard>
            </div>

            {/* === NEW: DEEP ANALYTICS (appended, non-destructive) === */}
            <h2 className="text-2xl font-bold text-emerald-400 mt-10 mb-6">Deep Analytics</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
              <AnalyticsChart title="Growth Trends (Monthly)">
                <Plot
                  data={(() => {
                    const trends = data.analytics?.growth_trends || [];
                    // group by product
                    const map = {};
                    trends.forEach((d) => {
                      const prod = d.product || d[data.detected_columns?.sales?.product] || "Unknown";
                      if (!map[prod]) map[prod] = { x: [], y: [], type: "scatter", mode: "lines+markers", name: prod };
                      map[prod].x.push(d.month);
                      map[prod].y.push(d.sales || 0);
                    });
                    return Object.values(map);
                  })()}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                  useResizeHandler
                />
              </AnalyticsChart>

              <AnalyticsChart title="Profitability (Top 10)">
                <Plot
                  data={[
                    {
                      x: (data.analytics?.profitability || []).map((d) => d.product || d[data.detected_columns?.sales?.product] || "Unknown"),
                      y: (data.analytics?.profitability || []).map((d) => d.profit || 0),
                      type: "bar",
                      marker: { color: "rgba(16,185,129,0.8)" },
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                  useResizeHandler
                />
              </AnalyticsChart>

              <AnalyticsChart title="ABC Analysis (Share of Sales)">
                <Plot
                  data={[
                    {
                      labels: (data.analytics?.abc_analysis || []).map((d) => d.abc_category || "Unknown"),
                      values: (data.analytics?.abc_analysis || []).map((d) => d.sales || d[data.detected_columns?.sales?.sales] || 0),
                      type: "pie",
                      textinfo: "percent+label",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Lifecycle Trend (Slope per Product)">
                <Plot
                  data={[
                    {
                      x: (data.analytics?.lifecycle || []).map((d) => d.product || "Unknown"),
                      y: (data.analytics?.lifecycle || []).map((d) => d.slope || 0),
                      type: "bar",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Inventory Risk (Days of Supply)">
                <Plot
                  data={[
                    {
                      x: (data.analytics?.inventory_risk || []).map((d) => d.product || "Unknown"),
                      y: (data.analytics?.inventory_risk || []).map((d) => d.days_of_supply || 0),
                      type: "bar",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Price Sensitivity (Avg Price vs Total Sales)">
                <Plot
                  data={[
                    {
                      x: (data.analytics?.price_sensitivity || []).map((d) => d.avg_price || 0),
                      y: (data.analytics?.price_sensitivity || []).map((d) => d.total_sales || 0),
                      mode: "markers",
                      type: "scatter",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" }, xaxis:{title:"Avg Price"}, yaxis:{title:"Total Sales"} }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Customer Engagement (Unique Buyers)">
                <Plot
                  data={[
                    {
                      x: (data.analytics?.customer_engagement || []).map((d) => d.product || "Unknown"),
                      y: (data.analytics?.customer_engagement || []).map((d) => d.unique_buyers || 0),
                      type: "bar",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Regional Bestsellers">
                <Plot
                  data={[
                    {
                      x: (data.analytics?.regional_bestsellers || []).map((d) => d.region || "Unknown"),
                      y: (data.analytics?.regional_bestsellers || []).map((d) => d.sales || 0),
                      type: "bar",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Product Correlation (Heatmap)">
                <Plot
                  data={[
                    {
                      z: (() => {
                        const corr = data.analytics?.product_correlation || {};
                        const rows = Object.keys(corr);
                        if (!rows.length) return [];
                        return rows.map((r) => Object.values(corr[r] || {}));
                      })(),
                      type: "heatmap",
                      colorscale: "Viridis",
                    },
                  ]}
                  layout={{ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#cbd5e1" } }}
                  style={{ width: "100%", height: "380px" }}
                />
              </AnalyticsChart>

              <AnalyticsChart title="Cross-Selling Rules (Top 10)">
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm text-left">
                    <thead>
                      <tr className="text-cyan-400 border-b border-cyan-800/30">
                        <th className="py-2 pr-4">Antecedent</th>
                        <th className="py-2 pr-4">Consequent</th>
                        <th className="py-2 pr-4">Support</th>
                        <th className="py-2 pr-4">Confidence</th>
                        <th className="py-2 pr-4">Lift</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(data.analytics?.cross_selling || []).map((r, i) => (
                        <tr key={i} className="border-b border-cyan-800/10">
                          <td className="py-2 pr-4">{r.antecedent}</td>
                          <td className="py-2 pr-4">{r.consequent}</td>
                          <td className="py-2 pr-4">{(r.support ?? 0).toFixed(2)}</td>
                          <td className="py-2 pr-4">{(r.confidence ?? 0).toFixed(2)}</td>
                          <td className="py-2 pr-4">{(r.lift ?? 0).toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </AnalyticsChart>
            </div>
          </>
        )}
      </div>

      {/* === PRODUCT MODAL (on click) === */}
      <AnimatePresence>
        {modalData && modalProduct && (
          <motion.div
            className="fixed inset-0 bg-black/60 flex justify-center items-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-[#0a162f] border border-cyan-700/40 p-8 rounded-2xl shadow-2xl w-[95%] max-w-5xl relative backdrop-blur-xl grid grid-cols-1 md:grid-cols-3 gap-6"
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
            >
              <button
                onClick={closeModal}
                className="absolute top-3 right-3 text-gray-400 hover:text-emerald-400"
              >
                <X />
              </button>

              {/* LEFT: Product Name + Image + Description */}
              <div className="flex flex-col items-center text-center md:text-left md:items-start space-y-3">
                <h2 className="text-2xl font-bold text-emerald-400">
                  {modalProduct.name}
                </h2>
                <img
                  src={modalProduct.image}
                  alt={modalProduct.name}
                  className="w-56 h-56 object-contain rounded-xl border border-cyan-800/40 shadow-lg"
                />
                <p className="text-gray-300 text-sm leading-relaxed">
                  {modalProduct.description || "No description available"}
                </p>
              </div>

              {/* RIGHT: Price, Category, Metadata */}
              <div className="md:col-span-2 bg-[#0e1b3a]/60 rounded-xl p-6 border border-cyan-800/30 shadow-lg">
                <div className="flex justify-between items-center mb-4">
                  <p className="text-lg font-semibold text-gray-400">
                    Category:{" "}
                    <span className="text-cyan-400">
                      {modalProduct.category || "N/A"}
                    </span>
                  </p>
                  <p className="text-2xl font-bold text-emerald-400">
                    {modalProduct.price
                      ? formatINR(Number(modalProduct.price))
                      : "—"}
                  </p>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {Object.entries(modalData).map(([key, val]) => (
                    <div
                      key={key}
                      className="bg-[#102349]/80 p-3 rounded-lg border border-cyan-800/20"
                    >
                      <p className="text-gray-400 text-xs">{key}</p>
                      <p className="text-cyan-300 font-semibold text-sm mt-1">
                        {typeof val === "number" ? formatINR(val) : String(val)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* 🔹 KPI Card */
const KPI = ({ label, value }) => (
  <div className="bg-[#0d1936]/80 rounded-xl p-4 border border-cyan-800/40 shadow">
    <p className="text-xs text-gray-400">{label}</p>
    <p className="text-xl font-bold text-emerald-400 mt-1">{value}</p>
  </div>
);

/* 🔹 Metric Card (Swiper Slide Item) */
const MetricCard = ({ title, subtitle, value, image, onClick }) => (
  <div
    onClick={onClick}
    className="cursor-pointer bg-[#0b142e] rounded-2xl p-5 shadow-lg hover:scale-105 hover:bg-[#101c3c]/90 transition transform border border-cyan-800/30 text-center"
  >
    <img
      src={image}
      alt={title}
      className="w-20 h-20 object-contain mx-auto mb-3"
    />
    <h3 className="text-md font-semibold text-gray-200 truncate">{title}</h3>
    <p className="text-xs text-gray-400">{subtitle}</p>
    <p className="text-sm font-bold text-emerald-400 mt-2">{value}</p>
  </div>
);

/* 🔹 Chart Card */
const ChartCard = ({ title, children }) => (
  <div className="bg-[#0d1936]/80 rounded-2xl p-6 shadow-lg border border-cyan-800/40">
    <h2 className="text-xl font-semibold text-emerald-400 mb-4">{title}</h2>
    {children}
  </div>
);

/* 🔹 Analytics Card */
const AnalyticsChart = ({ title, children }) => (
  <div className="bg-[#0b1530]/70 rounded-2xl p-6 shadow-lg border border-cyan-800/40">
    <h2 className="text-lg font-semibold text-emerald-400 mb-4">{title}</h2>
    {children}
  </div>
);
