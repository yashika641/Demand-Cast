import "./sales.css"
import React, { useState } from "react";
import power from "../assets/power.png"
import bg_image from "../assets/bg_image2.png"
import { useNavigate, useLocation } from "react-router-dom";
import QuarterlyPerformance from "./sub_pages/performance";
import FilteringSegmentation from "./sub_pages/filtering_segmentation";
import GoogleTrends from "./sub_pages/google_api";
import Forecasts from "./sub_pages/forecasts_sales";
function TabContent({ pathname }) {
    if (pathname === "/services/sales-trend") return <div>Sales Trend content here</div>;
    if (pathname === "/services/filtering") return <div>Filtering and Segmentation content here</div>;
    if (pathname === "/services/revenue") return <div>Revenue Analysis content here</div>;
    if (pathname === "/services/google-trends") return <div>Google Trends content here</div>;
    if (pathname === "/services/forecast") return <div>Forecast & Predictions content here</div>;
    // Default tab (if needed)
    return <div>Select a tab above.</div>;
}

export default function Sales() {
    // Mapping titles to paths
    const tabList = [
        { label: "SALES TREND" },
        { label: "FILTERING AND SEGMENTATION" },
        { label: "REVENUE ANALYSIS" },
        { label: "GOOGLE TRENDS" },
        { label: "FORECAST & PREDICTIONS" },
    ];
    const [activeTab, setActiveTab] = useState(0);
    return (
        <div className="min-h-screen w-360 flex flex-col items-center justify-center px-4 font-sans overflow-x-hidden relative -ml-20">
            <img src={bg_image} alt="Background" className="fixed inset-0 object-cover w-full -mt-100 -z-10" />
            <nav className="w-full flex justify-between items-center py-4 px-8 backdrop-blur-2xl bg-opacity-80 shadow-md">
                <h1 className="font-xl font-bold font-white
            ">demandcast</h1>
                <div className="flex flex-row gap-10">
                    <button className="pill-tab">Dashboard</button>
                    <button className="pill-tab"><img src={power} alt="switch" className="h-4 w-4" /></button>
                </div>
            </nav>
            <main className="w-full flex justify-center items-center py-4 px-8 backdrop-blur-2xl bg-opacity-80 shadow-md">
                <div className="flex items-start flex-col hero-section  w-2/5  bg-opacity-80 rounded-lg shadow-lg text-center">
                    <p className=" left-0 hero-tittle font-extrabold text-5xl text-left text-[#ed8865]  font-serif text-shadow: 0 6px 30px rgba(255, 106, 77, 0.12)">SALES & REVENUE DASHBOARD</p>
                    <p className="text-[#8b5d79] text-2xl text-left font-serif">unlock Growth. Visualize Performance.</p>
                    <p className="text-[#8b5d79] text-2xl text-left font-serif">Forecast the Future</p>
                </div>
                <div className="flex flex-row divider my-8  border-gray-300 w-3/5 gap-3 p-2">
                    <div className="flex flex-col w-1/2  border-gray-300 border-2 rounded-2xl backdrop-blur-2xl">
                        <p className="text-xl text-white font-semibold" >TOTAL SALES</p>
                        <p className="text-3xl text-shadow: 0 6px 30px rgba(255, 106, 77, 0.12) text-[#ed8865] font-bold">7.2M</p>
                        <p>UNITS</p>
                        <p className="text-xl text-white font-semibold">TOTAL REVENUE</p>
                        <p className="text-3xl text-shadow: 0 6px 30px rgba(255, 106, 77, 0.12) text-[#ed8865] font-bold">$14.5M</p>
                        <p>USD</p>
                    </div>
                    <div className="flex flex-col w-1/2 border-gray-300 border-2 rounded-2xl backdrop-blur-2xl">
                        <p className="text-xl text-white font-semibold">AVERAGE ORDER VALUE </p>
                        <p >$185.30</p>
                        <p className="text-3xl text-shadow: 0 6px 30px rgba(255, 106, 77, 0.12) text-[#ed8865] font-bold">$185.30</p>
                        <p>USD</p>
                        <p className="text-0.5xl text-white font-semibold">HIGHEST PERFORMING PRODUCTS</p>
                        <p className="text-3xl text-shadow: 0 6px 30px rgba(255, 106, 77, 0.12) text-[#ed8865] font-bold">SHAMPOO</p>
                    </div>
                </div>
            </main>
            <div className="w-full">
                {/* Tab Buttons */}
                <div className="flex gap-7 mb-6 w-full justify-center items-center py-4 px-8 backdrop-blur-2xl bg-opacity-80 shadow-md">
                    {tabList.map((tab, idx) => (
                        <button
                            key={tab.label}
                            className={`px-6 py-2 rounded-full text-pink-400 border border-pink-500 
  shadow-[0_0_15px_2px_rgba(236,72,153,0.6)] 
  hover:shadow-[0_0_25px_4px_rgba(236,72,153,0.9)] 
  hover:text-white transition-all duration-300 
  bg-transparent
          ${activeTab === idx
                                    ? "bg-linear-to-r from-pink-700 to-blue-700 text-white shadow-[0_0_20px_rgba(236,72,153,0.6)]"
                                    : "bg-linear-to-r from-blue-950 to-pink-950 text-gray-300 hover:from-pink-800 hover:to-blue-800 hover:text-white hover:shadow-[0_0_15px_rgba(147,197,253,0.4)]"
                                }`}
                            onClick={() => setActiveTab(idx)}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Tab Content */}
                <div>
                    {activeTab === 0 && <div><QuarterlyPerformance/></div>}
                    {activeTab === 1 && <div><FilteringSegmentation/></div>}
                    {activeTab === 2 && <div>Revenue Analysis Content here</div>}
                    {activeTab === 3 && <div><GoogleTrends/></div>}
                    {activeTab === 4 && <div><Forecasts/></div>}
                </div>
            </div>
            {/* <div className="flex flex-col items-center justify-center rounded-2xl border-amber-700 border-2 shadow-[0_0_20px_3px_rgba(217,119,6,0.6)] w-full mt-10 mb-10 gap-4">
                <div className="flex flex-row justify-between items-center w-full">
                    <p className="ml-5 text-shadow-white mt-5 p-4  text-4xl ">Quaterly Performance</p>
                    <div className="flex flex-row items-center justify-center">
                        

                        <button>yearly</button>
                        <button>monthly</button>
                        <button>weekly</button>
                        <button>daily</button>
                        <button>back to hub!</button>
                    </div>
                </div>
            </div> */}
        </div>
    )
}