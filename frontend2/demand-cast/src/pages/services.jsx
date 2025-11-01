// src/pages/Services.jsx
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { BarChart2, Home, Tag, Users, Megaphone, Plus } from "lucide-react";
import bg_image from '../assets/bg_image1.png'
export default function Services() {
    const cards = [
        {
            title: "SALES ANALYTICS",
            info: "Q1 Revenue +12% YOY",
            icon: <BarChart2 size={40} />,
            color: "neon-orange",
        },
        {
            title: "INVENTORY OPTIMIZATION",
            info: "Stock Utilization 88%",
            icon: <Home size={40} />,
            color: "neon-blue",
        },
        {
            title: "CUSTOMER INSIGHTS",
            info: "CRM Score: A-",
            icon: <Users size={40} />,
            color: "neon-purple",
        },
        {
            title: "PRODUCT LIFECYCLE",
            info: "2 New Lines Launched",
            icon: <Tag size={40} />,
            color: "neon-green",
        },
        {
            title: "CAMPAIGN STRATEGIST",
            info: "Last Campaign ROI: 4.5x",
            icon: <Megaphone size={40} />,
            color: "neon-yellow",
        },
        {
            title: "ADD NEW SERVICE",
            info: "",
            icon: <Plus size={40} />,
            color: "neon-gray",
        },
    ];

    return (
        <div className="min-h-screen flex flex-col items-center justify-center px-4">
            <img src={bg_image} alt="Background" className="fixed inset-0 object-cover w-full -mt-100 -z-10" />
            {/* Navbar */}
            <div className="flex justify-between w-full max-w-6xl mb-10">
                <h1 className="text-2xl font-bold text-white tracking-wide">DemandCast</h1>
                <div className="flex gap-8 text-gray-300">
                    <span className="hover:text-white cursor-pointer">Dashboard</span>
                    <span className="hover:text-white cursor-pointer">Uploader</span>
                    <span className="hover:text-white cursor-pointer">Settings</span>
                    <span className="text-cyan-400 font-semibold">Services</span>
                </div>
            </div>

            {/* Title Section */}
            <div className="text-center max-w-2xl mb-14">
                <h2 className="text-3xl font-bold mb-3 tracking-wide">
                    OUR INTEGRATED SERVICES
                </h2>
                <p className="text-gray-400">
                    Access specialized tools and insights across your Sales, Product, Customer, and Campaign operations.
                </p>
            </div>

            {/* Cards Section */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                {cards.map((card, i) => (
                    <Link to={card.path} key={i}>
                        <motion.div
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.98 }}
                            className={`glow-card ${card.color} p-6 flex flex-col items-start justify-between min-w-[250px] min-h-[150px] rounded-xl cursor-pointer transition-all duration-300`}
                        >
                            <div className="text-white mb-4 text-3xl">{card.icon}</div>
                            <h3 className="text-lg font-semibold text-white">{card.title}</h3>
                            <p className="text-gray-300 text-sm mt-2">{card.info}</p>
                        </motion.div>
                    </Link>
                ))}
            </div>
            
        </div>
    );
}
