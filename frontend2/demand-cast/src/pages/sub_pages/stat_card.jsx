// frontend2/src/components/StatCard.jsx
import React from "react";
import { motion } from "framer-motion";

export default function StatCard({ title, value }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-white shadow-md rounded-2xl p-5 flex flex-col justify-center items-center"
    >
      <p className="text-gray-500 font-semibold">{title}</p>
      <h2 className="text-2xl font-bold text-indigo-600 mt-2">{value ?? "-"}</h2>
    </motion.div>
  );
}
