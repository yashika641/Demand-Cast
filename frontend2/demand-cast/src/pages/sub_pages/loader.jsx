// frontend2/src/components/Loader.jsx
import React from "react";
import { motion } from "framer-motion";

export default function Loader({ text }) {
  return (
    <div className="flex flex-col justify-center items-center h-[80vh]">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        className="border-4 border-indigo-500 border-t-transparent w-12 h-12 rounded-full"
      />
      <p className="mt-4 text-indigo-600 text-lg font-medium">{text}</p>
    </div>
  );
}
