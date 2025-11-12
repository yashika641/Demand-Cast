// src/components/LoginForm.jsx
import React, { useState } from "react";
import { signIn } from "../firebase";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function LoginForm() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate(); // ✅ parentheses added

    const handleLogin = async () => {
        try {
            setLoading(true);

            // 1️⃣ Log in using Firebase
            const user = await signIn(email, password);

            // 2️⃣ Get Firebase ID token
            const idToken = await user.getIdToken();
            sessionStorage.setItem("firebaseIdToken", idToken);
            // 3️⃣ Send token to backend for verification
            const res = await axios.post(
                "http://127.0.0.1:8000/login",
                {},
                { headers: { Authorization: `Bearer ${idToken}` } }
            );

            alert(`✅ Logged in as ${res.data.email}`);
            console.log("Server verified:", res.data);

            // 4️⃣ Navigate to uploader after successful login
            navigate("/Uploader");
        } catch (err) {
            console.error(err);
            alert("❌ Login failed: " + (err.response?.data?.detail || err.message));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col space-y-4 w-80 mx-auto mt-10 p-6 border rounded-xl shadow-md bg-white">
            <h2 className="text-xl font-semibold text-center text-gray-800">Login</h2>

            <input
                className="border p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email"
            />

            <input
                className="border p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
            />

            <button
                className="bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 transition disabled:opacity-50"
                onClick={handleLogin}
                disabled={loading}
            >
                {loading ? "Signing In..." : "Sign In"}
            </button>
            <button className="bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 transition disabled:opacity-50">
                <a href="/signup">Don't have an account? Sign Up</a>
            </button>
        </div>
    );
}

export default LoginForm;
