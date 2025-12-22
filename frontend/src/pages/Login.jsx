import { useState } from 'react';
import { BarChart3, Mail, Lock, ArrowRight } from 'lucide-react';
import { FcGoogle } from "react-icons/fc";
import { FaGithub } from "react-icons/fa";
import { useNavigate } from 'react-router-dom';
import { supabase } from '../supabaseClient';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  const handleEmailLogin = async (e) => {
  e.preventDefault();

  try {
    const res = await fetch("http://localhost:8000/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email,
        password
      })
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "Login failed");
      return;
    }

    // Save token locally
    sessionStorage.setItem("token", data.access_token);
    console.log("Stored token:", data.access_token);
    // Tell parent login succeeded
    onLogin(data.user);
    navigate('/');
  } catch (error) {
    console.error(error);
    alert("Something went wrong");
  }
};

  const handleGoogleLogin = async () => {
    supabase.auth.signInWithOAuth({
  provider: "google",
  options: {
    redirectTo: "http://localhost:5173/auth/callback",
  },
});

  };

  const handleGithubLogin = async () => {
    supabase.auth.signInWithOAuth({
  provider: "github",
  options: {
    redirectTo: "http://localhost:5173/auth/callback",
  },
});

  };

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4 relative overflow-hidden">

      {/* Animated circles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400/10 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-400/10 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />
      </div>

      <div className="w-full max-w-md relative z-10">

        {/* Logo + heading */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-linear-to-br from-blue-500 to-blue-600 rounded-2xl mb-4 shadow-xl shadow-blue-500/30">
            <BarChart3 className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-gray-900 text-3xl font-semibold mb-2">DemandCast</h1>
          <p className="text-gray-600">AI-Powered Demand Forecasting Platform</p>
        </div>

        {/* Card */}
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-100/50 p-8">
          <h2 className="text-gray-900 text-2xl font-semibold mb-6">Sign in to your account</h2>

          {/* Email Login */}
          <form onSubmit={handleEmailLogin} className="space-y-4">
            <div>
              <label className="block text-gray-700 text-sm mb-2 font-medium">Email address</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all bg-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-gray-700 text-sm mb-2 font-medium">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all bg-white"
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full bg-linear-to-r from-blue-500 to-blue-600 text-white py-3 rounded-xl hover:shadow-lg hover:shadow-blue-500/30 transition-all flex items-center justify-center gap-2 font-medium hover:-translate-y-0.5"
            >
              Sign in
              <ArrowRight className="w-5 h-5" />
            </button>
          </form>

          {/* Divider */}
          <div className="my-6 flex items-center">
            <div className="flex-1 h-px bg-gray-200"></div>
            <span className="px-4 text-gray-500 text-sm">or</span>
            <div className="flex-1 h-px bg-gray-200"></div>
          </div>

          {/* OAuth Buttons */}
          <div className="space-y-3">
            <button
              onClick={handleGoogleLogin}
              className="w-full flex items-center justify-center gap-3 border border-gray-200 py-3 rounded-xl hover:bg-gray-50 transition-all font-medium"
            >
              <FcGoogle className="w-6 h-6" />
              Continue with Google
            </button>

            <button
              onClick={handleGithubLogin}
              className="w-full flex items-center justify-center gap-3 border border-gray-200 py-3 rounded-xl hover:bg-gray-50 transition-all font-medium"
            >
              <FaGithub className="w-6 h-6" />
              Continue with GitHub
            </button>
          </div>

          {/* Signup Link */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Don’t have an account?{' '}
              <button
                onClick={() => navigate("/signup")}
                className="text-blue-600 hover:text-blue-700 font-medium transition-colors"
              >
                Sign up
              </button>
            </p>
          </div>
        </div>

        <p className="text-center text-sm text-gray-500 mt-8">
          Enterprise-grade demand forecasting and inventory intelligence
        </p>
      </div>
    </div>
  );
}
