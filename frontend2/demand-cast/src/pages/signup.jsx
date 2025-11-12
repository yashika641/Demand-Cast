import React, { useState } from "react";
import { signUp } from "../authutils";
import { auth } from "../firebase";  // Make sure to export/getAuth properly
import { useNavigate } from "react-router-dom";

function SignupForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSignup = async () => {
    try {
      await signUp(email, password);
      const auth = getAuth();                 // Get Firebase Auth instance
      const user = auth.currentUser;          // Get current user after signup
      if (user) {
        const idToken = await user.getIdToken();  // Generate JWT token
        sessionStorage.setItem("firebaseIdToken", idToken);  // Save JWT in sessionStorage
      }
      alert("User signed up!");
      navigate("/Uploader");
    } catch (err) {
      alert("Signup failed: " + err.message);
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      <input
        className="border"
        value={email}
        onChange={e => setEmail(e.target.value)}
        placeholder="Email"
      />
      <input
        className="border"
        type="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        placeholder="Password"
      />
      <button className="bg-green-500 text-white" onClick={handleSignup}>
        Sign Up
      </button>
    </div>
  );
}

export default SignupForm;
