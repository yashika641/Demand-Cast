import React, { useState } from "react";
import { signIn, signUp } from "./authutils";

function AuthForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="flex flex-col space-y-4">
      <input className="border" value={email} onChange={e => setEmail(e.target.value)} placeholder="Email" />
      <input className="border" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" />
      <button className="bg-blue-500 text-white" onClick={() => signIn(email, password)}>Sign In</button>
      <button className="bg-green-500 text-white" onClick={() => signUp(email, password)}>Sign Up</button>
    </div>
  );
}

export default AuthForm;
