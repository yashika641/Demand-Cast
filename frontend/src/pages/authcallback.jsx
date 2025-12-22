import { useEffect } from "react";
import { supabase } from "../supabaseClient";
import { useNavigate } from "react-router-dom";

export default function AuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    async function finish() {
      // Get oauth session directly from supabase
      const { data } = await supabase.auth.getSession();

      if (data?.session) {
        sessionStorage.setItem("token", data.session.access_token);
        console.log("Stored token:", data.session.access_token);
        navigate("/");
      } else {
        console.error("No session found");
      }
    }

    finish();
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <p>Finishing login...</p>
    </div>
  );
}
