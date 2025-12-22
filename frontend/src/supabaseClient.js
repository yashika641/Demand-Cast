import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = "https://waryjyqdedzdrwhxzare.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhcnlqeXFkZWR6ZHJ3aHh6YXJlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTk5NDI5MSwiZXhwIjoyMDc3NTcwMjkxfQ.5M4RLa6o-Ii1MAXLdyUUhOYFQmUHAZEVE0xiM2SxkOc";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
