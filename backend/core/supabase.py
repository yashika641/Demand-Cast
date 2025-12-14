from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_user_files(uid: str):
    """
    Get all uploaded file URLs for the user from Supabase 'uploads' bucket.
    Expected keys: customers.csv, transactions.csv, activity_logs.csv, support_tickets.csv
    """
    result = supabase.storage.from_("uploads").list(path=uid)
    files = {f["filename"].split(".")[0]: supabase.storage.from_("uploads").get_public_url(f"{uid}/{f['filename']}") for f in result}
    return files
