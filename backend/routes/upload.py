from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from supabase import create_client, Client
from datetime import datetime
from backend.routes.auth import verify_firebase_user  # Assumes your Firebase Auth utility is here

SUPABASE_URL = "https://waryjyqdedzdrwhxzare.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndhcnlqeXFkZWR6ZHJ3aHh6YXJlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTk5NDI5MSwiZXhwIjoyMDc3NTcwMjkxfQ.5M4RLa6o-Ii1MAXLdyUUhOYFQmUHAZEVE0xiM2SxkOc"  # Keep this secret in backend only!
SUPABASE_BUCKET = "uploads"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
router = APIRouter(tags=["upload"])

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user = Depends(verify_firebase_user)  # Returns decoded Firebase user info
):
    try:
        content = await file.read()
        filename = file.filename
        uid = user["uid"]

        # Upload file to Supabase Storage
        result = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, content)
        if hasattr(result, "error") and result.error:
            raise HTTPException(status_code=400, detail=result.error.message)

        # Get public URL
        base_url = "https://waryjyqdedzdrwhxzare.supabase.co/storage/v1/object/public"
        public_url = f"{base_url}/{SUPABASE_BUCKET}/{filename}"
        # Insert metadata row in files table
        metadata = {
            "filename": filename,
            "url": public_url,
            "user_id": uid,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        insert_result = supabase.table("files").insert(metadata).execute()
        if hasattr(result, "error") and result.error:
            raise HTTPException(status_code=400, detail=result.error.message)
        return {"status": "success", "url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/delete-files")
async def delete_user_files(user = Depends(verify_firebase_user)):
    try:
        uid = user["uid"]
        
        # Step 1: Get all file records for this user from database
        result = supabase.table("files").select("*").eq("user_id", uid).execute()
        
        if not result.data:
            return {"status": "success", "message": "No files to delete"}
        
        files_to_delete = result.data
        
        # Step 2: Delete each file from Supabase Storage
        for file_record in files_to_delete:
            filename = file_record["filename"]
            try:
                supabase.storage.from_(SUPABASE_BUCKET).remove([filename])
            except Exception as e:
                print(f"Failed to delete {filename} from storage: {e}")
        
        # Step 3: Delete all file metadata records from database
        delete_result = supabase.table("files").delete().eq("user_id", uid).execute()
        
        return {"status": "success", "message": f"Deleted {len(files_to_delete)} files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

