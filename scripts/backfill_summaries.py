import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai

load_dotenv()

# Setup Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SECRET_KEY")
if not supabase_url or not supabase_key:
    print("Error: SUPABASE_URL or SUPABASE_SECRET_KEY is missing.")
    exit(1)

supabase: Client = create_client(supabase_url, supabase_key)

# Setup Google Gemini
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY is missing.")
    exit(1)

ai_client = genai.Client(api_key=google_api_key)

def summarize_transcript(transcript: str) -> str:
    prompt = (
        "You are an expert sales analyst. Read the following call transcript "
        "and provide a short, 1-2 sentence summary of the caller's main requirement, "
        "interest, or the outcome of the call. If the call is empty or too short, "
        "reply with 'Short call. Caller disconnected or did not provide requirements.'\n\n"
        f"Transcript:\n{transcript}"
    )
    
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Failed to analyze."

def main():
    print("Fetching calls with missing requirements...")
    
    # We fetch all calls. We can filter in memory.
    resp = supabase.table("calls").select("id, requirement").execute()
    calls = resp.data
    
    # Fetch all messages to build transcripts
    msg_resp = supabase.table("messages").select("call_id, speaker, message, created_at").order("created_at").execute()
    
    # Group messages by call_id
    messages_by_call = {}
    for msg in msg_resp.data:
        cid = str(msg["call_id"])
        if cid not in messages_by_call:
            messages_by_call[cid] = []
        messages_by_call[cid].append(msg)
        
    updated_count = 0
    for call in calls:
        # If requirement is already present and not just our fallback placeholder, skip it.
        # But wait, the fallback placeholder was injected in dashboard_sync, not saved to DB.
        req = call.get("requirement")
        if req and req.strip():
            continue
            
        call_id = str(call["id"])
        call_msgs = messages_by_call.get(call_id, [])
        
        if not call_msgs:
            new_req = "Short call. No transcript data found."
        else:
            # Build transcript
            transcript_lines = []
            for m in call_msgs:
                speaker = "Agent" if m["speaker"] == "agent" else "Caller"
                transcript_lines.append(f"{speaker}: {m['message']}")
            
            full_transcript = "\n".join(transcript_lines)
            
            print(f"\nAnalyzing call {call_id} ({len(call_msgs)} turns)...")
            new_req = summarize_transcript(full_transcript)
            time.sleep(1) # Simple rate limit protection
            
        print(f"Generated Summary: {new_req}")
        
        # Update Supabase
        supabase.table("calls").update({"requirement": new_req}).eq("id", call_id).execute()
        updated_count += 1
        print(f"Updated call {call_id} in Supabase.")
        
    print(f"\nDone! Successfully backfilled {updated_count} calls.")

if __name__ == "__main__":
    main()
