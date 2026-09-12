import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('.env')
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SECRET_KEY'))
res = supabase.table('messages').select('*').order('created_at', desc=True).limit(10).execute()
if res.data:
    print('\n'.join(f"{r.get('speaker', 'Unknown')}: {r.get('message', '')}" for r in reversed(res.data)))
else:
    print("No messages found.")
