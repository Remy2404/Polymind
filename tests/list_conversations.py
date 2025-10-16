"""List all conversations to see what's left"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_database
import hashlib

db, _ = get_database()
conversations = db.conversations

# Get all conversations for this user
user_convs = list(conversations.find({'cache_key': {'$regex': '^user_806762900_model_'}}))
print(f"Total conversations: {len(user_convs)}")
print()

# Show first 10 with hashes
print("Sample conversations:")
for i, conv in enumerate(user_convs[:10]):
    cache_key = conv['cache_key']
    hash_val = hashlib.md5(cache_key.encode()).hexdigest()
    msg_count = len(conv.get('messages', []))
    print(f"{i+1}. Hash: {hash_val}")
    print(f"   Key: {cache_key}")
    print(f"   Messages: {msg_count}")
    print()
