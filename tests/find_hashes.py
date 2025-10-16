"""Find cache_keys for specific hashes"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_database
import hashlib

target_hashes = [
    '110ad46424c00027f98d220b2cb7ef79',
    '4d3af7660ffa7696556e5f862aa5fd76'
]

db, _ = get_database()
conversations = db.conversations

print("Searching for matching cache_keys...")
print()

# Get all conversations for this user
user_convs = list(conversations.find({'cache_key': {'$regex': '^user_806762900_model_'}}))
print(f"Found {len(user_convs)} total conversations")
print()

# Build hash map
hash_map = {}
for conv in user_convs:
    cache_key = conv['cache_key']
    hash_val = hashlib.md5(cache_key.encode()).hexdigest()
    hash_map[hash_val] = {
        'cache_key': cache_key,
        'message_count': len(conv.get('messages', []))
    }

# Check target hashes
for target in target_hashes:
    if target in hash_map:
        info = hash_map[target]
        print(f"✅ FOUND: {target}")
        print(f"   Cache key: {info['cache_key']}")
        print(f"   Messages: {info['message_count']}")
    else:
        print(f"❌ NOT FOUND: {target}")
    print()
