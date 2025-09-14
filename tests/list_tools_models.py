import json

with open("tools.json", "r") as f:
    tools = json.load(f)

print(f"Tools.json contains {len(tools)} tested tool-calling models:")
print("=" * 60)

for tool in tools:
    print(f"• {tool['id']}")
    print(f"  Name: {tool['name']}")
    print(f"  Context: {tool['context_length']:,}")
    print()
