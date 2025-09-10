
"""
Script to identify new models from tools.json that aren't in model_configs.py
"""

import json
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def extract_existing_models():
    """Extract existing model IDs from model_configs.py"""
    config_path = "src/services/model_handlers/model_configs.py"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all model_id patterns
    model_id_pattern = r'model_id="([^"]*)"'
    matches = re.findall(model_id_pattern, content)
    
    return set(matches)

def extract_tools_models():
    """Extract model IDs from tools.json"""
    tools_path = "tools.json"
    
    with open(tools_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)
    
    models = []
    for tool in tools:
        model_id = tool.get("id", "")
        name = tool.get("name", "")
        description = tool.get("description", "")
        context_length = tool.get("context_length", 32000)
        supported_params = tool.get("supported_parameters", [])
        
        # Determine if it supports tool calls
        supports_tools = any(param in ["tool_choice", "tools"] for param in supported_params)
        
        models.append({
            "id": model_id,
            "name": name, 
            "description": description,
            "context_length": context_length,
            "supports_tools": supports_tools,
            "supported_params": supported_params
        })
    
    return models

def create_model_id_from_openrouter_id(openrouter_id):
    """Create a simplified model_id from OpenRouter ID"""
    # Remove provider prefix and :free suffix
    model_id = openrouter_id.replace(":free", "")
    
    # Extract the model name part
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]
    
    # Replace common patterns
    replacements = {
        "-instruct": "",
        "-chat": "",
        "-v3.1": "-v3-1", 
        "-v3.2": "-v3-2",
        "24b-instruct-2501": "24b-2501",
    }
    
    for old, new in replacements.items():
        model_id = model_id.replace(old, new)
    
    return model_id

def main():
    print("🔍 Analyzing models in tools.json vs model_configs.py")
    print("=" * 60)
    
    # Get existing models
    existing_models = extract_existing_models()
    print(f"📄 Found {len(existing_models)} existing models in model_configs.py")
    
    # Get tools models
    tools_models = extract_tools_models()
    print(f"🔧 Found {len(tools_models)} models in tools.json")
    
    print("\n🆕 New models to add:")
    print("-" * 30)
    
    new_models = []
    
    for model in tools_models:
        openrouter_id = model["id"]
        suggested_id = create_model_id_from_openrouter_id(openrouter_id)
        
        # Check various possible IDs
        possible_ids = [
            suggested_id,
            openrouter_id.split("/")[-1],
            openrouter_id.split("/")[-1].replace(":free", ""),
            openrouter_id.replace(":", "-").replace("/", "-")
        ]
        
        # Check if any variation exists
        found = any(pid in existing_models for pid in possible_ids)
        
        if not found:
            new_models.append({
                "suggested_id": suggested_id,
                "openrouter_id": openrouter_id,
                "name": model["name"],
                "description": model["description"],
                "supports_tools": model["supports_tools"],
                "context_length": model["context_length"]
            })
            
            print(f"✅ {suggested_id}")
            print(f"   OpenRouter ID: {openrouter_id}")
            print(f"   Name: {model['name']}")
            print(f"   Tool Support: {'Yes' if model['supports_tools'] else 'No'}")
            print()
    
    print(f"\n📊 Summary: {len(new_models)} new models identified")
    
    if new_models:
        print("\n🚀 Ready to add these models to model_configs.py")
        return new_models
    else:
        print("\n✨ All models from tools.json are already in model_configs.py!")
        return []

if __name__ == "__main__":
    new_models = main()
    
    # Store results for use by other scripts
    with open("new_models_to_add.json", "w") as f:
        json.dump(new_models, f, indent=2)
    
    print(f"\n💾 Results saved to new_models_to_add.json")
