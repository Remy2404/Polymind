"""
Cross-check tools.json models with model_configs.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def cross_check_models():
    # Load tools.json
    with open('tools.json', 'r') as f:
        tools_models = json.load(f)
    
    try:
        from src.services.model_handlers.model_configs import ModelConfigurations
        
        all_models = ModelConfigurations.get_all_models()
        
        print(f"Cross-checking {len(tools_models)} tested tool-calling models:")
        print("=" * 70)
        
        found_count = 0
        missing_count = 0
        
        for tool in tools_models:
            openrouter_id = tool['id']
            name = tool['name']
            
            # Check if this OpenRouter ID exists in our configs
            found = False
            found_model_id = None
            
            for model_id, config in all_models.items():
                if (hasattr(config, 'openrouter_model_key') and 
                    config.openrouter_model_key == openrouter_id):
                    found = True
                    found_model_id = model_id
                    break
            
            if found:
                print(f"✅ FOUND: {found_model_id}")
                print(f"   OpenRouter: {openrouter_id}")
                found_count += 1
            else:
                print(f"❌ MISSING: {openrouter_id}")
                print(f"   Name: {name}")
                missing_count += 1
            print()
        
        print("=" * 70)
        print(f"Summary: {found_count} found, {missing_count} missing")
        
        if missing_count == 0:
            print("🎉 All tested tool-calling models are already configured!")
        else:
            print(f"⚠️  Need to add {missing_count} missing models")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    cross_check_models()
