import requests
import json
import sys

def fetch_and_filter_models():
    """Fetch and filter free models from OpenRouter API."""
    # API endpoint
    url = "https://openrouter.ai/api/v1/models"

    try:
        print("Fetching model data from OpenRouter...")
        # Fetch model data
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        models = response.json().get("data", [])
        print(f"Fetched {len(models)} models from API")

        # Filter for free models and extract supported parameters
        free_models = []
        for model in models:
            if model.get("pricing", {}).get("prompt") == "0":
                model_data = {
                    "name": model.get("name", "Unknown Model"),
                    "id": model.get("id", ""),
                    "description": model.get("description", ""),
                    "supported_parameters": model.get("supported_parameters", []),
                }
                # Only add models with valid IDs
                if model_data["id"]:
                    free_models.append(model_data)

        print(f"Found {len(free_models)} free models")

        # Save to JSON file
        with open("models.json", "w", encoding="utf-8") as f:
            json.dump(free_models, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(free_models)} models to models.json")

        # Print summary
        tool_models = 0
        reasoning_models = 0
        
        for m in free_models:
            supported_params = m.get("supported_parameters", [])
            
            has_tools = any(param in supported_params for param in ["tools", "tool_choice", "function_calling"])
            has_reasoning = any(param in supported_params for param in ["reasoning", "include_reasoning"])
            
            if has_tools:
                tool_models += 1
            if has_reasoning:
                reasoning_models += 1
                
            print(f"🧠 {m['name']}")
            print(f"ID: {m['id']}")
            print(f"Supported Parameters: {supported_params}")
            if has_tools:
                print("🛠️ SUPPORTS TOOL CALLING")
            if has_reasoning:
                print("🤔 SUPPORTS REASONING")
            print("-" * 60)
            
        print("\n📊 SUMMARY:")
        print(f"Total models: {len(free_models)}")
        print(f"Tool calling models: {tool_models}")
        print(f"Reasoning models: {reasoning_models}")

    except requests.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_and_filter_models()
