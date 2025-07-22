import requests
import json

url = "https://openrouter.ai/api/v1/models"
output_file = "models.json"

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    data = response.json()

    free_models_list = []
    if "data" in data and isinstance(data["data"], list):
        for model in data["data"]:
            model_id = model.get("id", "").lower()
            if ":free" in model_id:
                free_models_list.append(
                    {
                        "id": model.get("id"),
                        "name": model.get("name"),
                        "description": model.get(
                            "description", "No description available."
                        ),
                        "pricing": model.get("pricing"),
                    }
                )

    # Save the list of free models to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(free_models_list, f, indent=4, ensure_ascii=False)

    print(
        f"Successfully fetched {len(free_models_list)} free models and saved to {output_file}"
    )

except requests.exceptions.RequestException as e:
    print(f"Error fetching models from OpenRouter API: {e}")
except json.JSONDecodeError:
    print("Error decoding JSON response from OpenRouter API.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
