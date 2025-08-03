import google.generativeai as genai
import dotenv
import os

# Load environment variables from .env
dotenv.load_dotenv()

# Set the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
model = genai.GenerativeModel("gemini-pro-vision")

# Generate content (image)
response = model.generate_content(
    "Generate an image of fuzzy bunnies in a kitchen",
    generation_config={
        "temperature": 0.4,
    },
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ],
)

# Check if the response contains text (since Gemini-Pro-Vision doesn't generate images)
if response.text:
    print("Generated text description:")
    print(response.text)
else:
    print("No content was generated in the response.")
