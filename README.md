Telegram Gemini Bot

A sophisticated Telegram bot powered by Google's Gemini AI & DeepSeek AI that offers advanced conversational abilities, image processing, document handling, and more.
<div align="center">
  <img src="assets/templates/Project_report_group5.png" alt="Telegram Gemini Bot Logo" width="200">
</div>

## ✨ Features

- **🤖 AI-Powered Conversations**
  - Natural language interactions using Google's Gemini AI
  - Context-aware responses with conversation memory

- **🖼️ Advanced Media Capabilities**
  - **Image Analysis**: Upload images for detailed AI descriptions and analysis
  - **Image Generation**: Create images from text prompts with multiple AI models
  - **Video Generation**: Generate short videos from text descriptions

- **📄 Document Processing**
  - Extract and analyze text from PDF documents
  - Perform searches and answer questions about document content

- **🎤 Voice Processing**
  - Transcribe voice messages to text
  - Automatic language detection for better accuracy

- **🌐 Multiple AI Models**
  - **Gemini AI**: Primary model for text and multimodal conversations
  - **Flux**: Fast image generation model
  - **Together AI**: Alternative model for image generation
  - **DeepSeek LLM**: Additional LLM option for specialized tasks

- **🔄 Performance Optimizations**
  - Rate limiting to prevent API abuse
  - Response caching for improved speed
  - Asynchronous programming for handling multiple requests

- **🛡️ Security Features**
  - Secure handling of user data and files
  - API key protection through environment variables
  - Input validation and sanitization

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- MongoDB database
- API keys for:
  - Telegram Bot
  - Google Gemini
  - Hugging Face (for image generation)
  - Together AI (optional)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Telegram-Gemini-Bot.git
cd Telegram-Gemini-Bot
```

2. **Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

  Create a .env file in the root directory:

```
DATABASE_URL=your_mongodb_connection_string
MONGODB_DB_NAME=your_database_name
TELEGRAM_BOT_TOKEN=your_bot_token
GEMINI_API_KEY=your_gemini_api_key
TEXT_TO_IMAGE_API_KEY=your_huggingface_api_key
TEXT_TO_VIDEO_API_KEY=your_video_api_key (HuggingFace)
TOGETHER_API_KEY=your_together_api_key
BOT_MODE=webhook  # or polling
WEBHOOK_URL=your_webhook_url  # if using webhook mode
PORT=8000
DEV_MODE=true
IGNORE_DB_ERROR=true
LOG_LEVEL=INFO
LOGS_DIR=logs
ADMIN_USER_ID=your_telegram_id
```

5. **Start the bot**

```bash
python app.py
```

## 🤖 Bot Commands

- `/start` - Initialize the bot and receive a welcome message
- `/help` - Display available commands and usage instructions
- `/settings` - Configure bot preferences
- `/generate_image` - Create an image from text description
- `/genimg` - Generate an image using Together AI
- `/generate_video` - Create a short video from text description
- `/reset` - Clear your conversation history
- `/stats` - View your usage statistics
- `/switchmodel` - Change between available AI models
- `/language` - Change the interface language

## 🏗️ Project Structure

```
src/
├── handlers/         # Message and command handlers
├── services/         # Core services (Gemini API, LLMs, etc.)
├── utils/            # Utility functions and helpers
├── database/         # Database connections and models
└── main.py           # Entry point for the application
```

## 🐳 Docker Deployment

### Using Docker

```bash
# Build the image
docker build -t telegram-gemini-bot .

# Run the container
docker run -d -p 8000:8000 --env-file .env telegram-gemini-bot
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 🔌 Webhook Setup

For webhook mode:

1. Expose your server (e.g., using ngrok for local development)
```bash
ngrok http 8000
```

2. Update your .env file with the webhook URL:
```
WEBHOOK_URL=https://your-domain.ngrok.io
BOT_MODE=webhook
```

3. Restart the bot to apply changes

## 🧪 Development

Run tests:
```bash
pytest tests/
```

## 🛡️ Security Considerations

- All API keys are stored in environment variables
- User data is securely stored in MongoDB
- Input validation prevents injection attacks
- Rate limiting protects against abuse

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Credits

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [Google Gemini AI](https://ai.google.dev/)
- [MongoDB](https://www.mongodb.com/)
- [Hugging Face](https://huggingface.co/)
- [Together AI](https://www.together.ai/)
