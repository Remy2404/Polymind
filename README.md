Telegram Gemini Bot


A sophisticated Telegram bot powered by Google's Gemini AI & DeepSeek AI that offers advanced conversational abilities, image processing, document handling, and more.

<div align="center">
  <img src="assets/templates/Project_report_group5.png" alt="Telegram Gemini Bot Logo" width="200">
</div>

## ✨ Features

- **🤖 AI-Powered Conversations**

  - Natural language interactions with multiple AI models (Gemini, DeepSeek, OpenRouter)
  - Context-aware responses with conversation memory that persists across sessions
  - Dynamic personality that avoids generic, robotic responses
  - Seamless topic transitions with maintained conversation context

- **🖼️ Advanced Media Capabilities**

  - **Image Analysis**: Upload images for detailed AI descriptions and contextual analysis
  - **Image Generation**: Create stunning images from text prompts via multiple AI engines:
    - Flux (fast, stylized generation)
    - Together AI (high-quality artistic renderings)
    - Imagen 3 (Google's advanced image model via Gemini)
  - **Video Generation**: Transform text descriptions into short, creative videos

- **📄 Document Processing**

  - Extract and analyze text from PDF documents with intelligent parsing
  - Perform semantic searches across document content
  - Generate AI-authored documents (reports, articles, summaries)
  - Export conversations to PDF/DOCX with customizable formatting

- **🧠 Knowledge Management**

  - Integrated knowledge graph for entity relationship tracking
  - Long-term memory storage for personalized user experiences
  - Contextual understanding that improves with usage

- **🎤 Voice & Language Processing**

  - Transcribe voice messages to text with high accuracy
  - Automatic language detection and multilingual support
  - Natural language processing for intent recognition

- **🌐 Multiple AI Models & Smart Switching**

  - **Gemini AI**: Primary model for text and multimodal conversations
  - **DeepSeek**: Thoughtful, philosophical responses with visual language
  - **Optimus Alpha**: Insightful problem-solving with colorful analogies
  - **DeepCoder**: Specialized programming and software development assistant
  - **Llama-4 Maverick**: Laid-back, friendly conversational model
  - **Seamless Switching**: Change between models using `/switchmodel` command
  - **Model Registry**: Centralized management of AI model configurations

- **🔄 Performance & Reliability**

  - Smart rate limiting to prevent API abuse
  - Response caching for improved speed and reduced API costs
  - Asynchronous processing for handling multiple requests
  - Automatic retry mechanisms for API failures

- **🛡️ Enhanced Security**

  - Secure handling of user data with privacy controls
  - API key protection through environment variables
  - Input validation to prevent injection attacks
  - User authentication and access controls

- **⚙️ User Customization**
  - Personalized user preferences that persist across sessions
  - Language settings for multilingual interface
  - Customizable response styles and conversation history management
  - Reminder system for scheduled notifications

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [MongoDB](https://www.mongodb.com/) database
- API keys for:
  - [Telegram Bot](https://t.me/botfather)
  - [Google Gemini](https://ai.google.dev/)
  - [Hugging Face](https://huggingface.co/settings/tokens) (for image/video generation)
  - [OpenRouter AI](https://openrouter.ai/keys) (for Optimus Alpha, DeepCoder, Llama-4)
  - [Together AI](https://api.together.ai/settings/api-keys) (optional)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Remy2404/Telegram-Gemini-Bot.git
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

Create a `.env` file in the root directory:

```env
DATABASE_URL=your_mongodb_connection_string
TELEGRAM_BOT_TOKEN=your_bot_token # Get from BotFather on Telegram
GEMINI_API_KEY=your_gemini_api_key # Get from Google AI Studio
OPENROUTER_API_KEY=your_openrouter_api_key # Get from OpenRouter website
TEXT_TO_IMAGE_API_KEY=your_huggingface_api_key # Get from Hugging Face settings  or The same API of huggingface
TEXT_TO_VIDEO_API_KEY=your_video_api_key # Get from Hugging Face settings or The same API of huggingface
TOGETHER_API_KEY=your_together_api_key # Get from Together AI website
BOT_MODE=webhook
WEBHOOK_URL=your_webhook_url
PORT=8000
DEV_MODE=true
IGNORE_DB_ERROR=true
LOGS_DIR=logs
```

5. **Start the bot**

```bash
python app.py or uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 🌐 Webhook Setup before running bot

1. Install ngrok:
2. Run ngrok:
3. Update your .env file with the webhook URL:
4. Restart the bot to apply changes

```bash
ngrok http 8000 # Replace in WEBHOOK_URL
```

## 🤖 Bot Commands

- `/start` - Initialize the bot and receive a welcome message
- `/help` - Display available commands and usage instructions
- `/settings` - Configure bot preferences
- `/generate_image` - Create images from text description (Flux)
- `/imagen3` - Generate images using Imagen 3 (via Gemini)
- `/genimg` - Generate an image using Together AI
- `/generate_video` - Create a short video from text description
- `/reset` - Clear your conversation history for the current model
- `/stats` - View your usage statistics
- `/switchmodel` - Change between available AI models (Gemini, DeepSeek, Optimus Alpha, DeepCoder, Llama-4)
- `/language` - Change the interface language
- `/exportdoc` - Export conversation or custom text to PDF/DOCX
- `/gendoc` - Generate an AI-authored document (article, report, etc.)

## 🏗️ Project Structure

```plaintext
src/
├── database/         # Database connections and models
├── handlers/         # Message, command, and callback handlers
├── services/         # Core services (API wrappers, data managers, etc.)
│   └── model_handlers/ # Specific AI model handlers, factory, registry
├── utils/            # Utility functions, logging, config
└── main.py           # Entry point (if applicable, otherwise app.py)
app.py                # Main application file (FastAPI/Webhook setup)
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
- [DeepSeek AI](https://www.deepseek.com/)
- [OpenRouter AI](https://openrouter.ai/)
- [MongoDB](https://www.mongodb.com/)
- [Hugging Face](https://huggingface.co/)
- [Together AI](https://www.together.ai/)
- [ngrok](https://ngrok.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [PyMongo](https://pymongo.readthedocs.io/en/stable/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)
- [Python-telegram-bot](https://python-telegram-bot.readthedocs.io/en/stable/)
- **🤖 AI-Powered Conversations**

  - Natural language interactions using multiple AI models (Gemini, DeepSeek, OpenRouter)
  - Context-aware responses with conversation memory per model

- **🖼️ Advanced Media Capabilities**

  - **Image Analysis**: Upload images for detailed AI descriptions and analysis (Gemini)
  - **Image Generation**: Create images from text prompts with multiple AI models (Flux, Together AI, Imagen 3 via Gemini)
  - **Video Generation**: Generate short videos from text descriptions

- **📄 Document Processing**

  - Extract and analyze text from PDF documents
  - Perform searches and answer questions about document content

- **🎤 Voice Processing**

  - Transcribe voice messages to text
  - Automatic language detection for better accuracy

- **🌐 Multiple AI Models & Management**

  - **Gemini AI**: Primary model for text and multimodal conversations
  - **DeepSeek LLM**: Alternative LLM for specialized tasks
  - **OpenRouter Models**: Access to models like Optimus Alpha, DeepCoder, Llama-4 Maverick
  - **Flux**: Fast image generation model
  - **Together AI**: Alternative model for image generation
  - **Model Switching**: Easily switch between configured AI models using `/switchmodel`
  - **Centralized Model Registry**: Manages model configurations and capabilities

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
- [MongoDB](https://www.mongodb.com/) database
- API keys for:
  - [Telegram Bot](https://t.me/botfather)
  - [Google Gemini](https://ai.google.dev/)
  - [Hugging Face](https://huggingface.co/settings/tokens) (for image/video generation)
  - [OpenRouter AI](https://openrouter.ai/keys) (for Optimus Alpha, DeepCoder, Llama-4)
  - [Together AI](https://api.together.ai/settings/api-keys) (optional)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Remy2404/Telegram-Gemini-Bot.git
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

Create a `.env` file in the root directory:

```env
DATABASE_URL=your_mongodb_connection_string
TELEGRAM_BOT_TOKEN=your_bot_token # Get from BotFather on Telegram
APP_URL=your_webhook_url # Use ngrok URL in development or domain in production
OPENROUTER_API_KEY=your_openrouter_api_key # Get from OpenRouter website
TEXT_TO_IMAGE_API_KEY=your_huggingface_api_key # Get from Hugging Face settings  or The same API of huggingface
TEXT_TO_VIDEO_API_KEY=your_video_api_key # Get from Hugging Face settings or The same API of huggingface
TOGETHER_API_KEY=your_together_api_key # Get from Together AI website
BOT_MODE=webhook
WEBHOOK_URL=your_webhook_url
ENVIRONMENT=production
PORT=8000
DEV_MODE=true
IGNORE_DB_ERROR=true
LOGS_DIR=logs
APP_URL=https://your_domain.com or webhook_url
ENVIRONMENT=production
```

5. **Start the bot**

```bash
python app.py or uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 🌐 Webhook Setup before running bot

1. Install ngrok:
2. Run ngrok:
3. Update your .env file with the webhook URL:
4. Restart the bot to apply changes

```bash
ngrok http 8000 # Replace in WEBHOOK_URL
```

## 🤖 Bot Commands

- `/start` - Initialize the bot and receive a welcome message
- `/help` - Display available commands and usage instructions
- `/settings` - Configure bot preferences
- `/generate_image` - Create images from text description (Flux)
- `/imagen3` - Generate images using Imagen 3 (via Gemini)
- `/genimg` - Generate an image using Together AI
- `/generate_video` - Create a short video from text description
- `/reset` - Clear your conversation history for the current model
- `/stats` - View your usage statistics
- `/switchmodel` - Change between available AI models (Gemini, DeepSeek, Optimus Alpha, DeepCoder, Llama-4)
- `/language` - Change the interface language
- `/exportdoc` - Export conversation or custom text to PDF/DOCX
- `/gendoc` - Generate an AI-authored document (article, report, etc.)

## 🏗️ Project Structure

```plaintext
src/
├── database/         # Database connections and models
├── handlers/         # Message, command, and callback handlers
├── services/         # Core services (API wrappers, data managers, etc.)
│   └── model_handlers/ # Specific AI model handlers, factory, registry
├── utils/            # Utility functions, logging, config
└── main.py           # Entry point (if applicable, otherwise app.py)
app.py                # Main application file (FastAPI/Webhook setup)
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
- [DeepSeek AI](https://www.deepseek.com/)
- [OpenRouter AI](https://openrouter.ai/)
- [MongoDB](https://www.mongodb.com/)
- [Hugging Face](https://huggingface.co/)
- [Together AI](https://www.together.ai/)
