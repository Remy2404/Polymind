# Telegram Gemini Bot

This is a Telegram bot integrated with Gemini API for image generation and analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/TelegramGeminiBot.git
    cd TelegramGeminiBot
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables by creating a `.env` file with the following content:
    ```
    TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here
    GEMINI_API_KEY=your-gemini-api-key-here
    ```

4. Run the bot:
    ```bash
    python src/main.py
    ```

## Commands

- /start: Start the bot
- /help: List available commands
- /settings: Set user preferences
- /feedback: Provide feedback
