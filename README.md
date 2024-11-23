# Telegram Gemini Bot

This Telegram bot is integrated with the Gemini API for image generation, analysis, and PDF processing. It provides a versatile interface for users to interact with AI-powered features through Telegram.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Credits](#credits)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Remy2404/TelegramBot.git
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

## Usage

After starting the bot, you can interact with it using the following commands:

- `/start`: Start the bot
- `/help`: List available commands
- `/settings`: Set user preferences
- `/feedback`: Provide feedback
- `/start_pdf_conversation`: Start a conversation about an uploaded PDF
- `/end_pdf`: End the PDF conversation
- `/upload_pdf`: Upload a PDF for processing

## Features

1. **Image Generation and Analysis**: Utilize Gemini API for image-related tasks.
2. **PDF Processing**: 
   - Upload and process PDF documents
   - Extract text from PDFs
   - Generate summaries of PDF content
   - Ask questions about the PDF content
3. **Conversation Management**: Maintain conversation history for context-aware responses.
4. **Error Handling**: Robust error handling and logging system.
5. **Rate Limiting**: Implement rate limiting to prevent abuse.
6. **User Data Management**: Store and manage user preferences and data.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code adheres to the project's coding style and includes appropriate tests.

## FAQ

Q: How do I upload a PDF?
A: Use the `/upload_pdf` command and attach your PDF file to the message.

Q: What types of questions can I ask about the PDF?
A: You can ask any question related to the content of the PDF. The bot will use AI to analyze the content and provide answers.

Q: Is there a limit to the PDF file size?
A: While there's no hard limit, it's recommended to keep PDFs under 20MB for optimal performance.

## Credits

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for Telegram bot API wrapper
- [Google Generative AI](https://github.com/google/generative-ai-python) for Gemini API integration
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six) for PDF text extraction
- All contributors who have helped build and improve this project

For more information or support, please open an issue on the GitHub repository.