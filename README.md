Certainly! I'll create a more detailed README.md file that explains the purpose of each function and method in the `PDFHandler` class. Here's the enhanced version:

# Telegram Gemini Bot

A sophisticated Telegram bot leveraging the Gemini API for advanced image processing, analysis, and PDF handling capabilities.

Version: 1.0.0
License: MIT (assumed, please adjust if different)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Credits](#credits)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Remy2404/Telegram-Gemini-Bot.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Telegram-Gemini-Bot
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   DATABASE_URL=
TELEGRAM_BOT_TOKEN=
GEMINI_API_KEY=
MONGODB_DB_NAME=
ADMIN_USER_ID=
BOT_MODE=
PORT=8000
WEBHOOK_URL=
   ```

5. Launch the bot:
   ```bash
   python src/main.py
   ```

## Usage

After starting the bot, interact with it on Telegram using these commands:

- `/start`: Initiate interaction with the bot
- `/help`: Display available commands and their usage
- `/pdf_info`: Get information about the currently loaded PDF
- Upload a PDF file to process and analyze its content
- Ask questions about the uploaded PDF content

## Features

1. **PDF Processing and Analysis**:
   - Upload and extract text from PDF files
   - Generate summaries of PDF content
   - Answer questions based on PDF content using AI
   - Provide PDF information (content size, summary)

2. **Conversation Management**:
   - Maintain conversation history for each user
   - Allow resetting of conversations

3. **Error Handling and Logging**:
   - Robust error handling for various scenarios
   - Detailed logging of user actions and errors

4. **User-Specific Data Management**:
   - Store and manage PDF content for each user
   - Handle multiple users simultaneously

## Code Structure

The `PDFHandler` class in `pdf_handler.py` is the core component for PDF-related functionality:

### Class: `PDFHandler`

#### Methods:

1. `__init__(self, gemini_api: GeminiAPI, text_handler: TextHandler)`:
   - Initializes the PDFHandler with necessary dependencies and data structures.

2. `extract_text_from_pdf(self, file_content: io.BytesIO) -> str`:
   - Extracts text content from a PDF file.

3. `handle_pdf_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE)`:
   - Handles the upload of PDF files, processes them, and stores the extracted text.

4. `handle_pdf_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE)`:
   - Provides information about the currently loaded PDF, including content size and summary.

5. `get_pdf_summary(self, user_id: int) -> str`:
   - Generates a summary of the PDF content using the Gemini API.

6. `ask_pdf_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE)`:
   - Processes user questions about the PDF content and provides AI-generated answers.

7. `handle_reset_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE)`:
   - Resets the conversation history for a user.

8. `get_handlers(self)`:
   - Returns a list of handlers for different bot commands and actions.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Implement your changes and commit: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

Ensure your code adheres to the project's coding style and includes appropriate tests.

## FAQ

Q: What's the maximum size of PDF files the bot can handle?
A: While there's no hard-coded limit, we recommend keeping PDFs under 20MB for optimal performance.

Q: Can the bot handle multiple PDFs from a single user?
A: Currently, the bot stores only the most recently uploaded PDF for each user. Uploading a new PDF will replace the previous one.

Q: How does the bot generate answers to questions about the PDF?
A: The bot uses the Gemini API to analyze the PDF content and generate relevant answers based on the extracted text.

## Credits

This project utilizes several open-source libraries:

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot): Telegram Bot API wrapper
- [PyPDF2](https://github.com/py-pdf/PyPDF2): PDF processing library
- [Google Generative AI](https://ai.google.dev/): Gemini API for AI-powered features
- [SQLAlchemy](https://www.sqlalchemy.org/): Database ORM
- [FastAPI](https://fastapi.tiangolo.com/): Web framework for API endpoints
- [pydub](https://github.com/jiaaro/pydub): Audio processing library
- [SpeechRecognition](https://github.com/Uberi/speech_recognition): Speech-to-text library

Maintained by [Your Name/Organization]. For support or inquiries, please open an issue on the GitHub repository.