"""
Example integration of enhanced voice processor with existing Telegram bot
"""
from telegram.ext import Application, MessageHandler, CommandHandler, filters
from src.handlers.enhanced_voice_handler import voice_handlers, initialize_voice_processor

async def setup_enhanced_voice_processing(application: Application):
    """
    Setup enhanced voice processing for the Telegram bot
    """
    # Initialize voice processor
    await initialize_voice_processor()

    # Add voice message handler
    application.add_handler(
        MessageHandler(
            filters.VOICE,
            voice_handlers['voice_message']
        )
    )

    # Add voice settings command
    application.add_handler(
        CommandHandler(
            "voice_settings",
            voice_handlers['voice_settings']
        )
    )

    print("✅ Enhanced voice processing setup complete!")


# Example usage in your main bot file:
"""
from integration_example import setup_enhanced_voice_processing

async def main():
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()

    # Setup enhanced voice processing
    await setup_enhanced_voice_processing(application)

    # ... add other handlers ...

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""

# Integration with existing text handlers:
"""
# In your existing text handler file, modify to handle voice transcriptions:

async def handle_text_message(update, context, is_voice_transcription=False):
    try:
        user_message = update.message.text

        # Add voice transcription indicator
        if is_voice_transcription:
            context.user_data['last_message_type'] = 'voice'
            # You might want to add voice-specific processing here

        # ... rest of your existing text processing logic ...
    except Exception as e:
        # Handle errors
        pass
"""
