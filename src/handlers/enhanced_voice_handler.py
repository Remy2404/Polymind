"""
Voice message handler with Faster-Whisper support
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from src.services.media.voice_processor import create_voice_processor, SpeechEngine

logger = logging.getLogger(__name__)

# Global voice processor instance
voice_processor = None


async def initialize_voice_processor():
    """Initialize the voice processor with Faster-Whisper"""
    global voice_processor
    if voice_processor is None:
        try:
            voice_processor = await create_voice_processor(
                engine=SpeechEngine.FASTER_WHISPER  # Use Faster-Whisper engine
            )
            logger.info(
                "✅ Voice processor initialized successfully with Faster-Whisper"
            )

            # Log engine info

            info = voice_processor.get_engine_info()
            available = [
                name for name, avail in info["available_engines"].items() if avail
            ]
            logger.info(f"Available engines: {available}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize voice processor: {e}")
            voice_processor = None


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Enhanced voice message handler with multiple engine support
    """
    try:
        # Initialize processor if needed
        await initialize_voice_processor()

        if voice_processor is None:
            await update.message.reply_text(
                "❌ Voice processing is currently unavailable. Please try again later."
            )
            return

        # Get voice file
        voice_file = await update.message.voice.get_file()
        user_id = str(update.effective_user.id)

        # Show processing message
        processing_msg = await update.message.reply_text(
            "🎤 Processing your voice message..."
        )

        try:
            # Download and convert voice file
            ogg_path, wav_path = await voice_processor.download_and_convert(
                voice_file, user_id
            )

            # Detect language hint from user context
            language_hint = context.user_data.get("language", "en-US")

            # Get best transcription with fallback
            (
                text,
                detected_lang,
                metadata,
            ) = await voice_processor.get_best_transcription(
                wav_path, language=language_hint, confidence_threshold=0.6
            )

            # Update processing message
            await processing_msg.edit_text(
                "🔄 Transcription complete, generating response..."
            )

            if text.strip():
                # Prepare response
                engine = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)

                # Format transcription response
                response = f"🎤 **Voice Message Transcribed:**\\n\\n{text}"

                # Add metadata for debugging (optional)
                if confidence > 0:
                    response += (
                        f"\\n\\n_Engine: {engine}, Confidence: {confidence:.1%}_"
                    )

                # Send transcription
                await processing_msg.edit_text(response, parse_mode="Markdown")

                # Now process with AI (integrate with existing AI handlers)
                await process_transcribed_text(update, context, text, processing_msg)

            else:
                error_msg = "❌ Sorry, I couldn't understand the voice message."

                # Add specific error info if available
                if "error" in metadata:
                    logger.warning(f"Voice transcription error: {metadata['error']}")

                # Suggest alternatives
                error_msg += "\\n\\n💡 **Tips:**\\n"
                error_msg += "• Speak clearly and avoid background noise\\n"
                error_msg += "• Try speaking in English for better accuracy\\n"
                error_msg += "• Send shorter voice messages (under 30 seconds)"

                await processing_msg.edit_text(error_msg, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Voice processing error for user {user_id}: {e}")
            await processing_msg.edit_text(
                "❌ Sorry, there was an error processing your voice message. Please try again."
            )

    except Exception as e:
        logger.error(f"Voice handler error: {e}")
        await update.message.reply_text(
            "❌ Voice processing failed. Please try sending a text message instead."
        )


async def process_transcribed_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, processing_msg
):
    """
    Process transcribed text with AI (integrate with existing text handlers)
    """
    try:
        # Update the message text in context to simulate text message
        update.message.text = text

        # Import and call existing text handler
        from src.handlers.text_handlers import handle_text_message

        # Process with AI
        await handle_text_message(update, context, is_voice_transcription=True)

        # If AI response was successful, delete the processing message
        try:
            await processing_msg.delete()
        except:
            pass  # Message might already be edited

    except Exception as e:
        logger.error(f"AI processing error for transcribed text: {e}")

        # Fall back to just showing transcription
        await processing_msg.edit_text(
            f"🎤 **Voice Message Transcribed:**\\n\\n{text}\\n\\n"
            f"_Note: AI processing unavailable - showing transcription only_",
            parse_mode="Markdown",
        )


async def handle_voice_settings_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    """
    Handle voice settings command
    """
    try:
        await initialize_voice_processor()

        if voice_processor is None:
            await update.message.reply_text("❌ Voice processor not available")
            return

        # Get engine information
        info = voice_processor.get_engine_info()

        # Format settings message
        settings_msg = "🎤 **Voice Recognition Settings**\\n\\n"

        # Available engines
        settings_msg += "**Available Engines:**\\n"
        for engine, available in info["available_engines"].items():
            status = "✅" if available else "❌"
            settings_msg += f"{status} {engine.title()}\\n"

        settings_msg += "\\n"

        # Current configuration
        settings_msg += f"**Default Engine:** {info['default_engine'].title()}\\n"
        settings_msg += f"**Recommended for English:** {info['recommended_engines']['english'].title()}\\n"
        settings_msg += f"**Recommended for Multilingual:** {info['recommended_engines']['multilingual'].title()}\\n"

        settings_msg += "\\n"

        # Usage stats (if available)
        from src.services.media.voice_config import voice_stats

        stats = voice_stats.get_stats()
        if stats["total_processed"] > 0:
            settings_msg += "**Statistics:**\\n"
            settings_msg += f"• Total Processed: {stats['total_processed']}\\n"

            if stats["success_rate"]:
                best_engine = voice_stats.get_best_engine()
                settings_msg += f"• Best Engine: {best_engine.title()}\\n"

        await update.message.reply_text(settings_msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Voice settings error: {e}")
        await update.message.reply_text("❌ Error retrieving voice settings")


# Command handlers to register
voice_handlers = {
    "voice_message": handle_voice_message,
    "voice_settings": handle_voice_settings_command,
}
