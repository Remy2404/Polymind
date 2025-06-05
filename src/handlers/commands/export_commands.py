import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from datetime import datetime
import logging
import io
import json
import glob


class ExportCommands:
    def __init__(self, gemini_api, user_data_manager, telegram_logger):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)
        # Try to import and initialize memory manager with MongoDB
        self.memory_manager = None
        try:
            from src.services.memory_context.memory_manager import MemoryManager
            from src.database.connection import get_database  # Get database connection

            db, client = get_database()
            self.memory_manager = MemoryManager(db=db)
            self.logger.info("Memory manager initialized with MongoDB connection")
        except ImportError as e:
            self.logger.warning(f"Memory manager not available: {e}")
        except Exception as e:
            self.logger.error(
                f"Error initializing memory manager: {e}"
            )  # Fallback to memory manager without MongoDB
            try:
                from src.services.memory_context.memory_manager import MemoryManager

                self.memory_manager = MemoryManager()
            except Exception as fallback_error:
                self.logger.error(
                    f"Failed to initialize fallback memory manager: {fallback_error}"
                )

    async def export_to_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate a professionally formatted PDF or DOCX document from text"""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            f"Document export requested by user {user_id}", user_id
        )

        # Get format preference
        format_options = [
            [
                InlineKeyboardButton(
                    "📄 PDF Format", callback_data="export_format_pdf"
                ),
                InlineKeyboardButton(
                    "📝 DOCX Format", callback_data="export_format_docx"
                ),
            ]
        ]
        format_markup = InlineKeyboardMarkup(format_options)

        # Check if a prompt was provided
        if context.args:
            # Save the content to be converted
            context.user_data["doc_export_text"] = " ".join(context.args)
            await update.message.reply_text(
                "Please select the document format you want to export to:",
                reply_markup=format_markup,
            )
        else:
            # No content provided - offer to export conversation history
            export_options = [
                [
                    InlineKeyboardButton(
                        "📜 Export Conversation", callback_data="export_conversation"
                    ),
                    InlineKeyboardButton(
                        "✏️ Provide Custom Text", callback_data="export_custom"
                    ),
                ],
                [InlineKeyboardButton("❌ Cancel", callback_data="export_cancel")],
            ]
            export_markup = InlineKeyboardMarkup(export_options)

            await update.message.reply_text(
                "What would you like to export to a document?\n\n"
                "You can export your conversation history or provide custom text.",
                reply_markup=export_markup,
            )

    async def handle_export_conversation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Export conversation history to a document"""
        user_id = (
            update.effective_user.id
        )  # Get conversation history from multiple sources
        formatted_content = "# Conversation History\n\n"
        has_content = False  # Try to get from memory manager first (new system)
        if self.memory_manager:
            try:
                # Use the export method from memory manager
                conversation_data = await self.memory_manager.export_conversation_data(
                    str(user_id), is_group=False
                )

                self.logger.info(
                    f"Memory manager conversation data: {conversation_data}"
                )
                if conversation_data:
                    self.logger.info(
                        f"Messages count: {len(conversation_data.get('messages', []))}"
                    )

                if conversation_data and conversation_data.get("messages"):
                    formatted_content += "## Conversation History (Memory System)\n\n"

                    # Add summary if available
                    if conversation_data.get("summary"):
                        formatted_content += (
                            f"**Summary:** {conversation_data['summary']}\n\n"
                        )
                        formatted_content += "---\n\n"

                    # Group consecutive messages by role
                    current_user_msg = None
                    messages = conversation_data["messages"]

                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role") or msg.get("sender", "")
                            content = msg.get("content", "")

                            if role == "user":
                                current_user_msg = content
                            elif role in ["assistant", "bot"] and current_user_msg:
                                if has_content:
                                    formatted_content += "\n---\n\n"
                                formatted_content += f"**User:** {current_user_msg}\n\n"
                                formatted_content += f"**Assistant:** {content}\n\n"
                                current_user_msg = None
                                has_content = True

            except Exception as e:
                self.logger.error(
                    f"Error retrieving from memory manager: {str(e)}"
                )  # Try to get from database contexts (fallback)
        if not has_content:
            try:
                user_data = await self.user_data_manager.get_user_data(user_id)
                db_contexts = user_data.get("contexts", [])

                if db_contexts:
                    formatted_content += "## Database Conversations\n\n"
                    current_user_msg = None

                    for msg in db_contexts:
                        if msg.get("role") == "user":
                            current_user_msg = msg.get("content", "")
                        elif msg.get("role") == "assistant" and current_user_msg:
                            # Add a conversation pair
                            if has_content:
                                formatted_content += "\n---\n\n"
                            formatted_content += f"**User:** {current_user_msg}\n\n"
                            formatted_content += (
                                f"**Assistant:** {msg.get('content', '')}\n\n"
                            )
                            current_user_msg = None
                            has_content = True

            except Exception as e:
                self.logger.error(
                    f"Error retrieving database history: {str(e)}"
                )  # Try to get from JSON memory files
        try:
            # Get the project root directory
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
            memory_dir = os.path.join(project_root, "data", "memory")
            memory_pattern = os.path.join(
                memory_dir, f"conversation_user_{user_id}*.json"
            )
            memory_files = glob.glob(memory_pattern)

            for file_path in memory_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        memory_data = json.load(f)

                    # Extract from short_term and medium_term
                    all_messages = []
                    if "short_term" in memory_data:
                        all_messages.extend(memory_data["short_term"])
                    if "medium_term" in memory_data:
                        all_messages.extend(memory_data["medium_term"])

                    if all_messages:
                        if has_content:
                            formatted_content += "\n\n## Memory File Conversations\n\n"

                        # Sort by timestamp if available
                        all_messages.sort(key=lambda x: x.get("timestamp", 0))

                        current_user_msg = None
                        for msg in all_messages:
                            if msg.get("sender") == "user":
                                current_user_msg = msg.get("content", "")
                            elif msg.get("sender") == "assistant" and current_user_msg:
                                if has_content:
                                    formatted_content += "\n---\n\n"
                                formatted_content += f"**User:** {current_user_msg}\n\n"
                                formatted_content += (
                                    f"**Assistant:** {msg.get('content', '')}\n\n"
                                )
                                current_user_msg = None
                                has_content = True

                except Exception as file_error:
                    self.logger.error(
                        f"Error reading memory file {file_path}: {str(file_error)}"
                    )

        except Exception as e:
            self.logger.error(f"Error retrieving memory files: {str(e)}")

        # Check if we found any conversation history
        if not has_content:
            await update.callback_query.edit_message_text(
                "You don't have any conversation history to export."
            )
            return

        # Store formatted content
        context.user_data["doc_export_text"] = formatted_content

        # Add debugging to see what content we're exporting
        self.logger.info(f"Export content length: {len(formatted_content)} characters")
        self.logger.info(
            f"Export content preview (first 200 chars): {formatted_content[:200]}"
        )
        self.logger.info(f"Has content flag: {has_content}")

        # Ask for format preference
        format_options = [
            [
                InlineKeyboardButton(
                    "📄 PDF Format", callback_data="export_format_pdf"
                ),
                InlineKeyboardButton(
                    "📝 DOCX Format", callback_data="export_format_docx"
                ),
            ]
        ]
        format_markup = InlineKeyboardMarkup(format_options)

        await update.callback_query.edit_message_text(
            "Please select the document format you want to export to:",
            reply_markup=format_markup,
        )

    async def generate_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, document_format: str
    ) -> None:
        """Generate and send document in the specified format"""
        user_id = update.effective_user.id

        if "doc_export_text" not in context.user_data:
            await update.callback_query.edit_message_text(
                "No content found for document generation. Please try again."
            )
            return

        content = context.user_data["doc_export_text"]

        # Send processing message
        await update.callback_query.edit_message_text(
            f"Generating {document_format.upper()} document... This may take a moment."
        )

        # Generate the document
        await self.try_generate_document(update, context, document_format)

    async def try_generate_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, document_format: str
    ) -> None:
        try:
            # Get user ID
            user_id = update.effective_user.id

            # Get content from user data
            content = context.user_data.get("doc_export_text", "")

            if not content.strip():
                await update.callback_query.edit_message_text(
                    "No content available to export. Please try again."
                )
                return

            if document_format.lower() == "docx":
                # Generate DOCX using python-docx
                document_bytes, filename = await self._generate_docx_document(
                    content, user_id
                )
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                # For other formats (like PDF), show an error for now
                await update.callback_query.edit_message_text(
                    f"❌ {document_format.upper()} export is not yet implemented. Please use DOCX format for now."
                )
                return

            # Send document to user
            with io.BytesIO(document_bytes) as doc_io:
                doc_io.name = filename

                caption = f"📄 Conversation Export\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=doc_io,
                    filename=filename,
                    caption=caption,
                )

            self.telegram_logger.log_message(
                f"Document generated successfully in {document_format} format",
                user_id,
            )

            # Clear the user data
            for key in ["doc_export_text", "aidoc_type", "aidoc_format", "aidoc_model"]:
                if key in context.user_data:
                    del context.user_data[key]

        except Exception as e:
            self.logger.error(f"Document generation error: {str(e)}")
            error_text = f"Sorry, there was an error generating your document: {str(e)}"
            # Try to edit the message first, then send a new one if it fails
            try:
                if update.callback_query:
                    await update.callback_query.edit_message_text(error_text)
                else:
                    # If callback query is not available, send a new message
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id, text=error_text
                    )
            except Exception as edit_error:
                self.logger.error(f"Error editing message: {str(edit_error)}")
                # Fallback: send a new message
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=error_text
                )

    async def _generate_docx_document(
        self, content: str, user_id: int
    ) -> tuple[bytes, str]:
        """Generate a DOCX document from markdown content"""
        try:
            # Add debugging to see what content we received
            self.logger.info(
                f"DOCX generation - Content length: {len(content)} characters"
            )
            self.logger.info(f"DOCX generation - Content preview: {content[:500]}")

            from docx import Document
            from docx.shared import Inches, Pt
            from docx.enum.text import WD_ALIGN_PARAGRAPH

            # Create a new document
            doc = Document()

            # Set document title
            title = doc.add_heading("Telegram Conversation Export", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # Add metadata paragraph
            meta_p = doc.add_paragraph()
            meta_p.add_run(
                f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ).italic = True
            meta_p.add_run(f"\nUser ID: {user_id}").italic = True
            meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # Add a page break
            doc.add_page_break()

            # Check if content is empty and add a message
            if not content or content.strip() == "":
                doc.add_paragraph("No conversation content available for export.")
                self.logger.warning("Empty content passed to DOCX generation")
            else:
                # Process the content line by line
                lines = content.split("\n")
                current_paragraph = None

                for line in lines:
                    line = line.strip()

                    if not line:
                        if current_paragraph:
                            current_paragraph = None
                        continue

                    # Handle headers
                    if line.startswith("##"):
                        doc.add_heading(line.replace("##", "").strip(), level=1)
                        current_paragraph = None
                    elif line.startswith("#"):
                        doc.add_heading(line.replace("#", "").strip(), level=2)
                        current_paragraph = None

                    # Handle bold text (markdown style)
                    elif line.startswith("**") and line.endswith("**"):
                        p = doc.add_paragraph()
                        p.add_run(line.replace("**", "")).bold = True
                        current_paragraph = None

                    # Handle list items
                    elif line.startswith("-") or line.startswith("*"):
                        doc.add_paragraph(line[1:].strip(), style="List Bullet")
                        current_paragraph = None

                    # Handle numbered lists
                    elif any(line.startswith(f"{i}.") for i in range(1, 100)):
                        doc.add_paragraph(
                            line.split(".", 1)[1].strip(), style="List Number"
                        )
                        current_paragraph = None

                    # Handle horizontal rules
                    elif line.startswith("---"):
                        doc.add_paragraph().add_run("_" * 50).italic = True
                        current_paragraph = None

                    # Regular text
                    else:
                        if current_paragraph is None:
                            current_paragraph = doc.add_paragraph()

                        # Handle inline formatting
                        if "**" in line:
                            parts = line.split("**")
                            for i, part in enumerate(parts):
                                if i % 2 == 0:
                                    current_paragraph.add_run(part)
                                else:
                                    current_paragraph.add_run(part).bold = True
                        else:
                            current_paragraph.add_run(line + "\n")

            # Save to BytesIO
            doc_io = io.BytesIO()
            doc.save(doc_io)
            doc_io.seek(0)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telegram_export_{user_id}_{timestamp}.docx"

            return doc_io.getvalue(), filename

        except Exception as e:
            self.logger.error(f"Error generating DOCX: {str(e)}")
            raise

    async def handle_export_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle export-related callback queries"""
        callback_data = update.callback_query.data

        try:
            if callback_data == "export_conversation":
                await self.handle_export_conversation(update, context)
            elif callback_data == "export_custom":
                await update.callback_query.edit_message_text(
                    "Please send me the text you want to convert to a document, "
                    "then use the /exportdoc command again."
                )
            elif callback_data == "export_cancel":
                await update.callback_query.edit_message_text("Export cancelled.")
            elif callback_data == "export_format_pdf":
                await self.generate_document(update, context, "pdf")
            elif callback_data == "export_format_docx":
                await self.generate_document(update, context, "docx")
            else:
                self.logger.warning(f"Unknown export callback: {callback_data}")
                await update.callback_query.edit_message_text(
                    "❌ Unknown export action. Please try again."
                )

        except Exception as e:
            self.logger.error(f"Error handling export callback: {str(e)}")
            await update.callback_query.edit_message_text(
                "❌ An error occurred while processing your request. Please try again."
            )
