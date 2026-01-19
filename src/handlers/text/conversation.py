"""
Text conversation handling functionality.
"""
import asyncio
import logging
from src.services.model_handlers.factory import ModelHandlerFactory
from src.services.model_handlers.model_configs import ModelConfigurations
from src.services.mcp_bot_integration import generate_mcp_response, is_model_mcp_compatible
from src.utils.log.telegramlog import telegram_logger


class TextConversationMixin:
    """Mixin providing text conversation handling functionality."""

    async def _handle_text_conversation(
        self, update, context, thinking_message, message_text,
        quoted_text, quoted_message_id, history_context, user_id, preferred_model
    ):
        """Handle regular text conversation."""
        message = update.message or update.edited_message
        enhanced_prompt = message_text

        if quoted_text:
            enhanced_prompt = self.prompt_formatter.add_context(message_text, "quote", quoted_text)

        # Get intelligent context
        intelligent_context = await self.conversation_manager.get_intelligent_context(user_id, message_text)
        if intelligent_context and intelligent_context.get("relevant_memory"):
            context_texts = self._extract_context_texts(intelligent_context["relevant_memory"])
            if context_texts:
                formatted_context = "\n".join(context_texts)
                enhanced_prompt = self.prompt_formatter.add_context(enhanced_prompt, "context", formatted_context)

        # Apply response guidelines
        enhanced_prompt_with_guidelines = await self.prompt_formatter.apply_response_guidelines(
            enhanced_prompt,
            ModelHandlerFactory.get_model_handler(
                preferred_model,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            ),
            context,
        )

        # Determine timeout based on query complexity
        is_long_form_request, is_complex_question = self._analyze_request_complexity(message_text)
        model_timeout = self._calculate_timeout(is_complex_question, is_long_form_request)

        try:
            response, actual_model_used = await self._generate_response(
                enhanced_prompt_with_guidelines, history_context, quoted_text,
                preferred_model, user_id, model_timeout
            )

            if response:
                response = self._clean_response_content(response)

            if thinking_message:
                try:
                    await thinking_message.delete()
                except Exception:
                    pass

            if response is None:
                await self.response_formatter.safe_send_message(
                    message, "Sorry, I couldn't generate a response. Please try rephrasing your message."
                )
                return

            # Send formatted response
            actual_model_handler = ModelHandlerFactory.get_model_handler(
                actual_model_used,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )
            await self._send_formatted_response(
                update, context, message, response,
                actual_model_handler.get_model_indicator(actual_model_used),
                quoted_text, quoted_message_id,
            )

            # Save to conversation history
            if response:
                await self._save_conversation(user_id, message_text, response, quoted_text, actual_model_used)
                telegram_logger.log_message("Text response sent successfully", user_id)

        except asyncio.TimeoutError:
            await self._handle_timeout_error(thinking_message, message, is_complex_question, is_long_form_request)
        except Exception as e:
            await self._handle_generation_error(e, thinking_message, message, is_complex_question)

    def _extract_context_texts(self, relevant_memory: list) -> list:
        """Extract text content from relevant memory items."""
        context_texts = []
        for item in relevant_memory:
            if isinstance(item, dict):
                if "content" in item:
                    context_texts.append(item["content"])
                elif "assistant_message" in item:
                    context_texts.append(item["assistant_message"])
                elif "user_message" in item:
                    context_texts.append(f"Previous: {item['user_message']}")
            elif isinstance(item, str):
                context_texts.append(item)
        return context_texts

    def _analyze_request_complexity(self, message_text: str) -> tuple[bool, bool]:
        """Analyze if request is long-form or complex."""
        long_form_indicators = [
            "100", "list", "q&a", "qcm", "questions", "examples",
            "write me", "generate", "create", "explain in detail",
            "step by step", "tutorial", "guide", "comprehensive",
        ]
        complex_indicators = [
            "compare", "comparison", "vs", "versus", "difference",
            "differences", "analyze", "analysis", "explain", "detailed",
            "comprehensive", "performance", "benchmark", "pros and cons",
            "advantages", "disadvantages",
        ]
        is_long_form = any(ind in message_text.lower() for ind in long_form_indicators)
        is_complex = any(ind in message_text.lower() for ind in complex_indicators)
        return is_long_form, is_complex

    def _calculate_timeout(self, is_complex: bool, is_long_form: bool) -> float:
        """Calculate timeout based on query complexity."""
        if is_complex:
            return 120.0
        elif is_long_form:
            return 90.0
        return 60.0

    async def _generate_response(
        self, prompt: str, history_context: list, quoted_text: str,
        preferred_model: str, user_id: int, timeout: float
    ) -> tuple[str, str]:
        """Generate AI response using MCP or standard model."""
        if is_model_mcp_compatible(preferred_model):
            mcp_response = await generate_mcp_response(
                prompt=prompt, user_id=user_id, model=preferred_model,
                temperature=0.7, context=history_context,
            )
            if mcp_response:
                return mcp_response, preferred_model

        # Use standard model handler
        model_handler = ModelHandlerFactory.get_model_handler(
            preferred_model,
            gemini_api=self.gemini_api,
            openrouter_api=self.openrouter_api,
            deepseek_api=self.deepseek_api,
        )
        response = await asyncio.wait_for(
            model_handler.generate_response(
                prompt, history_context, quoted_message=quoted_text, model=preferred_model
            ),
            timeout=timeout
        )
        return response, preferred_model

    async def _save_conversation(
        self, user_id: int, message_text: str, response: str,
        quoted_text: str, model_used: str
    ):
        """Save conversation to memory."""
        await self.memory_manager.extract_and_save_user_info(user_id, message_text)
        if quoted_text:
            await self.conversation_manager.add_quoted_message_context(
                user_id, quoted_text, message_text, response, model_used
            )
        else:
            await self.conversation_manager.save_message_pair(user_id, message_text, response, model_used)

        # Log to RAG
        if hasattr(self, "rag_integration"):
            try:
                await self.rag_integration.process_user_message(user_id=user_id, message_content=message_text)
                await self.rag_integration.process_ai_response(user_id=user_id, response_content=response)
            except Exception:
                pass

    async def _handle_timeout_error(self, thinking_message, message, is_complex: bool, is_long_form: bool):
        """Handle timeout errors."""
        if thinking_message:
            await thinking_message.delete()
        if is_complex:
            msg = "Your complex question required more processing time. Try breaking it into smaller parts."
        elif is_long_form:
            msg = "Your long-form request timed out. Try asking for a shorter response."
        else:
            msg = "Sorry, the request took too long. Please try again or rephrase your question."
        await self.response_formatter.safe_send_message(message, msg)

    async def _handle_generation_error(self, error: Exception, thinking_message, message, is_complex: bool):
        """Handle generation errors."""
        self.logger.error(f"Error generating response: {error}")
        if thinking_message:
            await thinking_message.delete()
        if "timeout" in str(error).lower() or isinstance(error, asyncio.TimeoutError):
            if is_complex:
                msg = "Your detailed question needed more time. Try breaking it into simpler parts."
            else:
                msg = "Processing took too long. Please try rephrasing your question."
        else:
            msg = "Sorry, there was an error processing your request. Please try again."
        await self.response_formatter.safe_send_message(message, msg)
