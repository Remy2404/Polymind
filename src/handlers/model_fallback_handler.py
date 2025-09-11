"""
Model Fallback Handler

This module handles automatic fallback to alternative models when the primary model
fails or times out. It provides intelligent model selection and user notification.
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional


class ModelFallbackHandler:
    """
    Handles automatic fallback to alternative models when primary models fail.

    Features:
    - Intelligent fallback model selection based on model capabilities
    - Automatic retry with exponential backoff
    - User notifications for fallback usage
    - Progress updates for complex questions
    - Timeout management per model
    """

    def __init__(self, response_formatter):
        self.logger = logging.getLogger(__name__)
        self.response_formatter = response_formatter

        # Fallback model mapping - defines which models to try when primary fails
        self.fallback_map = {
            # DeepSeek R1 variants fallback chain - prioritize more stable models
            "deepseek-r1-0528": [
                "deepseek-r1-distill-llama-70b",
                "deepseek-chat-v3-0324",
                "deepseek",
                "gemini",
            ],
            "deepseek-r1": [
                "deepseek-r1-0528",
                "deepseek-r1-distill-llama-70b",
                "deepseek-chat-v3-0324",
                "deepseek",
                "gemini",
            ],
            "deepseek-r1-distill-llama-70b": [
                "deepseek-r1-0528",
                "deepseek-chat-v3-0324",
                "deepseek",
                "gemini",
            ],
            "deepseek-chat-v3-0324": [
                "deepseek-r1-distill-llama-70b",
                "deepseek-r1-0528",
                "deepseek",
                "gemini",
            ],
            # Standard fallback for common models
            "gemini": [
                "deepseek",
                "deepseek-chat-v3-0324",
                "llama4_maverick",
            ],
            "deepseek": [
                "gemini",
                "deepseek-chat-v3-0324",
                "llama4_maverick",
            ],
            "llama4_maverick": [
                "deepseek-chat-v3-0324",
                "deepseek",
                "gemini",
            ],
        }

    def get_fallback_models(self, primary_model: str) -> List[str]:
        """
        Get fallback models based on the primary model.
        Returns a list of alternative models in order of preference.

        Args:
            primary_model: The primary model that failed

        Returns:
            List of fallback model names in order of preference
        """
        return self.fallback_map.get(
            primary_model,
            [
                "gemini",
                "deepseek",
                "deepseek-chat-v3-0324",
                "llama4_maverick",
            ],
        )

    async def attempt_with_fallback(
        self,
        primary_model: str,
        model_handler_factory,
        enhanced_prompt: str,
        history_context: List,
        max_tokens: int,
        model_timeout: float,
        message,
        is_complex_question: bool = False,
        quoted_text: str = None,
        gemini_api=None,
        openrouter_api=None,
        deepseek_api=None,
    ) -> Tuple[Optional[str], str]:
        """
        Attempt to generate response with automatic fallback to alternative models.

        Args:
            primary_model: The preferred model to try first
            model_handler_factory: Factory to create model handlers
            enhanced_prompt: The processed prompt for the model
            history_context: Conversation history for context
            max_tokens: Maximum tokens for response
            model_timeout: Timeout in seconds
            message: The original message object
            is_complex_question: Whether this is a complex question needing progress updates
            quoted_text: Quoted text if replying to another message
            gemini_api: Gemini API instance
            openrouter_api: OpenRouter API instance
            deepseek_api: DeepSeek API instance

        Returns:
            tuple: (response_text, actual_model_used)
        """
        models_to_try = [primary_model] + self.get_fallback_models(primary_model)

        # Remove duplicates while preserving order
        seen = set()
        models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]

        last_error = None

        for i, model_name in enumerate(models_to_try):
            try:
                self.logger.info(
                    f"Attempting response with model: {model_name} "
                    f"(attempt {i + 1}/{len(models_to_try)})"
                )

                # Get model handler for current model
                model_handler = model_handler_factory.get_model_handler(
                    model_name,
                    gemini_api=gemini_api,
                    openrouter_api=openrouter_api,
                    deepseek_api=deepseek_api,
                )

                # Adjust timeout for fallback models (slightly shorter to allow more attempts)
                current_timeout = (
                    model_timeout if i == 0 else min(model_timeout * 0.7, 180.0)
                )

                # Generate response based on question complexity
                if is_complex_question and current_timeout > 120:
                    # Use progress handling for complex questions
                    response = await self._handle_complex_question_with_progress(
                        message,
                        model_handler,
                        enhanced_prompt,
                        history_context,
                        max_tokens,
                        quoted_text,
                        current_timeout,
                    )
                else:
                    # Standard response generation
                    response = await asyncio.wait_for(
                        model_handler.generate_response(
                            prompt=enhanced_prompt,
                            context=history_context,
                            temperature=0.7,
                            max_tokens=max_tokens,
                        ),
                        timeout=current_timeout,
                    )

                if response and response.strip():
                    if i > 0:  # If we used a fallback model
                        self.logger.info(
                            f"Successfully generated response using fallback model: {model_name}"
                        )
                        # Send notification about fallback usage
                        await self._notify_fallback_usage(
                            message, primary_model, model_name
                        )
                        # Add a note about the fallback at the beginning of the response
                        fallback_note = (
                            f"*Using {model_name} (primary model unavailable)*\n\n"
                        )
                        response = fallback_note + response

                    return response, model_name
                else:
                    raise Exception(f"Empty response from {model_name}")

            except asyncio.TimeoutError as e:
                last_error = e
                self.logger.warning(f"Timeout with model {model_name}: {str(e)}")
                if (
                    "deepseek" in model_name.lower()
                    and "experiencing high load" not in str(e)
                ):
                    self.logger.info(
                        f"DeepSeek model {model_name} timed out, likely due to server load"
                    )
                continue

            except Exception as e:
                last_error = e
                self.logger.warning(f"Error with model {model_name}: {str(e)}")
                continue

        # If all models failed, raise the last error
        raise Exception(f"All fallback models failed. Last error: {str(last_error)}")

    async def _handle_complex_question_with_progress(
        self,
        message,
        model_handler,
        enhanced_prompt_with_guidelines,
        history_context,
        max_tokens,
        quoted_text,
        timeout_seconds,
    ) -> str:
        """
        Handle complex questions with progress updates to prevent timeout appearance.

        Args:
            message: The original message object
            model_handler: The model handler to use
            enhanced_prompt_with_guidelines: The processed prompt
            history_context: Conversation history
            max_tokens: Maximum tokens for response
            quoted_text: Quoted text if replying
            timeout_seconds: Timeout in seconds

        Returns:
            The generated response text
        """
        progress_messages = [
            "🔍 Analyzing your complex question...",
            "🧠 Processing detailed comparison...",
            "📊 Gathering comprehensive information...",
            "✍️ Formulating detailed response...",
        ]

        progress_msg = None
        progress_task = None

        async def update_progress():
            """Update progress messages periodically"""
            nonlocal progress_msg
            try:
                progress_msg = await message.reply_text(progress_messages[0])

                # Update progress every 45 seconds
                for i, msg in enumerate(progress_messages[1:], 1):
                    await asyncio.sleep(45)
                    try:
                        await progress_msg.edit_text(msg)
                    except Exception:
                        pass

            except asyncio.CancelledError:
                # Progress task was cancelled, clean up
                if progress_msg:
                    try:
                        await progress_msg.delete()
                    except Exception:
                        pass
                raise

        try:
            # Start progress updates
            progress_task = asyncio.create_task(update_progress())

            # Start the actual API request
            response = await model_handler.generate_response(
                prompt=enhanced_prompt_with_guidelines,
                context=history_context,
                temperature=0.7,
                max_tokens=max_tokens,
                quoted_message=quoted_text,
                timeout=timeout_seconds,
            )

            # Cancel progress updates and clean up
            if progress_task:
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

            if progress_msg:
                try:
                    await progress_msg.delete()
                except Exception:
                    pass

            return response

        except Exception as e:
            # Cancel progress updates
            if progress_task:
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

            if progress_msg:
                try:
                    await progress_msg.delete()
                except Exception:
                    pass

            raise e

    async def _notify_fallback_usage(
        self, message, primary_model: str, fallback_model: str
    ):
        """
        Send a brief notification to the user about fallback model usage.

        Args:
            message: The original message object
            primary_model: The primary model that failed
            fallback_model: The fallback model being used
        """
        try:
            notification_text = (
                f"⚠️ *{primary_model}* is temporarily unavailable. "
                f"Using *{fallback_model}* instead."
            )

            # Send the notification and delete it after 3 seconds
            notification_msg = await self.response_formatter.safe_send_message(
                message, notification_text
            )

            if notification_msg:
                # Delete the notification after 3 seconds
                await asyncio.sleep(3)
                try:
                    await notification_msg.delete()
                except Exception:
                    pass  # Ignore if message was already deleted

        except Exception as e:
            self.logger.debug(f"Failed to send fallback notification: {e}")

    def add_custom_fallback_mapping(self, model: str, fallback_list: List[str]):
        """
        Add or update custom fallback mapping for a specific model.

        Args:
            model: The primary model name
            fallback_list: List of fallback models in order of preference
        """
        self.fallback_map[model] = fallback_list
        self.logger.info(f"Added custom fallback mapping for {model}: {fallback_list}")

    def get_available_models(self) -> List[str]:
        """
        Get list of all available models (primary + fallback models).

        Returns:
            List of all unique model names
        """
        all_models = set(self.fallback_map.keys())
        for fallback_list in self.fallback_map.values():
            all_models.update(fallback_list)
        return sorted(list(all_models))

    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the fallback configuration.

        Returns:
            Dictionary with fallback statistics
        """
        return {
            "total_primary_models": len(self.fallback_map),
            "total_unique_models": len(self.get_available_models()),
            "average_fallback_count": sum(len(fb) for fb in self.fallback_map.values())
            / len(self.fallback_map),
            "fallback_map": self.fallback_map,
        }
