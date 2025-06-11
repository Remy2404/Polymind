"""
AI Command Router - Intelligent Intent Detection and Command Execution
Automatically detects user intent from natural language and routes to appropriate commands
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import asyncio

from telegram import Update
from telegram.ext import ContextTypes


class CommandIntent(Enum):
    """Enumeration of possible command intents"""
    GENERATE_DOCUMENT = "generate_document"
    GENERATE_IMAGE = "generate_image" 
    GENERATE_VIDEO = "generate_video"
    EXPORT_CHAT = "export_chat"
    SWITCH_MODEL = "switch_model"
    GET_STATS = "get_stats"
    HELP = "help"
    RESET = "reset"
    SETTINGS = "settings"
    UNKNOWN = "unknown"


class AICommandRouter:
    """
    Intelligent AI-powered command router that detects user intent
    and automatically executes appropriate commands
    """
    
    def __init__(self, command_handlers, gemini_api=None):
        self.command_handlers = command_handlers
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
        
        # Intent detection patterns
        self.intent_patterns = {
            CommandIntent.GENERATE_DOCUMENT: [
                r"(?i).*(?:create|generate|make|write)\s+(?:a\s+)?(?:document|doc|report|article|paper|pdf|docx)",
                r"(?i).*(?:write|create)\s+(?:me\s+)?(?:a\s+)?(?:business\s+)?(?:plan|proposal|summary|essay)",
                r"(?i).*(?:generate|create|make)\s+.*(?:documentation|manual|guide)",
                r"(?i).*(?:need|want)\s+(?:a\s+)?(?:document|report|paper)\s+(?:about|on|regarding)",
                r"(?i).*document.*(?:about|on|regarding|for)",
                r"(?i).*(?:create|write|generate)\s+.*(?:report|analysis|study)"
            ],
            CommandIntent.GENERATE_IMAGE: [
                r"(?i).*(?:create|generate|make|draw|design)\s+(?:an?\s+)?image",
                r"(?i).*(?:create|generate|make|draw)\s+(?:a\s+)?(?:picture|photo|artwork|illustration)",
                r"(?i).*(?:draw|paint|design)\s+(?:me\s+)?(?:a\s+)?",
                r"(?i).*image.*(?:of|showing|with|depicting)",
                r"(?i).*(?:visualize|show\s+me|picture\s+of)",
                r"(?i).*(?:create|generate)\s+.*(?:logo|banner|poster|art)"
            ],
            CommandIntent.GENERATE_VIDEO: [
                r"(?i).*(?:create|generate|make)\s+(?:a\s+)?video",
                r"(?i).*(?:create|generate|make)\s+(?:a\s+)?(?:animation|movie|clip)",
                r"(?i).*video.*(?:of|showing|about|depicting)",
                r"(?i).*(?:animate|film|record)\s+",
                r"(?i).*(?:motion|moving)\s+(?:picture|image)"
            ],
            CommandIntent.EXPORT_CHAT: [
                r"(?i).*(?:export|download|save)\s+(?:this\s+)?(?:chat|conversation|history)",
                r"(?i).*(?:create|generate)\s+(?:a\s+)?(?:summary|transcript)\s+(?:of\s+)?(?:this\s+)?(?:chat|conversation)",
                r"(?i).*(?:save|backup)\s+(?:our\s+)?(?:conversation|messages)",
                r"(?i).*(?:export|convert).*(?:to\s+)?(?:pdf|docx|document)"
            ],            CommandIntent.SWITCH_MODEL: [
                r"(?i).*(?:switch|change|use)\s+(?:to\s+)?(?:a\s+)?(?:different\s+)?(?:model|ai|assistant)",
                r"(?i).*(?:change|switch)\s+(?:the\s+)?(?:ai\s+)?model",
                r"(?i).*(?:use\s+)?(?:gemini|deepseek|claude|gpt|llama)(?:\s+model|\s+ai|\s+assistant|$)",
                r"(?i).*(?:what\s+)?(?:models?\s+)?(?:are\s+)?available",
                r"(?i).*(?:list|show)\s+(?:me\s+)?(?:available\s+)?models?",
                r"(?i).*switch\s+to\s+(?:gemini|deepseek|claude|gpt|llama)",
                r"(?i).*change\s+(?:ai\s+)?model",
                r"(?i).*use\s+(?:the\s+)?(?:gemini|deepseek|claude|gpt|llama)\s+(?:model|ai)?",
                r"(?i).*models?\s+(?:available|list)"
            ],
            CommandIntent.GET_STATS: [
                r"(?i).*(?:show|display|get)\s+(?:my\s+)?(?:stats|statistics|usage)",
                r"(?i).*(?:how\s+much|how\s+many).*(?:used|messages|images|documents)",
                r"(?i).*(?:usage\s+)?(?:statistics|analytics|data)",
                r"(?i).*(?:my\s+)?(?:activity|history|performance)"
            ],
            CommandIntent.HELP: [
                r"(?i).*(?:help|commands|what\s+can\s+you\s+do)",
                r"(?i).*(?:how\s+do\s+i|how\s+to|instructions)",
                r"(?i).*(?:list\s+)?(?:available\s+)?(?:commands|features|functions)",
                r"(?i).*(?:what\s+are\s+your|what's\s+your)\s+(?:capabilities|features)"
            ],
            CommandIntent.RESET: [
                r"(?i).*(?:reset|clear|delete)\s+(?:chat|conversation|history|memory)",
                r"(?i).*(?:start\s+)?(?:over|fresh|new\s+conversation)",
                r"(?i).*(?:forget|erase)\s+(?:everything|all|history)"
            ],
            CommandIntent.SETTINGS: [
                r"(?i).*(?:settings|preferences|configuration|config)",
                r"(?i).*(?:change|modify|update)\s+(?:my\s+)?(?:settings|preferences)",
                r"(?i).*(?:customize|personalize|configure)"
            ]
        }
        
        # Keywords that strongly indicate specific intents
        self.strong_keywords = {
            CommandIntent.GENERATE_DOCUMENT: ["document", "report", "article", "essay", "paper", "pdf", "docx", "documentation"],
            CommandIntent.GENERATE_IMAGE: ["image", "picture", "photo", "draw", "artwork", "illustration", "visual", "logo"],
            CommandIntent.GENERATE_VIDEO: ["video", "animation", "movie", "clip", "animate", "motion"],
            CommandIntent.EXPORT_CHAT: ["export", "download", "save", "backup", "transcript"],
            CommandIntent.SWITCH_MODEL: ["model", "switch", "change", "gemini", "deepseek", "claude", "gpt"],
            CommandIntent.GET_STATS: ["stats", "statistics", "usage", "analytics", "activity"],
            CommandIntent.HELP: ["help", "commands", "instructions", "how"],
            CommandIntent.RESET: ["reset", "clear", "delete", "forget", "erase"],
            CommandIntent.SETTINGS: ["settings", "preferences", "configuration", "customize"]
        }

    async def detect_intent(self, message: str) -> Tuple[CommandIntent, float]:
        """
        Detect the user's intent from their message
        Returns: (intent, confidence_score)
        """
        message_lower = message.lower()
        best_intent = CommandIntent.UNKNOWN
        best_score = 0.0
        
        # Check pattern matches
        for intent, patterns in self.intent_patterns.items():
            pattern_score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, message):
                    matches += 1
                    pattern_score += 1.0
            
            if matches > 0:
                # Normalize by number of patterns
                pattern_score = pattern_score / len(patterns)
                
                # Boost score based on keyword presence
                keyword_boost = 0.0
                if intent in self.strong_keywords:
                    for keyword in self.strong_keywords[intent]:
                        if keyword in message_lower:
                            keyword_boost += 0.1
                
                total_score = pattern_score + keyword_boost
                
                if total_score > best_score:
                    best_score = total_score
                    best_intent = intent        # Minimum confidence threshold - balanced to avoid false positives while catching real commands
        if best_score < 0.4:
            best_intent = CommandIntent.UNKNOWN
            
        self.logger.info(f"Intent detection: '{message}' -> {best_intent.value} (confidence: {best_score:.2f})")
        return best_intent, best_score

    async def route_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, intent: CommandIntent, original_message: str) -> bool:
        """
        Route the detected intent to appropriate command handler
        Returns: True if command was executed, False otherwise
        """
        try:
            if intent == CommandIntent.GENERATE_DOCUMENT:
                return await self._handle_document_generation(update, context, original_message)
            
            elif intent == CommandIntent.GENERATE_IMAGE:
                return await self._handle_image_generation(update, context, original_message)
            
            elif intent == CommandIntent.GENERATE_VIDEO:
                return await self._handle_video_generation(update, context, original_message)
            
            elif intent == CommandIntent.EXPORT_CHAT:
                return await self._handle_export_chat(update, context)
            
            elif intent == CommandIntent.SWITCH_MODEL:
                return await self._handle_model_switch(update, context)
            
            elif intent == CommandIntent.GET_STATS:
                return await self._handle_stats(update, context)
            
            elif intent == CommandIntent.HELP:
                return await self._handle_help(update, context)
            
            elif intent == CommandIntent.RESET:
                return await self._handle_reset(update, context)
            
            elif intent == CommandIntent.SETTINGS:
                return await self._handle_settings(update, context)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error routing command for intent {intent}: {str(e)}")
            return False

    async def _handle_document_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
        """Handle document generation with extracted prompt"""
        try:
            # Extract the prompt from the message
            prompt = self._extract_prompt_for_document(message)
            
            # Set up context for document generation
            context.args = prompt.split() if prompt else []
            
            # Call document generation command
            await self.command_handlers.generate_ai_document_command(update, context)
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling document generation: {str(e)}")
            return False

    async def _handle_image_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
        """Handle image generation with extracted prompt"""
        try:
            # Extract the prompt from the message
            prompt = self._extract_prompt_for_image(message)
            
            if not prompt:
                await update.message.reply_text("🎨 I'd be happy to generate an image! Could you please describe what you'd like me to create?")
                return True
            
            # Set up context for image generation
            context.args = prompt.split()
            
            # Call image generation command
            await self.command_handlers.generate_together_image(update, context)
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling image generation: {str(e)}")
            return False

    async def _handle_video_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
        """Handle video generation (placeholder for future implementation)"""
        await update.message.reply_text(
            "🎬 Video generation is coming soon! For now, I can help you create:\n"
            "📄 Documents and reports\n"
            "🎨 Images and artwork\n"
            "📊 Data exports\n\n"
            "Would you like me to create something else?"
        )
        return True

    async def _handle_export_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle chat export"""
        try:
            await self.command_handlers.export_to_document(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling chat export: {str(e)}")
            return False

    async def _handle_model_switch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle model switching"""
        try:
            await self.command_handlers.switch_model_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling model switch: {str(e)}")
            return False

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle stats display"""
        try:
            await self.command_handlers.handle_stats(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling stats: {str(e)}")
            return False

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle help command"""
        try:
            await self.command_handlers.help_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling help: {str(e)}")
            return False

    async def _handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle reset command"""
        try:
            await self.command_handlers.reset_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling reset: {str(e)}")
            return False

    async def _handle_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle settings command"""
        try:
            await self.command_handlers.settings(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling settings: {str(e)}")
            return False

    def _extract_prompt_for_document(self, message: str) -> str:
        """Extract the actual prompt for document generation from user message"""
        # Remove command-like words and extract the core topic
        message_lower = message.lower()
        
        # Common phrases to remove
        remove_phrases = [
            r"(?i)can\s+you\s+",
            r"(?i)please\s+",
            r"(?i)could\s+you\s+",
            r"(?i)i\s+want\s+(?:you\s+to\s+)?",
            r"(?i)i\s+need\s+(?:you\s+to\s+)?",
            r"(?i)create\s+(?:a\s+)?(?:document|doc|report|article|paper)\s+(?:about|on|regarding|for)\s+",
            r"(?i)generate\s+(?:a\s+)?(?:document|doc|report|article|paper)\s+(?:about|on|regarding|for)\s+",
            r"(?i)write\s+(?:a\s+)?(?:document|doc|report|article|paper)\s+(?:about|on|regarding|for)\s+",
            r"(?i)make\s+(?:a\s+)?(?:document|doc|report|article|paper)\s+(?:about|on|regarding|for)\s+"
        ]
        
        cleaned_message = message
        for phrase in remove_phrases:
            cleaned_message = re.sub(phrase, "", cleaned_message).strip()
        
        # If nothing left, return original message
        if not cleaned_message or len(cleaned_message) < 3:
            return message
            
        return cleaned_message

    def _extract_prompt_for_image(self, message: str) -> str:
        """Extract the actual prompt for image generation from user message"""
        message_lower = message.lower()
        
        # Common phrases to remove
        remove_phrases = [
            r"(?i)can\s+you\s+",
            r"(?i)please\s+",
            r"(?i)could\s+you\s+",
            r"(?i)i\s+want\s+(?:you\s+to\s+)?",
            r"(?i)i\s+need\s+(?:you\s+to\s+)?",
            r"(?i)create\s+(?:an?\s+)?image\s+(?:of|showing|with|depicting)\s+",
            r"(?i)generate\s+(?:an?\s+)?image\s+(?:of|showing|with|depicting)\s+",
            r"(?i)make\s+(?:an?\s+)?image\s+(?:of|showing|with|depicting)\s+",
            r"(?i)draw\s+(?:me\s+)?(?:an?\s+)?(?:image\s+of\s+|picture\s+of\s+)?",
            r"(?i)create\s+(?:an?\s+)?(?:picture|photo|artwork|illustration)\s+(?:of|showing|with|depicting)\s+"
        ]
        
        cleaned_message = message
        for phrase in remove_phrases:
            cleaned_message = re.sub(phrase, "", cleaned_message).strip()
          # If nothing meaningful left, return None to ask for clarification
        if not cleaned_message or len(cleaned_message) < 3:
            return None
            
        return cleaned_message

    async def should_route_message(self, message: str) -> bool:
        """
        Determine if a message should be routed through the command system
        Returns True if the message looks like a command request
        """        # Add minimum length check to avoid routing very short messages
        if len(message.strip()) < 5:
            return False
            
        intent, confidence = await self.detect_intent(message)
        return intent != CommandIntent.UNKNOWN and confidence > 0.4
