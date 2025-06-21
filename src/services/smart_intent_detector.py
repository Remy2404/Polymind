"""
Enhanced Smart Intent Detector - Advanced Educational Content Detection
Modernized intent detection system using spaCy NLP for better accuracy
Specifically optimized for educational and tutorial requests
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import asyncio

from telegram import Update
from telegram.ext import ContextTypes
import spacy


class IntentType(Enum):
    """Enhanced enumeration of possible intents with better educational detection"""
    GENERATE_DOCUMENT = "generate_document"
    GENERATE_IMAGE = "generate_image" 
    GENERATE_VIDEO = "generate_video"
    EXPORT_CHAT = "export_chat"
    SWITCH_MODEL = "switch_model"
    GET_STATS = "get_stats"
    HELP = "help"
    RESET = "reset"
    SETTINGS = "settings"
    EDUCATIONAL = "educational"  
    CHAT = "chat"  
    ANALYZE = "analyze"
    UNKNOWN = "unknown"


# Backward compatibility - Legacy intent enum
CommandIntent = IntentType


class SmartIntentDetector:
    """
    Advanced Intent Detection using spaCy NLP
    Significantly reduces code complexity while improving accuracy
    """
    
    def __init__(self, command_handlers, gemini_api=None):
        self.command_handlers = command_handlers
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("✅ spaCy model loaded successfully")
        except OSError:
            self.logger.error("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Intent keywords for semantic matching
        self.intent_keywords = {
            CommandIntent.GENERATE_DOCUMENT: {
                "primary": ["document", "report", "article", "essay", "paper", "pdf", "docx"],
                "actions": ["create", "generate", "write", "make", "produce"],
                "context": ["business plan", "proposal", "summary", "documentation", "manual", "guide"]
            },
            CommandIntent.GENERATE_IMAGE: {
                "primary": ["image", "picture", "photo", "artwork", "illustration", "visual", "logo"],
                "actions": ["create", "generate", "draw", "design", "make", "paint", "render"],
                "context": ["art", "poster", "banner", "sketch", "diagram"]
            },
            CommandIntent.GENERATE_VIDEO: {
                "primary": ["video", "animation", "movie", "clip", "film"],
                "actions": ["create", "generate", "make", "produce", "animate"],
                "context": ["motion", "moving", "record"]
            },
            CommandIntent.EXPORT_CHAT: {
                "primary": ["export", "download", "save", "backup"],
                "actions": ["export", "save", "download", "convert"],
                "context": ["chat", "conversation", "history", "transcript", "messages"]
            },
            CommandIntent.SWITCH_MODEL: {
                "primary": ["model", "ai", "assistant"],
                "actions": ["switch", "change", "use", "select"],
                "context": ["gemini", "deepseek", "claude", "gpt", "llama", "available", "list"]
            },
            CommandIntent.GET_STATS: {
                "primary": ["stats", "statistics", "usage", "analytics"],
                "actions": ["show", "display", "get", "view"],
                "context": ["activity", "history", "performance", "data"]
            },
            CommandIntent.HELP: {
                "primary": ["help", "commands", "instructions"],
                "actions": ["help", "show", "list", "explain"],
                "context": ["how", "what", "capabilities", "features", "functions"]
            },
            CommandIntent.RESET: {
                "primary": ["reset", "clear", "delete", "forget", "erase"],
                "actions": ["reset", "clear", "delete", "start", "fresh"],
                "context": ["chat", "conversation", "history", "memory", "over"]
            },
            CommandIntent.SETTINGS: {
                "primary": ["settings", "preferences", "configuration", "config"],
                "actions": ["change", "modify", "update", "customize", "configure"],
                "context": ["personalize", "setup"]
            },
            CommandIntent.ANALYZE: {
                "primary": ["analyze", "analyse", "examine", "describe", "identify"],
                "actions": ["analyze", "describe", "explain", "tell", "identify", "recognize"],
                "context": ["what is", "what are", "what do you see", "extract", "summarize"]
            },
            CommandIntent.CHAT: {
                "primary": ["tutorial", "guide", "explanation", "question", "help"],
                "actions": ["write", "create", "explain", "teach", "tell", "compare", "list"],
                "context": ["how to", "step by step", "comprehensive", "detailed", "what is", "why", "when"]
            }
        }

    async def detect_intent(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        """
        Detect intent using spaCy's advanced NLP capabilities
        Returns: (intent, confidence_score)
        """
        if not message or len(message.strip()) < 2:
            return CommandIntent.UNKNOWN, 0.0
        
        # Special case: media analysis
        if has_attached_media:
            doc = self.nlp(message.lower())
            analyze_keywords = ["analyze", "describe", "what", "tell", "identify", "explain"]
            
            if any(token.lemma_ in analyze_keywords for token in doc) or message.strip() == "":
                return CommandIntent.ANALYZE, 0.9
        
        # Process text with spaCy
        doc = self.nlp(message.lower())
        
        # Extract semantic features
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        entities = [ent.label_ for ent in doc.ents]
        
        best_intent = CommandIntent.UNKNOWN
        best_score = 0.0
        
        # Score each intent based on semantic similarity
        for intent, keywords in self.intent_keywords.items():
            score = self._calculate_intent_score(tokens, entities, keywords)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Apply thresholds and defaults
        if best_score < 0.3:
            # For longer messages without clear command intent, assume chat
            if len(message.split()) > 5 and not has_attached_media:
                best_intent = CommandIntent.CHAT
                best_score = 0.5
            else:
                best_intent = CommandIntent.UNKNOWN
                best_score = 0.0
        
        # Educational content detection
        if self._is_educational_request(doc):
            if best_intent == CommandIntent.UNKNOWN or best_score < 0.6:
                best_intent = CommandIntent.CHAT
                best_score = max(best_score, 0.7)
        
        self.logger.info(f"🎯 Intent: '{message[:50]}...' -> {best_intent.value} ({best_score:.2f})")
        return best_intent, best_score

    def _calculate_intent_score(self, tokens: List[str], entities: List[str], keywords: Dict[str, List[str]]) -> float:
        """Calculate intent confidence score using semantic matching"""
        score = 0.0
        total_weight = 0.0
        
        # Primary keywords (highest weight)
        primary_matches = sum(1 for token in tokens if token in keywords["primary"])
        if primary_matches > 0:
            score += primary_matches * 0.4
            total_weight += 0.4
        
        # Action keywords (medium weight)
        action_matches = sum(1 for token in tokens if token in keywords["actions"])
        if action_matches > 0:
            score += action_matches * 0.3
            total_weight += 0.3
        
        # Context keywords (lower weight)
        context_matches = sum(1 for token in tokens if any(ctx_word in token for ctx_word in keywords["context"]))
        if context_matches > 0:
            score += context_matches * 0.2
            total_weight += 0.2
        
        # Normalize score
        return score / max(total_weight, 1.0) if total_weight > 0 else 0.0

    def _is_educational_request(self, doc) -> bool:
        """Detect if this is an educational/tutorial request"""
        educational_patterns = [
            "tutorial", "guide", "how to", "step by step", "explain", "teach", 
            "comprehensive", "detailed", "introduction", "walkthrough", "learn"
        ]
        
        text = doc.text.lower()
        return any(pattern in text for pattern in educational_patterns)

    async def route_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, intent: CommandIntent, original_message: str) -> bool:
        """
        Route the detected intent to appropriate command handler
        Returns: True if command was executed, False otherwise
        """
        try:
            handler_map = {
                CommandIntent.GENERATE_DOCUMENT: self._handle_document_generation,
                CommandIntent.GENERATE_IMAGE: self._handle_image_generation,
                CommandIntent.GENERATE_VIDEO: self._handle_video_generation,
                CommandIntent.EXPORT_CHAT: self._handle_export_chat,
                CommandIntent.SWITCH_MODEL: self._handle_model_switch,
                CommandIntent.GET_STATS: self._handle_stats,
                CommandIntent.HELP: self._handle_help,
                CommandIntent.RESET: self._handle_reset,
                CommandIntent.SETTINGS: self._handle_settings,
            }
            
            if intent in handler_map:
                return await handler_map[intent](update, context, original_message)
            elif intent in [CommandIntent.CHAT, CommandIntent.ANALYZE]:
                # Let these be handled by normal conversation flow
                return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error routing command for intent {intent}: {str(e)}")
            return False

    async def should_route_message(self, message: str, has_attached_media: bool = False) -> bool:
        """
        Determine if a message should be routed through the command system
        """
        if len(message.strip()) < 5:
            return False
            
        intent, confidence = await self.detect_intent(message, has_attached_media)
        
        # Don't route CHAT and ANALYZE intents
        if intent in [CommandIntent.CHAT, CommandIntent.ANALYZE]:
            return False
            
        return intent != CommandIntent.UNKNOWN and confidence > 0.4

    # Handler methods (simplified implementations)
    async def _handle_document_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
        """Handle document generation"""
        try:
            prompt = self._extract_clean_prompt(message, ["document", "report", "article"])
            context.args = prompt.split() if prompt else []
            await self.command_handlers.generate_ai_document_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Document generation error: {e}")
            return False

    async def _handle_image_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
        """Handle image generation"""
        try:
            prompt = self._extract_clean_prompt(message, ["image", "picture", "draw"])
            if not prompt:
                await update.message.reply_text("🎨 I'd be happy to generate an image! Could you describe what you'd like me to create?")
                return True
            context.args = prompt.split()
            await self.command_handlers.generate_together_image(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return False

    async def _handle_video_generation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> bool:
        """Handle video generation (placeholder)"""
        await update.message.reply_text(
            "🎬 Video generation is coming soon! For now, I can help you create:\n"
            "📄 Documents and reports\n🎨 Images and artwork\n📊 Data exports"
        )
        return True

    async def _handle_export_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle chat export"""
        try:
            await self.command_handlers.export_to_document(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False

    async def _handle_model_switch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle model switching"""
        try:
            await self.command_handlers.switch_model_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Model switch error: {e}")
            return False

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle stats display"""
        try:
            await self.command_handlers.handle_stats(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Stats error: {e}")
            return False

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle help command"""
        try:
            await self.command_handlers.help_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Help error: {e}")
            return False

    async def _handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle reset command"""
        try:
            await self.command_handlers.reset_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Reset error: {e}")
            return False

    async def _handle_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle settings command"""
        try:
            await self.command_handlers.settings(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Settings error: {e}")
            return False

    def _extract_clean_prompt(self, message: str, command_words: List[str]) -> str:
        """Extract clean prompt using spaCy's linguistic features"""
        doc = self.nlp(message.lower())
        
        # Remove command-related tokens
        filtered_tokens = []
        skip_next = False
        
        for token in doc:
            if skip_next:
                skip_next = False
                continue
                
            # Skip command words and common prefixes
            if (token.lemma_ in command_words or 
                token.lemma_ in ["create", "generate", "make", "write", "can", "you", "please"]):
                # If followed by "a" or "an", skip that too
                if token.i + 1 < len(doc) and doc[token.i + 1].lemma_ in ["a", "an", "the"]:
                    skip_next = True
                continue
            
            # Skip stopwords and punctuation at the beginning
            if len(filtered_tokens) == 0 and (token.is_stop or token.is_punct):
                continue
                
            filtered_tokens.append(token.text)
        
        cleaned = " ".join(filtered_tokens).strip()
        return cleaned if len(cleaned) > 3 else message
