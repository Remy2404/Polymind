"""
Ultra-Optimized AI Command Router for Production
Designed specifically for resource-constrained environments:
- 0.1 vCPU, 512MB RAM, 2GB Disk
- NO spaCy or heavy NLP dependencies
- Uses only fast regex patterns for maximum performance
"""

import logging
import re
from typing import Dict, Any, Tuple, List
from enum import Enum
from dataclasses import dataclass
from telegram import Update
from telegram.ext import ContextTypes


class CommandIntent(Enum):
    """Streamlined command intents for production"""

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
    CODING = "coding"
    MATHEMATICAL = "mathematical"
    CREATIVE = "creative"
    MULTILINGUAL = "multilingual"
    VISION = "vision"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Lightweight result structure for intent detection"""

    intent: CommandIntent
    confidence: float
    recommended_models: List[str]
    reasoning: str
    detected_entities: List[Dict[str, Any]]
    linguistic_features: Dict[str, Any]


class EnhancedIntentDetector:
    """
    Uses ONLY fast regex patterns - zero NLP overhead"""

    def __init__(self):
        import gc

        gc.set_threshold(50, 5, 5)
        gc.collect()
        self.logger = logging.getLogger(__name__)
        self.model_configs = {}
        self.logger.info("⚡ Ultra-lightweight IntentDetector initialized (regex-only)")
        self.intent_patterns = {
            CommandIntent.SWITCH_MODEL: {
                "patterns": [
                    r"(?i)(?:switch|change|use|select).{0,10}model",
                    r"(?i)model.{0,10}(?:switch|change)",
                    r"(?i)(?:different|another).{0,5}model",
                ],
                "keywords": ["switch", "change", "model", "different", "another"],
                "priority": 1,
            },
            CommandIntent.GENERATE_DOCUMENT: {
                "patterns": [
                    r"(?i)(?:create|generate|write|make).{0,15}(?:document|report|article|paper)",
                    r"(?i)(?:business\s+plan|proposal|summary)",
                    r"(?i)write.{0,10}(?:document|report|article)",
                    r"(?i)(?:detailed|comprehensive).{0,5}(?:report|analysis)",
                ],
                "keywords": [
                    "document",
                    "report",
                    "article",
                    "paper",
                    "write",
                    "create",
                    "generate",
                ],
                "priority": 2,
            },
            CommandIntent.GENERATE_IMAGE: {
                "patterns": [
                    r"(?i)(?:create|generate|draw|make).{0,15}(?:image|picture|photo)",
                    r"(?i)draw.{0,10}(?:image|picture)",
                    r"(?i)(?:artwork|illustration|logo|visual)",
                ],
                "keywords": ["image", "picture", "draw", "paint", "visual", "artwork"],
                "priority": 2,
            },
            CommandIntent.EDUCATIONAL: {
                "patterns": [
                    r"(?i)how\s+(?:to|do|does|can)",
                    r"(?i)what\s+(?:is|are|does)",
                    r"(?i)why\s+(?:is|are|does|do)",
                    r"(?i)difference\s+between",
                    r"(?i)can\s+you\s+(?:explain|teach|show)",
                    r"(?i)(?:tutorial|guide|explanation)",
                ],
                "keywords": [
                    "how",
                    "what",
                    "why",
                    "explain",
                    "tutorial",
                    "guide",
                    "difference",
                ],
                "priority": 3,
            },
            CommandIntent.CODING: {
                "patterns": [
                    r"(?i)(?:write|create|generate).{0,10}(?:code|script|function)",
                    r"(?i)(?:programming|coding).{0,10}(?:problem|task)",
                    r"(?i)debug.{0,5}code",
                    r"(?i)(?:python|javascript|java).{0,5}(?:code|function)",
                ],
                "keywords": [
                    "code",
                    "function",
                    "script",
                    "programming",
                    "debug",
                    "python",
                    "javascript",
                ],
                "priority": 2,
            },
        }

    async def detect_intent(self, text: str) -> IntentResult:
        """
        Ultra-fast intent detection using only regex patterns
        Optimized for 0.1 vCPU / 512MB RAM environments
        """
        try:
            return self._detect_intent_fast(text)
        except Exception as e:
            self.logger.error(f"❌ Error in intent detection: {e}")
            return IntentResult(
                intent=CommandIntent.UNKNOWN,
                confidence=0.0,
                recommended_models=[],
                reasoning="Error in intent detection",
                detected_entities=[],
                linguistic_features={},
            )

    def _detect_intent_fast(self, text: str) -> IntentResult:
        """Ultra-fast regex-based intent detection - zero overhead"""
        text_lower = text.lower().strip()
        best_match = None
        best_confidence = 0.0
        for intent, config in self.intent_patterns.items():
            keyword_matches = sum(
                1 for keyword in config.get("keywords", []) if keyword in text_lower
            )
            if keyword_matches == 0:
                continue
            for pattern in config.get("patterns", []):
                if re.search(pattern, text):
                    confidence = min(
                        0.9,
                        0.5
                        + (keyword_matches * 0.1)
                        + (config.get("priority", 3) * 0.1),
                    )
                    if confidence > best_confidence:
                        best_match = intent
                        best_confidence = confidence
                        break
        if best_match:
            return IntentResult(
                intent=best_match,
                confidence=best_confidence,
                recommended_models=self._get_recommended_models(best_match),
                reasoning=f"Fast detection: {best_match.value}",
                detected_entities=[],
                linguistic_features={},
            )
        return IntentResult(
            intent=CommandIntent.CHAT,
            confidence=0.6,
            recommended_models=["deepseek-v3-base"],
            reasoning="Default chat intent",
            detected_entities=[],
            linguistic_features={},
        )

    def _get_recommended_models(self, intent: CommandIntent) -> List[str]:
        """Get recommended models for each intent type - minimal memory usage"""
        model_map = {
            CommandIntent.GENERATE_DOCUMENT: ["gemini", "deepseek"],
            CommandIntent.GENERATE_IMAGE: ["gemini"],
            CommandIntent.EDUCATIONAL: ["deepseek", "llama4-maverick"],
            CommandIntent.CODING: ["deepcoder", "olympiccoder-32b"],
            CommandIntent.MATHEMATICAL: ["deepseek-prover-v2", "phi-4-reasoning-plus"],
            CommandIntent.CREATIVE: ["deephermes-3-mistral-24b"],
            CommandIntent.MULTILINGUAL: ["qwen3-235b"],
            CommandIntent.VISION: ["llama-3.2-11b-vision"],
            CommandIntent.SWITCH_MODEL: [],
        }
        return model_map.get(intent, ["deepseek-v3-base"])


class AICommandRouter:
    """
    Ultra-lightweight AI command router for production
    Optimized for resource-constrained environments"""

    def __init__(self, command_handlers, gemini_api=None):
        import gc

        gc.collect()
        self.intent_detector = EnhancedIntentDetector()
        self.command_handlers = command_handlers
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)

    async def detect_intent(
        self, message: str, has_attached_media: bool = False
    ) -> Tuple[CommandIntent, float]:
        """Detect intent and return tuple for backward compatibility"""
        result = await self.intent_detector.detect_intent(message)
        return result.intent, result.confidence

    async def detect_intent_with_recommendations(
        self, message: str, has_attached_media: bool = False
    ) -> IntentResult:
        """Detect intent with model recommendations"""
        return await self.intent_detector.detect_intent(message)

    async def route_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        intent: CommandIntent,
        original_message: str,
    ) -> bool:
        """Route detected intent to appropriate handler - streamlined for performance"""
        try:
            if intent == CommandIntent.GENERATE_DOCUMENT:
                return await self._handle_document_generation(
                    update, context, original_message
                )
            elif intent == CommandIntent.GENERATE_IMAGE:
                return await self._handle_image_generation(
                    update, context, original_message
                )
            elif intent == CommandIntent.GENERATE_VIDEO:
                return await self._handle_video_generation(
                    update, context, original_message
                )
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
            elif intent in [
                CommandIntent.EDUCATIONAL,
                CommandIntent.CHAT,
                CommandIntent.ANALYZE,
            ]:
                return False
            return False
        except Exception as e:
            self.logger.error(f"❌ Error routing command for intent {intent}: {str(e)}")
            return False

    async def _handle_document_generation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str
    ) -> bool:
        """Handle document generation - optimized for performance"""
        try:
            prompt = self._extract_prompt_for_document(message)
            context.args = prompt.split() if prompt else []
            await self.command_handlers.document_commands.generate_ai_document_command(
                update, context
            )
            return True
        except Exception as e:
            self.logger.error(f"Error handling document generation: {str(e)}")
            return False

    async def _handle_image_generation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str
    ) -> bool:
        """Handle image generation - optimized for performance"""
        try:
            prompt = self._extract_prompt_for_image(message)
            if not prompt:
                await update.message.reply_text(
                    "🎨 I'd be happy to generate an image! Could you please describe what you'd like me to create?"
                )
                return True
            context.args = prompt.split()
            await self.command_handlers.generate_together_image(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling image generation: {str(e)}")
            return False

    async def _handle_video_generation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, message: str
    ) -> bool:
        """Handle video generation (placeholder)"""
        await update.message.reply_text(
            "🎬 Video generation is coming soon! For now, I can help you create:\n"
            "📄 Documents and reports\n"
            "🎨 Images and artwork\n"
            "📊 Data exports"
        )
        return True

    async def _handle_export_chat(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Handle chat export"""
        try:
            await self.command_handlers.export_to_document(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling chat export: {str(e)}")
            return False

    async def _handle_model_switch(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Handle model switching"""
        try:
            await self.command_handlers.switch_model_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling model switch: {str(e)}")
            return False

    async def _handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Handle stats display"""
        try:
            await self.command_handlers.handle_stats(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling stats: {str(e)}")
            return False

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Handle help command"""
        try:
            await self.command_handlers.help_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling help: {str(e)}")
            return False

    async def _handle_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Handle reset command"""
        try:
            await self.command_handlers.reset_command(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling reset: {str(e)}")
            return False

    async def _handle_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Handle settings command"""
        try:
            await self.command_handlers.settings(update, context)
            return True
        except Exception as e:
            self.logger.error(f"Error handling settings: {str(e)}")
            return False

    def _extract_prompt_for_document(self, message: str) -> str:
        """Extract document prompt - optimized for speed"""
        cleaned = re.sub(
            r"(?i)(?:create|generate|write|make)\s+(?:a\s+)?(?:document|report|article|paper)\s+(?:about|on|regarding|for)\s+",
            "",
            message,
            count=1,
        )
        cleaned = re.sub(
            r"(?i)(?:can\s+you\s+|please\s+|could\s+you\s+|i\s+(?:want|need)\s+(?:you\s+to\s+)?)",
            "",
            cleaned,
            count=1,
        )
        return cleaned.strip() or message

    def _extract_prompt_for_image(self, message: str) -> str:
        """Extract image prompt - optimized for speed"""
        cleaned = re.sub(
            r"(?i)(?:create|generate|draw|make)\s+(?:an?\s+)?(?:image|picture|photo)\s+(?:of|showing|with|depicting)\s+",
            "",
            message,
            count=1,
        )
        cleaned = re.sub(
            r"(?i)(?:can\s+you\s+|please\s+|could\s+you\s+|i\s+(?:want|need)\s+(?:you\s+to\s+)?)",
            "",
            cleaned,
            count=1,
        )
        result = cleaned.strip()
        return result if len(result) > 3 else None

    def cleanup_memory(self):
        """Force garbage collection to free memory on low-resource instances"""
        import gc

        gc.collect()
        if hasattr(gc, "set_debug"):
            gc.set_debug(0)

    async def should_route_message(
        self, message: str, has_attached_media: bool = False
    ) -> bool:
        """
        Optimized routing decision - minimal processing
        """
        if len(message.strip()) < 5:
            return False
        intent, confidence = await self.detect_intent(message, has_attached_media)
        if intent in [
            CommandIntent.EDUCATIONAL,
            CommandIntent.CHAT,
            CommandIntent.ANALYZE,
        ]:
            return False
        return intent != CommandIntent.UNKNOWN and confidence > 0.4
