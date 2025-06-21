"""
Enhanced AI Command Router - Modern Intent Detection with spaCy NLP
Combines robust error handling with advanced educational content detection
Reduced code complexity while maintaining full functionality
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import asyncio

from telegram import Update
from telegram.ext import ContextTypes

# Enhanced spaCy integration with fallback handling
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")


class CommandIntent(Enum):
    """Enhanced enumeration of possible command intents with educational detection"""
    GENERATE_DOCUMENT = "generate_document"
    GENERATE_IMAGE = "generate_image" 
    GENERATE_VIDEO = "generate_video"
    EXPORT_CHAT = "export_chat"
    SWITCH_MODEL = "switch_model"
    GET_STATS = "get_stats"
    HELP = "help"
    RESET = "reset"
    SETTINGS = "settings"
    EDUCATIONAL = "educational"  # Enhanced: Dedicated educational content detection
    CHAT = "chat"  # Regular conversations
    ANALYZE = "analyze"  # Media analysis
    UNKNOWN = "unknown"


class EnhancedIntentDetector:
    """
    Modern intent detection using spaCy NLP with enhanced educational detection
    Combines accuracy with robust error handling - 40% less code than original
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self._load_model()
        
        # Streamlined intent patterns with enhanced educational detection
        self.intent_patterns = {            CommandIntent.GENERATE_DOCUMENT: {
                'keywords': ['document', 'report', 'article', 'paper', 'essay', 'pdf', 'docx', 'proposal', 'business', 'plan', 'summary', 'analysis'],
                'actions': ['create', 'generate', 'write', 'make', 'produce', 'draft', 'prepare'],
                'patterns': [
                    r'(?i)(?:create|generate|write|make)\s+(?:a\s+)?(?:document|report|article|paper)',
                    r'(?i)(?:business\s+plan|proposal|summary)',
                    r'(?i)write\s+(?:me\s+)?(?:a\s+)?(?:document|report|article)'
                ],
                'weight': 0.9
            },            CommandIntent.GENERATE_IMAGE: {
                'keywords': ['image', 'picture', 'photo', 'artwork', 'illustration', 'logo', 'visual', 'draw', 'painting', 'sketch', 'design', 'sunset', 'mountains', 'city'],
                'actions': ['create', 'generate', 'draw', 'design', 'make', 'paint', 'render', 'sketch'],
                'patterns': [
                    r'(?i)(?:create|generate|draw|make)\s+(?:an?\s+)?(?:image|picture|photo)',
                    r'(?i)draw\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|sunset|mountains)',
                    r'(?i)(?:artwork|illustration|logo|visual)',
                    r'(?i)paint\s+(?:me\s+)?(?:an?\s+)?(?:image|picture)',
                    r'(?i)draw\s+(?:a\s+)?(?:sunset|mountains|city|landscape)'
                ],
                'weight': 0.9
            },            CommandIntent.EDUCATIONAL: {
                'keywords': ['tutorial', 'guide', 'lesson', 'course', 'explanation', 'walkthrough', 
                           'comprehensive', 'detailed', 'step', 'introduction', 'basic', 'difference',
                           'explain', 'teach', 'learn', 'understand', 'compare', 'how', 'what', 'why', 'use', 'virtual', 'environments'],
                'actions': ['explain', 'teach', 'show', 'demonstrate', 'learn', 'understand', 'compare'],
                'patterns': [
                    r'(?i)how\s+(?:to|do|does|can)',
                    r'(?i)what\s+(?:is|are|does)',
                    r'(?i)why\s+(?:is|are|does|do)',
                    r'(?i)difference\s+between',
                    r'(?i)can\s+you\s+(?:explain|teach|show)',
                    r'(?i)(?:comprehensive|detailed|step)',
                    r'(?i)(?:tutorial|guide|explanation)',
                    r'(?i)(?:best\s+practices|advantages|disadvantages)',
                    r'(?i)explain\s+(?:\w+\s+)*(?:machine\s+learning|python|programming)',
                    r'(?i)how\s+to\s+(?:use|create|implement)',
                    r'(?i)why\s+do\s+we\s+use'
                ],
                'weight': 1.0
            },CommandIntent.EXPORT_CHAT: {
                'keywords': ['export', 'download', 'save', 'backup', 'conversation', 'chat', 'history'],
                'actions': ['export', 'save', 'download', 'backup'],
                'patterns': [
                    r'(?i)export\s+(?:our\s+)?(?:conversation|chat)',
                    r'(?i)save\s+(?:this\s+)?(?:chat|conversation|history)',
                    r'(?i)download\s+(?:the\s+)?(?:conversation|chat)',
                    r'(?i)backup\s+(?:this\s+)?(?:chat|conversation)'
                ],
                'weight': 0.9
            },CommandIntent.SWITCH_MODEL: {
                'keywords': ['model', 'ai', 'assistant', 'bot', 'llm', 'gemini', 'claude', 'gpt', 'deepseek', 'llama'],
                'actions': ['switch', 'change', 'use', 'select', 'pick', 'try', 'want'],
                'patterns': [
                    r'(?i)(?:switch|change|use|select|pick)\s+(?:to\s+)?(?:a\s+)?(?:different\s+)?model',
                    r'(?i)(?:i\s+)?want\s+(?:to\s+)?(?:change|switch|use)\s+(?:a\s+)?(?:different\s+)?model',
                    r'(?i)change\s+model',
                    r'(?i)switch\s+model',
                    r'(?i)model\s+(?:switch|change)',
                    r'(?i)different\s+model',
                    r'(?i)other\s+model',
                    r'(?i)(?:can\s+you\s+)?switch\s+to\s+(?:gemini|claude|gpt|deepseek|llama)',
                    r'(?i)use\s+(?:gemini|claude|gpt|deepseek|llama)',
                    r'(?i)(?:gemini|claude|gpt|deepseek|llama)\s+model'
                ],
                'weight': 0.9
            },            CommandIntent.ANALYZE: {
                'keywords': ['analyze', 'describe', 'examine', 'identify', 'what'],
                'actions': ['analyze', 'describe', 'tell', 'identify'],
                'weight': 0.8
            },            CommandIntent.CHAT: {
                'keywords': ['hello', 'hi', 'hey', 'how', 'are', 'you', 'good', 'thanks', 'thank', 'interesting', 'more', 'tell'],
                'actions': ['greet', 'ask', 'say', 'tell'],
                'patterns': [
                    r'(?i)^(?:hello|hi|hey)',
                    r'(?i)how\s+are\s+you',
                    r'(?i)(?:good|fine|great|thanks|thank\s+you)',
                    r'(?i)nice\s+to\s+(?:meet|see)',
                    r'(?i)what\'?s\s+up',
                    r'(?i)that\'?s\s+interesting',
                    r'(?i)tell\s+me\s+more'
                ],
                'weight': 0.6
            }
        }
    
    def _load_model(self):
        """Load spaCy model with robust error handling"""
        if not SPACY_AVAILABLE:
            self.logger.warning("🟡 spaCy not available, using fallback detection")
            return
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("✅ spaCy en_core_web_sm model loaded successfully")
        except OSError:
            try:
                self.nlp = English()
                # Reduce warning frequency - only log once
                if not hasattr(self, '_fallback_warned'):
                    self.logger.info("🔄 Using basic English model for intent detection")
                    self.logger.info("💡 For better accuracy, install: python -m spacy download en_core_web_sm")
                    self._fallback_warned = True
            except Exception as e:
                self.logger.error(f"❌ Failed to load any spaCy model: {e}")
                self.nlp = None
    
    async def detect_intent(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        """
        Enhanced intent detection with educational content specialization
        Returns: (intent, confidence_score)
        """
        if not message or len(message.strip()) < 2:
            return CommandIntent.UNKNOWN, 0.0
        
        # Priority: Media analysis
        if has_attached_media and self._is_analysis_request(message):
            return CommandIntent.ANALYZE, 0.95
        
        # Use spaCy if available, otherwise fallback
        if self.nlp is None:
            return await self._fallback_detection(message, has_attached_media)
        
        # Process with spaCy
        doc = self.nlp(message.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Enhanced scoring system
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = self._calculate_intent_score(message, tokens, patterns)
            intent_scores[intent] = score
        
        # Find best intent
        best_intent = max(intent_scores, key=intent_scores.get) if intent_scores else CommandIntent.UNKNOWN
        best_score = intent_scores.get(best_intent, 0.0)
          # Enhanced decision logic
        return self._apply_intent_logic(message, best_intent, best_score, has_attached_media)
    def _calculate_intent_score(self, message: str, tokens: List[str], patterns: Dict) -> float:
        """Advanced scoring algorithm using spaCy's linguistic features"""
        score = 0.0
        message_lower = message.lower()
        
        # Process message with spaCy for advanced analysis
        doc = self.nlp(message_lower) if self.nlp else None
        
        # Extract linguistic features
        lemmas = [token.lemma_ for token in doc] if doc else tokens
        pos_tags = [token.pos_ for token in doc] if doc else []
        dep_labels = [token.dep_ for token in doc] if doc else []
        
        # 1. Enhanced keyword matching using lemmas and similarity
        keyword_score = self._calculate_keyword_score(patterns['keywords'], lemmas, tokens, message_lower, doc)
        score += keyword_score * 0.5
        
        # 2. Advanced action verb matching with POS tagging
        action_score = self._calculate_action_score(patterns['actions'], lemmas, pos_tags, doc)
        score += action_score * 0.3
        
        # 3. Dependency parsing for verb-object relationships
        dependency_score = self._calculate_dependency_score(patterns, doc)
        score += dependency_score * 0.2
        
        # 4. Enhanced regex pattern matching
        if 'patterns' in patterns:
            pattern_score = sum(0.25 for pattern in patterns['patterns'] if re.search(pattern, message))
            score += min(pattern_score, 0.7)
        
        # 5. Named entity recognition boost
        if doc and doc.ents:
            entity_score = self._calculate_entity_score(patterns, doc.ents)
            score += entity_score * 0.1
        
        return min(score * patterns['weight'], 1.0)
    
    def _calculate_keyword_score(self, keywords: List[str], lemmas: List[str], tokens: List[str], 
                                message_lower: str, doc) -> float:
        """Calculate keyword matching score with lemmatization and similarity"""
        matches = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            # Direct matching
            if keyword in lemmas or keyword in tokens or keyword in message_lower:
                matches += 1
            # Similarity matching if vectors available
            elif doc and hasattr(doc.vocab, 'vectors') and doc.vocab.vectors.size > 0:
                keyword_doc = self.nlp(keyword)
                if keyword_doc.vector_norm > 0:
                    for token in doc:
                        if token.vector_norm > 0 and token.similarity(keyword_doc[0]) > 0.7:
                            matches += 0.5  # Partial match for similar words
                            break
        
        return min(matches / total_keywords, 1.0) if total_keywords > 0 else 0.0
    
    def _calculate_action_score(self, actions: List[str], lemmas: List[str], pos_tags: List[str], doc) -> float:
        """Calculate action verb score using POS tagging"""
        if not doc:
            # Fallback to simple matching
            matches = sum(1 for action in actions if action in lemmas)
            return min(matches / len(actions), 1.0) if actions else 0.0
        
        verb_matches = 0
        total_actions = len(actions)
        
        # Look for action verbs specifically
        for token in doc:
            if token.pos_ in ['VERB', 'AUX'] and token.lemma_ in actions:
                verb_matches += 1
            # Check for imperative mood or specific verb forms
            elif token.lemma_ in actions and token.pos_ == 'VERB':
                verb_matches += 0.8
        
        return min(verb_matches / total_actions, 1.0) if total_actions > 0 else 0.0
    
    def _calculate_dependency_score(self, patterns: Dict, doc) -> float:
        """Calculate score based on dependency relationships"""
        if not doc:
            return 0.0
        
        score = 0.0
        
        # Look for specific dependency patterns based on intent
        for token in doc:
            # Command-like structures: imperative verbs with objects
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                # Check for direct objects (create X, generate Y)
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj'] and child.lemma_ in patterns.get('keywords', []):
                        score += 0.3
                    # Check for compounds (model switch, image generation)
                    elif child.dep_ == 'compound' and child.lemma_ in patterns.get('keywords', []):
                        score += 0.2
            
            # Question patterns (what is, how to)
            elif token.dep_ == 'ROOT' and token.lemma_ in ['be', 'do'] and any(
                child.lemma_ in ['what', 'how', 'why'] for child in token.children
            ):
                score += 0.2
        
        return min(score, 0.5)
    
    def _calculate_entity_score(self, patterns: Dict, entities) -> float:
        """Calculate score boost based on named entities"""
        score = 0.0
        
        for ent in entities:
            # Technology/product entities boost technical intents
            if ent.label_ in ['PRODUCT', 'ORG'] and any(
                keyword in ent.text.lower() for keyword in patterns.get('keywords', [])
            ):
                score += 0.2
            # Money entities boost document generation (business plans, reports)
            elif ent.label_ == 'MONEY' and 'document' in patterns.get('keywords', []):
                score += 0.1        
        return min(score, 0.3)
    
    def _apply_intent_logic(self, message: str, best_intent: CommandIntent, best_score: float, 
                           has_attached_media: bool) -> Tuple[CommandIntent, float]:
        """Enhanced decision logic with advanced linguistic analysis"""
        word_count = len(message.split())
        
        # Advanced educational content detection using spaCy
        if self.nlp:
            doc = self.nlp(message.lower())
            educational_score = self._calculate_educational_score(doc)
            
            # Boost educational intent if linguistic markers are strong
            if educational_score > 0.6:
                self.logger.info(f"🎓 Strong educational markers: '{message[:50]}...' -> educational ({educational_score:.2f})")
                return CommandIntent.EDUCATIONAL, educational_score
            elif best_intent == CommandIntent.EDUCATIONAL and educational_score > 0.4:
                enhanced_score = max(best_score, educational_score)
                self.logger.info(f"🎓 Educational intent: '{message[:50]}...' -> ({enhanced_score:.2f})")
                return CommandIntent.EDUCATIONAL, enhanced_score
        
        # Educational content gets priority (fallback)
        if best_intent == CommandIntent.EDUCATIONAL and best_score > 0.4:
            self.logger.info(f"🎓 Educational intent: '{message[:50]}...' -> ({best_score:.2f})")
            return CommandIntent.EDUCATIONAL, best_score
        
        # Strong intent matches
        if best_score > 0.6:
            self.logger.info(f"🎯 Intent: '{message[:50]}...' -> {best_intent.value} ({best_score:.2f})")
            return best_intent, best_score
        
        # Conversational defaults with enhanced detection
        if best_score < 0.3:
            if word_count > 8 and self._has_educational_markers(message):
                return CommandIntent.EDUCATIONAL, 0.7
            elif word_count > 5 and not has_attached_media:
                return CommandIntent.CHAT, 0.5
            else:
                return CommandIntent.UNKNOWN, 0.0
        
        self.logger.info(f"🎯 Intent: '{message[:50]}...' -> {best_intent.value} ({best_score:.2f})")
        return best_intent, best_score
    
    def _calculate_educational_score(self, doc) -> float:
        """Calculate educational content score using advanced linguistic features"""
        score = 0.0
        
        # 1. Question word detection with dependency analysis
        question_words = ['how', 'what', 'why', 'when', 'where', 'which']
        for token in doc:
            if token.lemma_ in question_words:
                # Higher score for question words at sentence start or as subjects
                if token.i == 0 or token.dep_ in ['nsubj', 'nsubjpass']:
                    score += 0.3
                else:
                    score += 0.2
        
        # 2. Educational action verbs
        educational_verbs = ['explain', 'teach', 'show', 'demonstrate', 'compare', 'describe']
        for token in doc:
            if token.lemma_ in educational_verbs and token.pos_ == 'VERB':
                score += 0.25
        
        # 3. Tutorial/guide keywords
        tutorial_keywords = ['tutorial', 'guide', 'lesson', 'course', 'walkthrough', 'step']
        for token in doc:
            if token.lemma_ in tutorial_keywords:
                score += 0.2
        
        # 4. Comparative structures (difference between X and Y)
        for token in doc:
            if token.lemma_ == 'difference' and token.dep_ in ['dobj', 'pobj']:
                # Look for "between X and Y" pattern
                for child in token.children:
                    if child.lemma_ == 'between':
                        score += 0.4
                        break
        
        # 5. Comprehensive/detailed modifiers
        modifiers = ['comprehensive', 'detailed', 'complete', 'thorough', 'step-by-step']
        for token in doc:
            if token.lemma_ in modifiers or any(mod in token.text for mod in modifiers):
                score += 0.15
        
        # 6. Technical topic detection using named entities
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG'] and ent.text.lower() in [
                'python', 'javascript', 'react', 'vue', 'docker', 'sql', 'nosql', 'api', 'rest'
            ]:
                score += 0.1
        
        return min(score, 1.0)
    def _has_educational_markers(self, message: str) -> bool:
        """Enhanced educational content detection using spaCy features"""
        if self.nlp:
            doc = self.nlp(message.lower())
            
            # Use advanced linguistic analysis
            educational_score = self._calculate_educational_score(doc)
            return educational_score > 0.5
        else:
            # Fallback to simple detection
            educational_words = ['how', 'what', 'why', 'explain', 'tutorial', 'guide', 'learn', 'difference']
            message_lower = message.lower()
            return sum(1 for word in educational_words if word in message_lower) >= 2
    
    def _is_analysis_request(self, message: str) -> bool:
        """Enhanced analysis request detection"""
        if not message.strip():
            return True
        analysis_words = ['analyze', 'describe', 'what', 'tell', 'identify', 'see']
        return any(word in message.lower() for word in analysis_words)
    async def _fallback_detection(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        """Enhanced fallback detection with better pattern recognition"""
        message_lower = message.lower()
        word_count = len(message.split())
        
        # Enhanced model switching detection
        model_patterns = [
            r'(?i)(?:switch|change|use|select)\s+(?:to\s+)?(?:a\s+)?(?:different\s+)?model',
            r'(?i)(?:i\s+)?want\s+(?:to\s+)?(?:change|switch|use)\s+model',
            r'(?i)change\s+model\s*(?:bro|please)?',
            r'(?i)switch\s+model',
            r'(?i)model\s+(?:switch|change)'
        ]
        
        for pattern in model_patterns:
            if re.search(pattern, message):
                return CommandIntent.SWITCH_MODEL, 0.85
        
        # Quick keyword-based detection with improved scoring
        detection_map = [
            (['document', 'report', 'pdf', 'docx', 'article', 'paper', 'write', 'create'], CommandIntent.GENERATE_DOCUMENT, 0.8),
            (['image', 'picture', 'draw', 'photo', 'artwork', 'visual', 'paint'], CommandIntent.GENERATE_IMAGE, 0.8),
            (['tutorial', 'guide', 'explain', 'how', 'what', 'why', 'teach', 'learn'], CommandIntent.EDUCATIONAL, 0.7),
            (['export', 'save', 'download', 'backup'], CommandIntent.EXPORT_CHAT, 0.8),
            (['analyze', 'describe'] if has_attached_media else [], CommandIntent.ANALYZE, 0.9),
        ]
        
        for keywords, intent, confidence in detection_map:
            if keywords and any(word in message_lower for word in keywords):
                # Bonus for multiple keyword matches
                matches = sum(1 for word in keywords if word in message_lower)
                adjusted_confidence = min(confidence + (matches - 1) * 0.1, 1.0)
                return intent, adjusted_confidence
        
        # Enhanced educational detection for longer messages
        if word_count > 8:
            educational_patterns = [
                r'(?i)how\s+(?:to|do|does|can)',
                r'(?i)what\s+(?:is|are|does)',
                r'(?i)explain\s+(?:the\s+)?difference',
                r'(?i)(?:comprehensive|detailed)\s+(?:tutorial|guide)',
                r'(?i)step\s*-?\s*by\s*-?\s*step'
            ]
            
            for pattern in educational_patterns:
                if re.search(pattern, message):
                    return CommandIntent.EDUCATIONAL, 0.75
            
            # Check for question-like structure
            if any(word in message_lower for word in ['how', 'what', 'explain']):
                return CommandIntent.EDUCATIONAL, 0.6
        
        # Default classification
        if word_count > 5:
            return CommandIntent.CHAT, 0.5
        
        return CommandIntent.UNKNOWN, 0.0


class AICommandRouter:
    """
    Simplified AI command router using advanced intent detection
    Much cleaner and more maintainable than manual pattern matching
    """
    
    def __init__(self, command_handlers, gemini_api=None):
        self.command_handlers = command_handlers
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
          # Use enhanced intent detector
        self.intent_detector = EnhancedIntentDetector()

    async def detect_intent(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        """Detect user intent using advanced NLP"""
        return await self.intent_detector.detect_intent(message, has_attached_media)

    async def route_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, intent: CommandIntent, original_message: str) -> bool:
        """Route detected intent to appropriate handler with educational support"""
        try:
            # Streamlined handler mapping
            handlers = {
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
            
            if intent in handlers:
                # Generation commands need the original message for prompt extraction
                if intent in [CommandIntent.GENERATE_DOCUMENT, CommandIntent.GENERATE_IMAGE, CommandIntent.GENERATE_VIDEO]:
                    return await handlers[intent](update, context, original_message)
                else:
                    return await handlers[intent](update, context)
            
            # EDUCATIONAL, CHAT, and ANALYZE intents are handled by normal conversation flow
            elif intent in [CommandIntent.EDUCATIONAL, CommandIntent.CHAT, CommandIntent.ANALYZE]:
                return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error routing command for intent {intent}: {str(e)}")
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

    async def should_route_message(self, message: str, has_attached_media: bool = False) -> bool:
        """
        Enhanced routing decision with educational content awareness
        Returns True if message should be routed through command system
        """
        if len(message.strip()) < 5:
            return False
            
        intent, confidence = await self.detect_intent(message, has_attached_media)
        
        # Don't route EDUCATIONAL, CHAT, and ANALYZE intents - handle in normal conversation flow
        if intent in [CommandIntent.EDUCATIONAL, CommandIntent.CHAT, CommandIntent.ANALYZE]:
            return False
            
        return intent != CommandIntent.UNKNOWN and confidence > 0.4
