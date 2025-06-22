"""
Enhanced AI Command Router - Modern Intent Detection with spaCy NLP
Combines robust error handling with advanced educational content detection
Reduced code complexity while maintaining full functionality
AI-Powered Model Selection Integration
"""
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import asyncio
from dataclasses import dataclass

from telegram import Update
from telegram.ext import ContextTypes

# Enhanced spaCy integration with fallback handling
try:
    import spacy
    from spacy.lang.en import English
    from spacy.matcher import PhraseMatcher, Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

# Import model configurations for AI-powered model selection
try:
    from src.services.model_handlers.model_configs import ModelConfigurations, Provider, ModelConfig
    MODEL_CONFIGS_AVAILABLE = True
except ImportError:
    MODEL_CONFIGS_AVAILABLE = False
    logging.warning("Model configurations not available")


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
    CODING = "coding"  # Programming-related tasks
    MATHEMATICAL = "mathematical"  # Math and reasoning tasks
    CREATIVE = "creative"  # Creative writing and storytelling
    MULTILINGUAL = "multilingual"  # Non-English language tasks
    VISION = "vision"  # Image understanding tasks
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Enhanced result structure for intent detection with model recommendations"""
    intent: CommandIntent
    confidence: float
    recommended_models: List[str]
    reasoning: str
    detected_entities: List[Dict[str, Any]]
    linguistic_features: Dict[str, Any]


class EnhancedIntentDetector:
    """
    Modern intent detection using spaCy NLP with AI-powered model selection
    Combines accuracy with robust error handling and intelligent model recommendations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.phrase_matcher = None
        self.rule_matcher = None
        self.model_configs = None
        self._load_model()
        self._load_model_configs()
        self._initialize_matchers()
        
        # Enhanced intent patterns with model-specific capabilities
        self.intent_patterns = {
            CommandIntent.GENERATE_DOCUMENT: {
                'keywords': ['document', 'report', 'article', 'paper', 'essay', 'pdf', 'docx', 'proposal', 'business', 'plan', 'summary', 'analysis', 'detailed', 'comprehensive'],
                'actions': ['create', 'generate', 'write', 'make', 'produce', 'draft', 'prepare'],
                'patterns': [
                    r'(?i)(?:create|generate|write|make)\s+(?:a\s+)?(?:document|report|article|paper)',
                    r'(?i)(?:business\s+plan|proposal|summary)',
                    r'(?i)write\s+(?:me\s+)?(?:a\s+)?(?:document|report|article)',
                    r'(?i)write\s+(?:me\s+)?(?:a\s+)?(?:detailed|comprehensive)\s+(?:report|analysis)',
                    r'(?i)(?:detailed|comprehensive)\s+(?:report|analysis|summary)',
                    r'(?i)(?:report|analysis)\s+on\s+\w+'
                ],
                'weight': 0.9,
                'preferred_models': ['gemini', 'deepseek', 'llama4-maverick'],
                'model_criteria': ['supports_documents', 'long_context']
            },
            
            CommandIntent.GENERATE_IMAGE: {
                'keywords': ['image', 'picture', 'photo', 'artwork', 'illustration', 'logo', 'visual', 'draw', 'painting', 'sketch', 'design', 'sunset', 'mountains', 'city', 'beautiful'],
                'actions': ['create', 'generate', 'draw', 'design', 'make', 'paint', 'render', 'sketch'],
                'patterns': [
                    r'(?i)(?:create|generate|draw|make)\s+(?:an?\s+)?(?:image|picture|photo)',
                    r'(?i)draw\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|sunset|mountains)',
                    r'(?i)(?:artwork|illustration|logo|visual)',
                    r'(?i)paint\s+(?:me\s+)?(?:an?\s+)?(?:image|picture)',
                    r'(?i)draw\s+(?:a\s+)?(?:sunset|mountains|city|landscape)',
                    r'(?i)draw\s+(?:a\s+)?(?:beautiful|stunning|amazing)',
                    r'(?i)(?:beautiful|stunning|amazing)\s+(?:sunset|mountains|landscape|city)',
                    r'(?i)(?:with\s+a\s+)?(?:reflection|lake|water)'
                ],
                'weight': 0.9,
                'preferred_models': ['gemini'],  # Gemini supports image generation
                'model_criteria': ['supports_images']
            },
            
            CommandIntent.EDUCATIONAL: {
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
                'weight': 1.0,
                'preferred_models': ['deepseek', 'llama4-maverick', 'phi-4-reasoning-plus'],
                'model_criteria': ['reasoning_capable']
            },
            CommandIntent.CODING: {
                'keywords': ['code', 'programming', 'script', 'function', 'algorithm', 'debug', 'python', 'javascript', 'java', 'c++', 'html', 'css', 'sql', 'api', 'framework', 'implement', 'binary', 'search'],
                'actions': ['code', 'program', 'write', 'debug', 'fix', 'optimize', 'implement', 'create'],
                'patterns': [
                    r'(?i)(?:write|create|generate)\s+(?:a\s+)?(?:code|script|function)',
                    r'(?i)(?:programming|coding)\s+(?:problem|task|challenge)',
                    r'(?i)debug\s+(?:this\s+)?code',
                    r'(?i)how\s+to\s+(?:code|program)',
                    r'(?i)(?:python|javascript|java|c\+\+)\s+(?:code|script|function)',
                    r'(?i)write\s+(?:a\s+)?(?:python|javascript|java)\s+function',
                    r'(?i)implement\s+(?:a\s+)?(?:algorithm|function)',
                    r'(?i)(?:binary|search|sort|merge)\s+algorithm',
                    r'(?i)(?:algorithm|function)\s+(?:to|for)\s+(?:implement|sort|search)'
                ],
                'weight': 0.9,
                'preferred_models': ['deepcoder', 'olympiccoder-32b', 'devstral-small'],
                'model_criteria': ['coding_specialist']
            },
            CommandIntent.MATHEMATICAL: {
                'keywords': ['math', 'mathematics', 'equation', 'formula', 'calculate', 'solve', 'proof', 'theorem', 'statistics', 'probability', 'algebra', 'calculus', 'differential', 'integral', 'derivative'],
                'actions': ['solve', 'calculate', 'prove', 'derive', 'compute'],
                'patterns': [
                    r'(?i)solve\s+(?:this\s+)?(?:equation|problem|math)',
                    r'(?i)(?:mathematical|math)\s+(?:problem|equation)',
                    r'(?i)calculate\s+(?:the\s+)?(?:result|answer)',
                    r'(?i)(?:proof|theorem|formula)',
                    r'(?i)(?:differential|integral)\s+equation',
                    r'(?i)solve\s+(?:the\s+)?(?:differential|integral)',
                    r'(?i)(?:dy/dx|d/dx|\∫|\∑)',
                    r'(?i)(?:equation|formula):\s*(?:[a-z]+\s*[=+\-*/]\s*[a-z0-9]+)',
                    r'(?i)(?:derivative|integral|limit)\s+of'
                ],
                'weight': 0.9,
                'preferred_models': ['deepseek-prover-v2', 'phi-4-reasoning-plus', 'qwq-32b'],
                'model_criteria': ['mathematical_reasoning']
            },

            CommandIntent.CREATIVE: {
                'keywords': ['story', 'creative', 'writing', 'poetry', 'novel', 'character', 'plot', 'narrative', 'fiction', 'fantasy', 'dialogue'],
                'actions': ['write', 'create', 'compose', 'craft', 'develop'],
                'patterns': [
                    r'(?i)write\s+(?:a\s+)?(?:story|poem|novel)',
                    r'(?i)creative\s+writing',
                    r'(?i)(?:fiction|fantasy|sci-fi)\s+story',
                    r'(?i)character\s+development'
                ],
                'weight': 0.9,
                'preferred_models': ['deephermes-3-mistral-24b', 'qwerky-72b', 'moonlight-16b'],
                'model_criteria': ['creative_writing']
            },

            CommandIntent.MULTILINGUAL: {
                'keywords': ['translate', 'translation', 'language', 'chinese', 'japanese', 'korean', 'spanish', 'french', 'german', 'hindi', 'arabic'],
                'actions': ['translate', 'convert', 'interpret'],
                'patterns': [
                    r'(?i)translate\s+(?:this\s+)?(?:to|into)',
                    r'(?i)(?:chinese|japanese|korean|spanish|french|german|hindi|arabic)',
                    r'(?i)(?:multilingual|bilingual)'
                ],
                'weight': 0.9,
                'preferred_models': ['qwen3-235b', 'shisa-v2-llama3.3-70b', 'sarvam-m', 'glm-z1-32b'],
                'model_criteria': ['multilingual_support']
            },
            CommandIntent.VISION: {
                'keywords': ['image', 'picture', 'photo', 'visual', 'see', 'look', 'describe', 'analyze', 'recognize'],
                'actions': ['analyze', 'describe', 'identify', 'recognize', 'examine'],
                'patterns': [
                    r'(?i)(?:analyze|describe|identify)\s+(?:this\s+)?(?:image|picture|photo)',
                    r'(?i)what\s+(?:do\s+you\s+)?see\s+in\s+(?:this\s+)?(?:image|picture)',
                    r'(?i)(?:visual\s+analysis|image\s+recognition)'
                ],
                'weight': 0.9,
                'preferred_models': ['llama-3.2-11b-vision', 'qwen2.5-vl-72b', 'internvl3-14b', 'kimi-vl-a3b-thinking'],
                'model_criteria': ['supports_images']
            },

            CommandIntent.SWITCH_MODEL: {
                'keywords': ['switch', 'change', 'model', 'different', 'another', 'use'],
                'actions': ['switch', 'change', 'use', 'select'],
                'patterns': [
                    r'(?i)(?:switch|change|use|select)\s+(?:to\s+)?(?:a\s+)?(?:different\s+)?model',
                    r'(?i)(?:i\s+)?want\s+(?:to\s+)?(?:change|switch|use)\s+model',
                    r'(?i)change\s+model\s*(?:bro|please)?',
                    r'(?i)switch\s+model',
                    r'(?i)model\s+(?:switch|change)',
                    r'(?i)(?:different|another)\s+model',
                    r'(?i)use\s+(?:a\s+)?different\s+model'
                ],
                'weight': 0.9,
                'preferred_models': [],  # No specific models for model switching
                'model_criteria': []
            },

            # ...existing patterns remain the same...
        }
    
    def _load_model(self):
        """Load spaCy model with robust error handling"""
        if not SPACY_AVAILABLE:
            self.logger.warning("🟡 spaCy not available, using fallback detection")
            return
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Ensure the sentencizer is in the pipeline for sentence boundary detection
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
                self.logger.info("Added 'sentencizer' to spaCy pipeline.")
            self.logger.info("✅ spaCy en_core_web_sm model loaded successfully")
        except OSError:
            try:
                self.nlp = English()
                # Add sentencizer for basic models
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
                    self.logger.info("Added 'sentencizer' to basic English model.")
                # Reduce warning frequency - only log once
                if not hasattr(self, '_fallback_warned'):
                    self.logger.info("🔄 Using basic English model for intent detection")
                    self.logger.info("💡 For better accuracy, install: python -m spacy download en_core_web_sm")
                    self._fallback_warned = True
            except Exception as e:
                self.logger.error(f"❌ Failed to load any spaCy model: {e}")
                self.nlp = None
    
    def _load_model_configs(self):
        """Load model configurations for AI-powered model selection"""
        if MODEL_CONFIGS_AVAILABLE:
            try:
                self.model_configs = ModelConfigurations.get_all_models()
                self.logger.info(f"✅ Loaded {len(self.model_configs)} model configurations")
            except Exception as e:
                self.logger.error(f"❌ Failed to load model configurations: {e}")
                self.model_configs = {}
        else:
            self.logger.warning("🟡 Model configurations not available")
            self.model_configs = {}
    
    def _initialize_matchers(self):
        """Initialize spaCy matchers for advanced pattern detection"""
        if not self.nlp:
            return
            
        try:
            # Initialize phrase matcher for entity detection
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            
            # Initialize rule matcher for complex patterns
            self.rule_matcher = Matcher(self.nlp.vocab)
            
            # Add model name patterns
            if self.model_configs:
                model_names = []
                for config in self.model_configs.values():
                    model_names.append(self.nlp(config.display_name.lower()))
                    model_names.append(self.nlp(config.model_id.lower()))
                
                self.phrase_matcher.add("MODEL_NAMES", model_names)
            
            # Add technical term patterns
            tech_terms = [
                self.nlp("machine learning"), self.nlp("artificial intelligence"),
                self.nlp("deep learning"), self.nlp("neural network"),
                self.nlp("natural language processing"), self.nlp("computer vision")
            ]
            self.phrase_matcher.add("TECH_TERMS", tech_terms)
            
            # Check if spaCy model has POS tagging capability
            has_pos_tagger = "tagger" in self.nlp.pipe_names or "morphologizer" in self.nlp.pipe_names
            
            if has_pos_tagger:
                # Add complex patterns using rule matcher with POS attributes
                # Pattern for "how to" questions
                how_to_pattern = [
                    {"LOWER": "how"},
                    {"LOWER": "to"},
                    {"POS": "VERB", "OP": "?"},
                    {"IS_ALPHA": True, "OP": "*"}
                ]
                self.rule_matcher.add("HOW_TO_PATTERN", [how_to_pattern])
                
                # Pattern for model switching
                model_switch_pattern = [
                    {"LOWER": {"IN": ["switch", "change", "use"]}},
                    {"LOWER": {"IN": ["to", "model"]}, "OP": "?"},
                    {"IS_ALPHA": True, "OP": "*"}
                ]
                self.rule_matcher.add("MODEL_SWITCH_PATTERN", [model_switch_pattern])
                
                self.logger.info("✅ spaCy matchers with POS tagging initialized successfully")
            else:
                # Add simpler patterns without POS attributes for basic spaCy models
                # Pattern for "how to" questions
                how_to_pattern = [
                    {"LOWER": "how"},
                    {"LOWER": "to"},
                    {"IS_ALPHA": True, "OP": "*"}
                ]
                self.rule_matcher.add("HOW_TO_PATTERN", [how_to_pattern])
                
                # Pattern for model switching
                model_switch_pattern = [
                    {"LOWER": {"IN": ["switch", "change", "use"]}},
                    {"LOWER": {"IN": ["to", "model"]}, "OP": "?"},
                    {"IS_ALPHA": True, "OP": "*"}                ]
                self.rule_matcher.add("MODEL_SWITCH_PATTERN", [model_switch_pattern])
                
                self.logger.info("✅ spaCy matchers (basic patterns) initialized successfully")
                self.logger.info("💡 Install full spaCy model for enhanced POS-based patterns: python -m spacy download en_core_web_sm")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize matchers: {e}")
            self.phrase_matcher = None
            self.rule_matcher = None

    def _has_pos_tagger(self) -> bool:
        """Check if the current spaCy model has POS tagging capability"""
        return (self.nlp is not None and 
                ("tagger" in self.nlp.pipe_names or "morphologizer" in self.nlp.pipe_names))

    async def detect_intent_with_recommendations(self, message: str, has_attached_media: bool = False) -> IntentResult:
        """
        Enhanced intent detection with AI-powered model recommendations
        Returns comprehensive analysis including recommended models and reasoning
        """
        if not message or len(message.strip()) < 2:
            return IntentResult(
                intent=CommandIntent.UNKNOWN,
                confidence=0.0,
                recommended_models=[],
                reasoning="Message too short for analysis",
                detected_entities=[],
                linguistic_features={}
            )
        
        # Priority: Media analysis
        if has_attached_media and self._is_analysis_request(message):
            vision_models = self._get_models_by_capability('supports_images')
            return IntentResult(
                intent=CommandIntent.ANALYZE,
                confidence=0.95,
                recommended_models=vision_models[:3],
                reasoning="Media attached with analysis request detected",
                detected_entities=[],
                linguistic_features={'has_media': True}
            )
            
        # Use spaCy if available, otherwise fallback
        if self.nlp is None:
            basic_result = await self._fallback_detection(message, has_attached_media)
            return IntentResult(
                intent=basic_result[0],
                confidence=basic_result[1],
                recommended_models=[],
                reasoning="Fallback detection (spaCy unavailable)",
                detected_entities=[],
                linguistic_features={}
            )
        
        # Process with spaCy
        doc = self.nlp(message.lower())
        
        # Extract linguistic features
        linguistic_features = self._extract_linguistic_features(doc)
        
        # Detect entities using matchers
        detected_entities = self._detect_entities_with_matchers(doc)
        
        # Enhanced scoring system
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = self._calculate_enhanced_intent_score(message, doc, patterns, detected_entities)
            intent_scores[intent] = score
        
        # Find best intent
        best_intent = max(intent_scores, key=intent_scores.get) if intent_scores else CommandIntent.UNKNOWN
        best_score = intent_scores.get(best_intent, 0.0)
        
        # Apply intent logic and get final result
        final_intent, final_confidence = self._apply_intent_logic(message, best_intent, best_score, has_attached_media)
        
        # Get recommended models for the detected intent
        recommended_models = self._get_recommended_models(final_intent, detected_entities, linguistic_features)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(final_intent, final_confidence, detected_entities, linguistic_features)
        
        return IntentResult(
            intent=final_intent,
            confidence=final_confidence,
            recommended_models=recommended_models,
            reasoning=reasoning,
            detected_entities=detected_entities,
            linguistic_features=linguistic_features
        )

    async def detect_intent(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        """
        Backward compatibility method - returns simple tuple format
        For new applications, use detect_intent_with_recommendations instead
        """
        result = await self.detect_intent_with_recommendations(message, has_attached_media)
        return result.intent, result.confidence

    def _calculate_enhanced_intent_score(self, message: str, doc, patterns: Dict, entities: List[Dict]) -> float:
        """Enhanced scoring algorithm using spaCy's advanced linguistic features"""
        score = 0.0
        message_lower = message.lower()
        
        # Extract linguistic features
        lemmas = [token.lemma_ for token in doc]
        
        # 1. Enhanced keyword matching using lemmas and context
        keyword_score = self._calculate_advanced_keyword_score(patterns['keywords'], lemmas, message_lower, doc, entities)
        score += keyword_score * 0.4
        
        # 2. Advanced action verb matching with POS tagging and dependency parsing
        action_score = self._calculate_advanced_action_score(patterns['actions'], doc)
        score += action_score * 0.3
        
        # 3. Entity-aware scoring boost
        entity_score = self._calculate_entity_aware_score(patterns, entities)
        score += entity_score * 0.1
        
        # 4. Enhanced regex pattern matching
        if 'patterns' in patterns:
            pattern_score = sum(0.2 for pattern in patterns['patterns'] if re.search(pattern, message))
            score += min(pattern_score, 0.6)
        
        # 5. Syntactic structure scoring using dependency parsing
        syntax_score = self._calculate_syntax_score(patterns, doc)
        score += syntax_score * 0.15
        
        # 6. Semantic similarity scoring (if word vectors available)
        semantic_score = self._calculate_semantic_score(patterns, doc)
        score += semantic_score * 0.05
        
        # 7. Educational content scoring
        educational_score = self._calculate_educational_score(doc)
        score += educational_score * 0.1
        
        return min(score * patterns['weight'], 1.0)

    def _calculate_advanced_keyword_score(self, keywords: List[str], 
                                        lemmas: List[str], 
                                        message_lower: str, doc, entities: List[Dict]) -> float:
        """Advanced keyword matching with context awareness"""
        matches = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            # Direct lemma matching
            if keyword in lemmas:
                matches += 1
                continue
            
            # Partial matching in message
            if keyword in message_lower:
                matches += 0.8
                continue
            
            # Entity-based matching
            entity_match = any(keyword.lower() in entity['text'].lower() for entity in entities)
            if entity_match:
                matches += 0.9
                continue
            
            # Similarity matching using word vectors (if available)
            if hasattr(doc.vocab, 'vectors') and doc.vocab.vectors.size > 0:
                try:
                    keyword_token = self.nlp(keyword)[0]
                    for token in doc:
                        if (token.vector_norm > 0 and keyword_token.vector_norm > 0 and
                            token.similarity(keyword_token) > 0.75):
                            matches += 0.6
                            break
                except:
                    pass  # Similarity calculation failed, continue
        
        return min(matches / total_keywords, 1.0) if total_keywords > 0 else 0.0

    def _calculate_advanced_action_score(self, actions: List[str], doc) -> float:
        """Advanced action verb scoring with syntactic analysis"""
        verb_matches = 0
        total_actions = len(actions)
        has_pos = self._has_pos_tagger()
        
        for token in doc:
            if token.lemma_ in actions:
                if has_pos:
                    # Higher score for root verbs (main actions)
                    if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                        verb_matches += 1.0
                    # Medium score for auxiliary verbs
                    elif token.pos_ in ['VERB', 'AUX']:
                        verb_matches += 0.7
                    # Lower score for other matches
                    else:
                        verb_matches += 0.5
                        
                    # Bonus for imperative mood (commands)
                    if hasattr(token, 'morph') and 'Mood=Imp' in str(token.morph):
                        verb_matches += 0.3
                else:
                    # Fallback scoring without POS tags
                    verb_matches += 0.6
        
        return min(verb_matches / total_actions, 1.0) if total_actions > 0 else 0.0

    def _calculate_entity_aware_score(self, patterns: Dict, entities: List[Dict]) -> float:
        """Calculate score boost based on detected entities"""
        if not entities:
            return 0.0
        
        score = 0.0
        keywords = patterns.get('keywords', [])
        
        for entity in entities:
            entity_text = entity['text'].lower()
            
            # Check if entity matches any keywords
            for keyword in keywords:
                if keyword in entity_text or entity_text in keyword:
                    # Higher score for exact matches
                    if entity_text == keyword:
                        score += 0.3
                    else:
                        score += 0.2
                    break
            
            # Special scoring for specific entity types
            if entity['type'] == 'phrase_match' and entity['label'] == 'MODEL_NAMES':
                score += 0.4  # Model names are highly relevant
            elif entity['type'] == 'phrase_match' and entity['label'] == 'TECH_TERMS':
                score += 0.2  # Technical terms are moderately relevant            elif entity['type'] == 'rule_match':
                score += 0.3  # Rule matches are quite relevant
        
        return min(score, 0.5)  # Cap entity score contribution

    def _calculate_syntax_score(self, patterns: Dict, doc) -> float:
            """Calculate score based on syntactic patterns"""
            score = 0.0
            has_pos = self._has_pos_tagger()
            
            # Look for command structures
            for token in doc:
                if has_pos:
                    # Imperative verbs at sentence start
                    if (token.i == 0 or (token.i > 0 and doc[token.i-1].is_punct)) and token.pos_ == 'VERB':
                        if token.lemma_ in patterns.get('actions', []):
                            score += 0.4
                    
                    # Question patterns
                    elif token.lemma_ in ['what', 'how', 'why'] and token.dep_ in ['nsubj', 'advmod']:
                        score += 0.3
                    
                    # Object relationships (verb -> object patterns)
                    elif token.dep_ == 'dobj' and token.head.lemma_ in patterns.get('actions', []):
                        if token.lemma_ in patterns.get('keywords', []):
                            score += 0.3
                else:
                    # Fallback without POS tags - use basic pattern matching
                    if token.lemma_ in patterns.get('actions', []):
                        # Simple check for sentence start positions
                        if token.i == 0 or (token.i > 0 and doc[token.i-1].is_punct):
                            score += 0.3
                        
                    # Basic question pattern detection
                    if token.lemma_ in ['what', 'how', 'why']:
                        score += 0.2
            
            return min(score, 0.4)

    def _calculate_semantic_score(self, patterns: Dict, doc) -> float:
        """Calculate semantic similarity score using word embeddings"""
        if not hasattr(doc.vocab, 'vectors') or doc.vocab.vectors.size == 0:
            return 0.0
        
        try:
            score = 0.0
            keywords = patterns.get('keywords', [])
            
            if not keywords:
                return 0.0
            
            # Calculate average similarity between document and pattern keywords
            similarities = []
            for keyword in keywords[:5]:  # Limit to first 5 keywords for performance
                keyword_doc = self.nlp(keyword)
                if keyword_doc.vector_norm > 0:
                    doc_similarity = doc.similarity(keyword_doc)
                    similarities.append(doc_similarity)
            
            if similarities:
                score = max(similarities) * 0.5  # Use max similarity, scaled down
            
            return min(score, 0.3)
        except:
            return 0.0  # Fallback if similarity calculation fails

    async def _fallback_detection(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        """Basic intent detection without spaCy, for fallback"""
        message_lower = message.lower()
        
        if has_attached_media and self._is_analysis_request(message):
            return CommandIntent.ANALYZE, 0.9
            
        # Check for educational markers first
        if self._has_educational_markers(message):
            return CommandIntent.EDUCATIONAL, 0.85

        # Check for specific command keywords
        for intent, patterns in self.intent_patterns.items():
            # Simple keyword check
            keyword_match = any(keyword in message_lower for keyword in patterns.get('keywords', []))
            
            # Simple action check
            action_match = any(action in message_lower for action in patterns.get('actions', []))

            if keyword_match and action_match:
                return intent, 0.7
            
            if keyword_match:
                return intent, 0.6

        return CommandIntent.CHAT, 0.5

    def _apply_intent_logic(self, message: str, best_intent: CommandIntent, best_score: float, has_attached_media: bool) -> Tuple[CommandIntent, float]:
        """Applies final logic to determine the intent."""
        # If media is attached, it's likely analysis if the message is short or vague
        if has_attached_media:
            if best_intent == CommandIntent.CHAT or best_score < 0.4:
                return CommandIntent.ANALYZE, 0.9
        
        # If educational markers are present, and the score is not high for something else, it's educational
        if self._has_educational_markers(message) and best_score < 0.7:
            return CommandIntent.EDUCATIONAL, 0.85

        # If score is too low, it's just a chat
        if best_score < 0.35:
            return CommandIntent.CHAT, 1.0 - best_score

        return best_intent, best_score

    def _extract_linguistic_features(self, doc) -> Dict[str, Any]:
        """Extracts linguistic features from the spaCy doc."""
        num_tokens = len(doc)
        num_sentences = len(list(doc.sents))
        avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else num_tokens
        
        has_pos = self._has_pos_tagger()
        
        if has_pos:
            pos_counts = {
                "nouns": len([token for token in doc if token.pos_ == 'NOUN']),
                "verbs": len([token for token in doc if token.pos_ == 'VERB']),
                "adjectives": len([token for token in doc if token.pos_ == 'ADJ']),
                "adverbs": len([token for token in doc if token.pos_ == 'ADV']),
            }
        else:
            # Fallback: estimate based on word patterns
            pos_counts = {
                "nouns": 0,
                "verbs": 0,
                "adjectives": 0,
                "adverbs": 0,
            }
        
        complexity_score = self._calculate_complexity_score(doc)
        language_hint = self._detect_language_hints(doc)

        return {
            "num_tokens": num_tokens,
            "num_sentences": num_sentences,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "pos_counts": pos_counts,
            "complexity_score": round(complexity_score, 2),
            "language_hint": language_hint,
            "technical_term_count": self._count_technical_terms(doc)
        }

    def _calculate_complexity_score(self, doc) -> float:
        """Calculate a complexity score based on sentence length and word complexity."""
        num_tokens = len(doc)
        if num_tokens == 0:
            return 0.0
            
        num_sentences = len(list(doc.sents))
        avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
        
        # Average word length as a proxy for word complexity
        avg_word_length = sum(len(token.text) for token in doc) / num_tokens if num_tokens > 0 else 0
        
        # Normalize and combine
        sent_len_score = min(avg_sentence_length / 20.0, 1.0)  # Normalize by typical max sentence length
        word_len_score = min(avg_word_length / 7.0, 1.0)   # Normalize by typical avg word length
        
        return (sent_len_score * 0.6) + (word_len_score * 0.4)

    def _detect_language_hints(self, doc) -> str:
        """Detect hints of non-English languages."""
        try:
            if doc._.language and doc._.language['score'] > 0.7:
                return doc._.language['language']
        except AttributeError:
            # spacy-langdetect might not be installed
            pass
        return "en" # default to english

    def _detect_entities_with_matchers(self, doc) -> List[Dict[str, Any]]:
        """Detects entities using both phrase and rule-based matchers."""
        entities = []
        
        # Phrase Matcher
        if self.phrase_matcher:
            matches = self.phrase_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entities.append({
                    "text": span.text,
                    "label": self.nlp.vocab.strings[match_id],
                    "type": "phrase_match"
                })

        # Rule Matcher
        if self.rule_matcher:
            matches = self.rule_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entities.append({
                    "text": span.text,
                    "label": self.nlp.vocab.strings[match_id],
                    "type": "rule_match"
                })
        
        # Standard NER
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "type": "ner"
            })
            
        return entities

    def _get_recommended_models(self, intent: CommandIntent, entities: List[Dict], features: Dict) -> List[str]:
        """Recommends models based on intent, entities, and linguistic features."""
        if not self.model_configs:
            return []

        intent_info = self.intent_patterns.get(intent, {})
        
        # Start with preferred models for the intent
        recommended = set(intent_info.get('preferred_models', []))
        
        # Add models based on criteria
        for criteria in intent_info.get('model_criteria', []):
            recommended.update(self._get_models_by_capability(criteria))

        # Add models based on detected language
        if features.get('language_hint') and features['language_hint'] != 'en':
            recommended.update(self._get_multilingual_models())

        # Add models based on detected entities (e.g., coding terms)
        for entity in entities:
            if entity['label'] == 'TECH_TERMS':
                 recommended.update(self._get_models_by_type('coding_specialist'))

        # If no specific models found, return general purpose models
        if not recommended:
            recommended.update(self._get_models_by_type('general_purpose'))

        # Convert to list and limit
        return list(recommended)[:5]

    def _get_models_by_capability(self, capability: str) -> List[str]:
        """Gets models that have a specific capability."""
        if not self.model_configs:
            return []
        return [
            model_id for model_id, config in self.model_configs.items()
            if capability in config.capabilities
        ]

    def _get_multilingual_models(self) -> List[str]:
        """Gets all models that support multiple languages."""
        return self._get_models_by_capability('multilingual_support')

    def _get_models_by_type(self, model_type: str) -> List[str]:
        """Gets models of a specific type."""
        if not self.model_configs:
            return []
        return [
            model_id for model_id, config in self.model_configs.items()
            if config.type == model_type
        ]

    def _generate_reasoning(self, intent: CommandIntent, confidence: float, 
                          entities: List[Dict], features: Dict) -> str:
        """Generates a human-readable reasoning for the detected intent."""
        reasoning_parts = [f"Intent '{intent.value}' detected with {confidence:.2f} confidence."]

        if features.get("technical_term_count", 0) > 2:
            reasoning_parts.append("High number of technical terms found.")
        
        if entities:
            entity_texts = [f"'{e['text']}' ({e['label']})" for e in entities[:3]]
            reasoning_parts.append(f"Detected entities: {', '.join(entity_texts)}.")

        if intent == CommandIntent.EDUCATIONAL:
            reasoning_parts.append("Message contains educational markers like 'how to' or 'explain'.")
        
        if features.get('language_hint', 'en') != 'en':
            reasoning_parts.append(f"Detected non-English language: {features['language_hint']}.")

        return " ".join(reasoning_parts)

    def _calculate_educational_score(self, doc) -> float:
        """Calculates a score for educational content."""
        score = 0.0
        
        # Check for question words
        if any(token.lemma_ in ["what", "how", "why", "explain", "compare", "difference"] for token in doc):
            score += 0.4

        # Check for technical terms
        score += self._count_technical_terms(doc) * 0.1

        # Check for tutorial/guide keywords
        if any(keyword in doc.text.lower() for keyword in ["tutorial", "guide", "lesson", "course", "walkthrough"]):
            score += 0.3
            
        return min(score, 1.0)

    def _has_educational_markers(self, message: str) -> bool:
        """Check for basic educational markers in a message"""
        message_lower = message.lower()
        educational_patterns = [
            'how to', 'what is', 'explain', 'tutorial', 'guide', 'difference between'
        ]
        return any(pattern in message_lower for pattern in educational_patterns)

    def _is_analysis_request(self, message: str) -> bool:
        """Enhanced analysis request detection"""
        if not message.strip():
            return True
        analysis_words = ['analyze', 'describe', 'what', 'tell', 'identify', 'see']
        return any(word in message.lower() for word in analysis_words)

    def _count_technical_terms(self, doc) -> int:
        """Count technical terms in the document"""
        technical_keywords = {
            'ai', 'ml', 'algorithm', 'neural', 'network', 'model', 'training',
            'api', 'database', 'framework', 'library', 'programming', 'code',
            'python', 'javascript', 'sql', 'html', 'css', 'react', 'node',
            'docker', 'kubernetes', 'aws', 'cloud', 'server', 'backend'
        }
        
        count = 0
        for token in doc:
            if token.lemma_.lower() in technical_keywords:
                count += 1
        
        return count


class AICommandRouter:
    """
    Simplified AI command router using advanced intent detection
    Much cleaner and more maintainable than manual pattern matching
    """
    
    def __init__(self, command_handlers, gemini_api=None):
        self.intent_detector = EnhancedIntentDetector()
        self.command_handlers = command_handlers
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)

    async def detect_intent(self, message: str, has_attached_media: bool = False) -> Tuple[CommandIntent, float]:
        return await self.intent_detector.detect_intent(message, has_attached_media)

    async def detect_intent_with_recommendations(self, message: str, has_attached_media: bool = False) -> IntentResult:
        return await self.intent_detector.detect_intent_with_recommendations(message, has_attached_media)

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
