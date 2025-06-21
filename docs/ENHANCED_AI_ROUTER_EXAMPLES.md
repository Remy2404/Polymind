# 🚀 Enhanced AI Command Router - Usage Examples

The Enhanced AI Command Router provides sophisticated NLP capabilities using spaCy integration with AI-powered model recommendations. Here's how to use its powerful features:

## 🎯 Quick Start

### Run the Demo
```bash
# Quick demo (basic features)
python test_ai_router_features.py

# Full comprehensive demo (all features)
python examples/enhanced_ai_command_router_demo.py
```

## 🧠 Key Features

### 1. Advanced Intent Detection
```python
from src.services.ai_command_router import EnhancedIntentDetector

detector = EnhancedIntentDetector()
intent, confidence = await detector.detect_intent(
    "How to create a comprehensive Python tutorial?"
)
# Result: CommandIntent.EDUCATIONAL, confidence: 0.87
```

### 2. AI-Powered Model Recommendations
```python
from src.services.ai_command_router import AICommandRouter

router = AICommandRouter(command_handlers)
result = await router.detect_intent_with_recommendations(
    "Write a complex algorithm in Python"
)

print(f"Intent: {result.intent}")  # CODING
print(f"Recommended models: {result.recommended_models}")  
# ['deepcoder', 'olympiccoder-32b', 'devstral-small']
```

### 3. Linguistic Feature Extraction
```python
# Extract comprehensive linguistic features
result = await router.detect_intent_with_recommendations(message)

features = result.linguistic_features
print(f"Complexity score: {features['complexity_score']}")
print(f"Technical terms: {features['technical_terms']}")
print(f"Has questions: {features['questions']}")
```

## 🎓 Intent Types & Model Recommendations

| Intent Type | Trigger Examples | Recommended Models |
|-------------|------------------|-------------------|
| **EDUCATIONAL** | "How to...", "Explain...", "Tutorial on..." | `deepseek-r1`, `phi-4-reasoning-plus` |
| **CODING** | "Write code...", "Debug...", "Algorithm..." | `deepcoder`, `olympiccoder-32b` |
| **MATHEMATICAL** | "Solve equation...", "Calculate..." | `deepseek-prover-v2`, `phi-4-reasoning` |
| **CREATIVE** | "Write story...", "Creative writing..." | `deephermes-3-mistral-24b`, `qwerky-72b` |
| **MULTILINGUAL** | "Translate...", "Chinese text..." | `qwen3-235b`, `glm-z1-32b` |
| **VISION** | "Analyze image...", "Describe photo..." | `llama-3.2-11b-vision`, `qwen2.5-vl-72b` |
| **GENERATE_IMAGE** | "Draw...", "Create image..." | `gemini` (with image generation) |
| **GENERATE_DOCUMENT** | "Create report...", "Business plan..." | `gemini`, `deepseek`, `llama4-maverick` |

## 🔧 Advanced Usage

### Custom Model Selection
```python
# Get models by capability
vision_models = router._get_models_by_capability('supports_images')
coding_models = router._get_models_by_type('coding')
multilingual_models = router._get_multilingual_models()
```

### Educational Content Detection
```python
# Advanced educational markers detection
if router.nlp:
    doc = router.nlp(message)
    educational_score = router._calculate_educational_score(doc)
    print(f"Educational markers strength: {educational_score}")
```

### Entity Recognition
```python
result = await router.detect_intent_with_recommendations(message)
for entity in result.detected_entities:
    print(f"Entity: {entity['text']} ({entity['label']}) - {entity['type']}")
```

## 🌟 Real-World Examples

### Example 1: Educational Query
```python
message = "Can you explain the difference between supervised and unsupervised learning?"

result = await router.detect_intent_with_recommendations(message)
# Intent: EDUCATIONAL
# Confidence: 0.89
# Recommended: ['deepseek-r1', 'phi-4-reasoning-plus']
# Reasoning: "Question pattern detected; comparison structure; technical terms found"
```

### Example 2: Coding Task
```python
message = "Write a Python function to implement binary search with error handling"

result = await router.detect_intent_with_recommendations(message)
# Intent: CODING
# Confidence: 0.92
# Recommended: ['deepcoder', 'olympiccoder-32b', 'devstral-small']
# Reasoning: "Imperative structure detected; technical terms found (3)"
```

### Example 3: Complex Document Request
```python
message = "Generate a comprehensive business plan for an AI startup including market analysis"

result = await router.detect_intent_with_recommendations(message)
# Intent: GENERATE_DOCUMENT
# Confidence: 0.86
# Recommended: ['gemini', 'deepseek', 'llama4-maverick']
# Reasoning: "Document generation detected; high complexity content; business terms found"
```

## 📊 Performance Features

### Fallback Mechanisms
- ✅ Works without spaCy (basic pattern matching)
- ✅ Works without model configurations
- ✅ Graceful degradation of features

### Linguistic Analysis (when spaCy available)
- 🔤 POS tagging and dependency parsing
- 🏷️ Named entity recognition
- 📈 Complexity scoring
- 🌍 Language detection
- 🔧 Technical term counting

### Smart Model Selection
- 🎯 Intent-based recommendations
- 🔍 Capability filtering
- 🌐 Multilingual support detection
- 📊 Complexity-based selection

## 🔨 Integration with Your Bot

### Basic Integration
```python
from src.services.ai_command_router import AICommandRouter

# Initialize with your command handlers
router = AICommandRouter(command_handlers, gemini_api)

# In your message handler
async def handle_message(update, context):
    message = update.message.text
    
    # Check if should route through command system
    if await router.should_route_message(message):
        intent, confidence = await router.detect_intent(message)
        routed = await router.route_command(update, context, intent, message)
        if routed:
            return  # Command handled
    
    # Continue with normal conversation flow
    # Educational, chat, and analysis intents flow here
```

### Enhanced Integration with Model Recommendations
```python
async def handle_message_enhanced(update, context):
    message = update.message.text
    
    # Get full analysis
    result = await router.detect_intent_with_recommendations(message)
    
    # Log analysis for debugging
    logger.info(f"Intent: {result.intent.value} ({result.confidence:.3f})")
    logger.info(f"Recommended models: {result.recommended_models}")
    logger.info(f"Reasoning: {result.reasoning}")
    
    # Use recommended model for response
    if result.recommended_models:
        best_model = result.recommended_models[0]
        # Switch to best model for this type of request
        context.user_data['preferred_model'] = best_model
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

2. **Run Quick Test**:
   ```bash
   python test_ai_router_features.py
   ```

3. **Run Full Demo**:
   ```bash
   python examples/enhanced_ai_command_router_demo.py
   ```

4. **Integrate in Your Bot**:
   ```python
   router = AICommandRouter(your_command_handlers, your_gemini_api)
   ```

The Enhanced AI Command Router transforms your bot into an intelligent system that understands user intent with remarkable accuracy and automatically selects the best AI model for each task! 🤖✨
