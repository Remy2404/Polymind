# Intent Detection Migration Guide

## Current State
- ✅ Using: `ai_command_router.py` (original system)
- 📁 Available: `smart_intent_detector.py` (enhanced system)

## Key Differences

| Feature | ai_command_router.py | smart_intent_detector.py |
|---------|---------------------|-------------------------|
| Architecture | Separate detector + router | Combined class |
| Educational Detection | Basic (in CHAT intent) | ✅ Dedicated EDUCATIONAL intent |
| Error Handling | ✅ Robust fallback | Assumes spaCy available |
| Code Lines | 446 lines | 360 lines |
| Complexity | Higher | Lower |

## Migration Options

### Option 1: Keep Current (Safe)
```python
# No changes needed - current system works
from src.services.ai_command_router import AICommandRouter
```

### Option 2: Full Migration (Advanced)
```python
# Replace in message_handlers.py
from src.services.smart_intent_detector import SmartIntentDetector

# Change initialization
self.ai_command_router = SmartIntentDetector(command_handlers, gemini_api)
```

### Option 3: Hybrid (Recommended)
- Extract enhanced educational detection from smart_intent_detector.py
- Add it to ai_command_router.py
- Keep robust error handling

## Test Before Migration
```bash
# Test current system
python tests/test_intent_comparison.py

# Test enhanced system
python tests/test_spacy_intent.py
```

## Recommendation
- **For now**: Keep using `ai_command_router.py` (it works well)
- **Future**: Consider migrating for better educational detection
- **Best**: Create a hybrid combining the best of both
