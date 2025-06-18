# Enhanced Voice Recognition System for Telegram Bot

This document describes the enhanced voice recognition system that supports multiple speech-to-text engines for better accuracy and multilingual support.

## 🚀 Features

### Multiple Speech Recognition Engines
- **OpenAI Whisper**: State-of-the-art multilingual speech recognition
- **Faster-Whisper**: Optimized version of Whisper for better performance  
- **Vosk**: Offline speech recognition with good accuracy
- **Google Speech Recognition**: Cloud-based recognition (fallback)

### Advanced Capabilities
- ✅ **Automatic Engine Selection**: Chooses the best engine based on language and availability
- ✅ **Voice Activity Detection (VAD)**: Filters out silent or non-speech audio
- ✅ **Multilingual Support**: 100+ languages supported across different engines
- ✅ **Quality Settings**: Low/Medium/High quality modes for speed vs accuracy trade-offs
- ✅ **Language-Specific Preprocessing**: Optimized audio processing for different languages
- ✅ **Confidence Scoring**: Reliability metrics for transcriptions
- ✅ **Fallback Mechanisms**: Automatic engine switching on failures
- ✅ **Performance Benchmarking**: Compare engines on your audio samples

## 📦 Installation

### 1. Install Dependencies

The enhanced voice processor requires several additional packages:

```bash
# Install the enhanced dependencies
pip install openai-whisper faster-whisper vosk torch transformers soundfile librosa webrtcvad
```

Or if you're using the updated `pyproject.toml`:

```bash
pip install -e .
```

### 2. Download Models (Optional)

For offline processing, you may want to pre-download models:

```python
# Download Whisper models
import whisper
whisper.load_model("base")    # ~140MB
whisper.load_model("large-v3") # ~1.5GB

# Download Vosk models
import vosk
vosk.Model(lang="en")  # Downloads automatically
```

## 🎯 Quick Start

### Basic Usage

```python
from src.services.media.voice_processor import create_voice_processor, SpeechEngine

# Create processor with automatic engine selection
processor = await create_voice_processor(engine=SpeechEngine.AUTO)

# Process a voice file
text, language, metadata = await processor.transcribe("audio.wav")
print(f"Transcribed: '{text}' using {metadata['engine']}")
```

### Telegram Bot Integration

```python
async def handle_voice_message(update, context):
    """Handle voice messages in Telegram bot"""
    voice_file = await update.message.voice.get_file()
    user_id = str(update.effective_user.id)
    
    # Download and convert voice file
    ogg_path, wav_path = await processor.download_and_convert(
        voice_file, user_id
    )
    
    # Get best transcription
    text, language, metadata = await processor.get_best_transcription(
        wav_path, language="en-US", confidence_threshold=0.7
    )
    
    if text.strip():
        # Send transcription back to user
        engine = metadata.get('engine', 'unknown')
        confidence = metadata.get('confidence', 0.0)
        
        response = f"🎤 **Voice Message:**\n\n{text}"
        if confidence > 0:
            response += f"\n\n_Engine: {engine}, Confidence: {confidence:.1%}_"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    else:
        await update.message.reply_text(
            "❌ Sorry, I couldn't understand the voice message."
        )
```

## 🔧 Configuration

### Engine Selection

```python
from src.services.media.voice_processor import SpeechEngine

# Specific engine
processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)

# Automatic selection (recommended)
processor = await create_voice_processor(engine=SpeechEngine.AUTO)
```

### Quality Settings

```python
from src.services.media.voice_config import VoiceQuality, VoiceConfig

# Different quality levels
model_size = VoiceConfig.get_model_size(VoiceQuality.HIGH)  # "large-v3"
text, lang, metadata = await processor.transcribe(
    audio_file, model_size=model_size
)
```

### Language-Specific Processing

```python
# Khmer language with specific preprocessing
text, lang, metadata = await processor.transcribe(
    audio_file, 
    language="km-KH", 
    is_khmer=True
)

# Chinese with high-quality model
text, lang, metadata = await processor.transcribe(
    audio_file, 
    language="zh-CN", 
    engine=SpeechEngine.WHISPER,
    model_size="large-v3"
)
```

## 🌍 Supported Languages

### Whisper & Faster-Whisper (100+ languages)
`en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi, fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la, mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy, ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be, tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl, mg, as, tt, haw, ln, ha, ba, jw, su`

### Vosk (23+ languages)
`en, en-in, zh, ru, fr, de, es, pt, tr, vn, it, nl, ca, ar, fa, ph, uk, kz, ja, eo, hi, cs, pl`

### Google Speech Recognition (120+ languages)
All major world languages supported

## ⚡ Performance Comparison

| Engine | Speed | Accuracy | Offline | Languages | Memory |
|--------|-------|----------|---------|-----------|---------|
| **Faster-Whisper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | 100+ | ~500MB |
| **Whisper** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | 100+ | ~1GB |
| **Vosk** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | 23+ | ~200MB |
| **Google** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | 120+ | ~10MB |

## 🔄 Advanced Features

### Multi-Engine Comparison

```python
# Compare results from all engines
results = await processor.transcribe_with_multiple_engines(audio_file)

for engine, (text, lang, metadata) in results.items():
    confidence = metadata.get('confidence', 0.0)
    print(f"{engine:15}: '{text}' (confidence: {confidence:.2f})")
```

### Benchmarking

```python
# Benchmark engines on your audio
benchmark = await processor.benchmark_engines(audio_file)

for engine, metrics in benchmark.items():
    print(f"{engine}: {metrics['processing_time']:.2f}s, "
          f"success: {metrics['success']}")
```

### Best Result Selection

```python
# Get the best transcription across all engines
text, lang, metadata = await processor.get_best_transcription(
    audio_file, 
    language="en-US",
    confidence_threshold=0.7
)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Some engines are optional dependencies
   ```python
   # Check which engines are available
   info = processor.get_engine_info()
   print("Available engines:", info['available_engines'])
   ```

2. **Model Download Issues**: Models download automatically on first use
   ```python
   # Pre-download models
   await processor._load_whisper_model("base")
   await processor._load_faster_whisper_model("base")
   ```

3. **Memory Issues**: Use smaller models for limited memory
   ```python
   # Use tiny model for low memory
   text, lang, metadata = await processor.transcribe(
       audio_file, model_size="tiny"
   )
   ```

4. **Poor Accuracy**: Try different engines or preprocessing
   ```python
   # Force high-quality processing
   text, lang, metadata = await processor.transcribe(
       audio_file, 
       engine=SpeechEngine.WHISPER,
       model_size="large-v3"
   )
   ```

### Performance Optimization

1. **Pre-load Models**: Load models at startup to reduce latency
2. **Use Appropriate Quality**: Match quality to your needs
3. **Enable VAD**: Filter out silent audio automatically
4. **Cache Results**: Store transcriptions to avoid re-processing

## 📊 Configuration Options

### Environment Variables

```bash
# Voice processing configuration
export VOICE_QUALITY=medium          # low, medium, high
export VOICE_MAX_FILE_SIZE_MB=50     # Maximum audio file size
export VOICE_TIMEOUT_SECONDS=300     # Maximum processing time
export VOICE_ENABLE_VAD=true         # Enable voice activity detection
export VOICE_CACHE_MODELS=true       # Cache models in memory
export VOICE_LOG_LEVEL=INFO          # Logging level
```

### Audio Settings

```python
from src.services.media.voice_config import VoiceConfig

# Customize audio preprocessing
VoiceConfig.AUDIO_SETTINGS = {
    "sample_rate": 16000,      # 16kHz for speech recognition
    "channels": 1,             # Mono audio
    "bit_depth": 16,           # 16-bit audio
    "chunk_size": 4000,        # Processing chunk size
    "max_file_size_mb": 50,    # File size limit
    "timeout_seconds": 300,    # Processing timeout
}
```

## 🔍 Examples

See the `examples/enhanced_voice_recognition_usage.py` file for comprehensive examples including:

- Basic usage patterns
- Multi-engine comparison
- Language-specific processing
- Quality settings
- Telegram integration
- Performance optimization
- Error handling

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
python -m pytest tests/test_enhanced_voice_processor.py -v
```

## 🤝 Contributing

When adding new features:

1. Update the `VoiceProcessor` class
2. Add configuration options to `VoiceConfig`
3. Include examples in the usage file
4. Add tests for new functionality
5. Update this documentation

## 📝 License

This enhanced voice recognition system follows the same license as the main Telegram bot project.

---

## 🎉 Getting Started Checklist

- [ ] Install required dependencies
- [ ] Test basic transcription with Google engine
- [ ] Install Whisper for better accuracy: `pip install openai-whisper`
- [ ] Install Faster-Whisper for speed: `pip install faster-whisper`
- [ ] Install Vosk for offline processing: `pip install vosk`
- [ ] Configure language-specific settings
- [ ] Integrate with your Telegram bot
- [ ] Test with your audio samples
- [ ] Set up monitoring and logging

**Ready to process voice messages like a pro! 🎤✨**
