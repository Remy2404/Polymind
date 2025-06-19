# Khmer Voice Transcription Fix Documentation

## 🚨 Problem Identified

**Issue**: Khmer speech "តើ អ្នក សុខសប្បាយ ជាទេ?" (How are you?) was incorrectly transcribed as English "So, slide on it. So, slide on it."

**Root Cause**: Missing Khmer language configuration and insufficient language forcing in Faster-Whisper.

## 🔧 Fixes Applied

### 1. **Enhanced Language Configuration** (`voice_config.py`)

#### ✅ Added Khmer to Engine Preferences
```python
ENGINE_PREFERENCES = {
    # ... existing languages
    "km": ["faster_whisper"],  # Khmer
    "kh": ["faster_whisper"],  # Alternative Khmer code
    "th": ["faster_whisper"],  # Thai (related language)
    "vi": ["faster_whisper"],  # Vietnamese (related language)
    # ...
}
```

#### ✅ Enhanced Audio Preprocessing for Khmer
```python
LANGUAGE_PREPROCESSING = {
    # ... existing languages
    "km": {"normalize": True, "high_pass_filter": 300, "volume_boost": 3},
    "kh": {"normalize": True, "high_pass_filter": 300, "volume_boost": 3},
    "th": {"normalize": True, "high_pass_filter": 200, "volume_boost": 2},
    "vi": {"normalize": True, "high_pass_filter": 150, "volume_boost": 2},
    # ...
}
```

#### ✅ Specialized Confidence Threshold for Khmer
```python
CONFIDENCE_THRESHOLDS = {
    "faster_whisper": 0.7,
    "faster_whisper_khmer": 0.5,  # Lower threshold for Khmer detection challenges
}
```

#### ✅ Updated Hardcoded Configuration
Added Khmer to the hardcoded engine preferences to ensure it works even without `.env` files.

### 2. **Existing Enhanced Khmer Processing** (Already Implemented)

The system already had sophisticated Khmer transcription support in `voice_processor.py`:

#### 🔄 Multi-Strategy Khmer Transcription
- **Strategy 1**: Force Khmer with `large-v3` model
- **Strategy 2**: Use `distil-large-v3` with forced Khmer
- **Strategy 3**: Auto-detection with large model
- **Strategy 4**: Base model fallback with forced Khmer

#### 🎯 Enhanced Language Detection Logic
```python
if lang_code in ["km", "kh"]:
    return await self._transcribe_khmer_enhanced(audio_file_path, language)
```

#### 📊 Improved Result Selection
- Prefers Khmer results that meet confidence threshold
- Falls back to weaker Khmer results if available
- Only accepts non-Khmer results if confidence > 0.8

## 🚀 Expected Improvements

### ✅ **Better Language Recognition**
- Khmer language codes (`km`, `kh`) now properly trigger enhanced processing
- System will use appropriate model sizes and parameters for Khmer

### ✅ **Enhanced Audio Processing**
- Higher frequency filters (300Hz) to capture Khmer phonetics
- Volume boost (+3dB) for clearer Khmer speech detection
- Normalization for consistent audio levels

### ✅ **Lower False Rejection Rate**
- Reduced confidence threshold (0.5 vs 0.7) for Khmer
- Multiple fallback strategies for difficult audio
- Better handling of Khmer vs English misdetection

### ✅ **Automatic Quality Selection**
- Khmer is marked as high-resource language
- Automatically uses HIGH quality (large-v3 model) for better accuracy
- Fallback to smaller models if needed

## 🧪 Testing the Fix

Run the test script to verify the configuration:

```bash
cd /d/Telegram-Gemini-Bot
python test_khmer_voice_fix.py
```

### Expected Test Results:
- ✅ Khmer engine preferences correctly configured
- ✅ Audio preprocessing settings optimized for Khmer
- ✅ Confidence thresholds properly set
- ✅ High-resource language detection working
- ✅ Voice processor creation successful

## 🛠️ Technical Implementation Details

### Language Detection Flow:
1. **User sends Khmer voice message**
2. **Message handler detects Khmer language** (`km-KH`, `is_khmer=True`)
3. **Voice processor routes to enhanced Khmer processing**
4. **Multiple transcription strategies attempted**:
   - Force `language="km"` with `large-v3` model
   - Use optimized `distil-large-v3` model
   - Auto-detection fallback
   - Base model emergency fallback
5. **Best result selected** based on language match and confidence
6. **Enhanced metadata provided** for debugging

### Key Parameters for Khmer:
```python
# Faster-Whisper parameters optimized for Khmer
segments, info = model.transcribe(
    audio_file_path,
    language="km",                    # Force Khmer language
    beam_size=5,                     # Optimal beam search
    word_timestamps=True,            # Detailed timing
    vad_filter=True,                 # Voice activity detection
    condition_on_previous_text=False # Better language detection
)
```

## 🔍 Debugging Khmer Issues

### Enhanced Logging
The system now provides detailed logging for Khmer transcription:

```
🇰🇭 ENHANCED KHMER TRANSCRIPTION STARTING
🧪 Strategy 1: Forcing Khmer with large-v3 model
🧪 Strategy 2: Using distil-large-v3 with forced Khmer
🧪 Strategy 3: Auto-detection with large-v3 model
🧪 Strategy 4: Base model fallback with forced Khmer
🔍 ANALYZING X TRANSCRIPTION RESULTS
🎯 FINAL KHMER TRANSCRIPTION RESULT
```

### User Notifications
When Khmer is misdetected as English, users receive:
- Bilingual notification (Khmer + English)
- Technical information about the detection attempt
- Tips for better voice recognition
- Strategy information for debugging

## 📚 Best Practices for Khmer Voice

### For Users:
1. **Speak clearly and slowly** (និយាយឱ្យច្បាស់និងយឺត)
2. **Avoid background noise** (ជៀសវាងសំឡេងរំខាន)
3. **Use pure Khmer phrases** (ប្រើឃ្លាខ្មែរសុទ្ធ)
4. **Speak for longer duration** (និយាយយូរជាង)
5. **Try sending again if needed** (សាកល្បងផ្ញើម្តងទៀត)

### For Developers:
1. **Monitor confidence scores** for Khmer vs other languages
2. **Check language mismatch flags** in metadata
3. **Review strategy usage** in enhanced processing
4. **Verify audio preprocessing** is applied correctly

## 🎯 Success Metrics

After this fix, expect:
- **Reduced English false positives** for Khmer speech
- **Higher Khmer language detection rate**
- **Better transcription accuracy** for clear Khmer speech
- **More informative error messages** when detection fails
- **Automatic quality optimization** for Khmer audio

## 🔄 Future Improvements

Consider:
1. **Fine-tuning confidence thresholds** based on real usage data
2. **Custom Khmer acoustic models** if available
3. **Contextual language hints** based on user history
4. **Advanced audio enhancement** for noisy Khmer speech
5. **Integration with Khmer spell checking** for post-processing

---

**Note**: This fix addresses the core configuration issues that were preventing proper Khmer language detection. The enhanced multi-strategy processing was already implemented and should now be properly triggered for Khmer voice messages.
