# Gemini 2.0 Flash Multimodal Integration

This document explains the new, clean, and maintainable Gemini 2.0 Flash API integration that allows sending **combined image + text + file inputs** in a single request.

## 🏗️ Architecture Overview

### New File Structure
```
src/
├── services/
│   ├── gemini_api.py           # 🆕 New clean Gemini 2.0 Flash API
│   ├── multimodal_processor.py # 🔄 Updated for new API
│   └── rate_limiter.py         # Rate limiting support
│
└── utils/docgen/
    └── document_processor.py   # 🆕 New document processor
```

### Removed Files
- ❌ `src/services/gemini_api.py` (old version with incomplete methods)
- ❌ `src/utils/docgen/document_processor.py` (old version)

## 🔥 Key Features

✅ **Combined Multimodal Requests**: Send image + text + files in ONE API call  
✅ **Clean Architecture**: Maintainable, scalable, and well-documented code  
✅ **Error Handling**: Proper retry logic and graceful error handling  
✅ **Media Support**: Images, documents, audio, video processing  
✅ **Optimized for Gemini 2.0 Flash**: Latest model with best performance  
✅ **Legacy Compatibility**: Existing code continues to work  
✅ **Document Analysis**: Advanced document processing capabilities  
✅ **Batch Processing**: Handle multiple files simultaneously  

## 🚀 Usage Examples

### 1. Combined Multimodal Request (What You Asked For!)

```python
from services.gemini_api import GeminiAPI, create_image_input, create_document_input
from services.rate_limiter import RateLimiter

# Initialize
rate_limiter = RateLimiter(requests_per_minute=60)
gemini_api = GeminiAPI(rate_limiter)

# Prepare inputs
media_inputs = []

# Add image
image_data = await get_image_bytes()  # Your image data
image_input = create_image_input(image_data, "screenshot.png")
media_inputs.append(image_input)

# Add document
doc_data = await get_document_bytes()  # Your document data
doc_input = create_document_input(doc_data, "report.pdf")
media_inputs.append(doc_input)

# Text prompt
text_prompt = "Analyze the image and document together. Provide insights on how they relate."

# Send COMBINED request to Gemini 2.0 Flash
result = await gemini_api.process_multimodal_input(
    text_prompt=text_prompt,
    media_inputs=media_inputs
)

if result.success:
    print(f"Response: {result.content}")
else:
    print(f"Error: {result.error}")
```

### 2. Telegram Message Processing

```python
from services.multimodal_processor import TelegramMultimodalProcessor

# Initialize
telegram_processor = TelegramMultimodalProcessor(gemini_api)

# Process any Telegram message (automatically handles all media types)
async def handle_telegram_message(update, context):
    message = update.message
    
    # This automatically extracts text + images + documents + audio + video
    result = await telegram_processor.process_telegram_message(
        message=message,
        custom_prompt="Analyze all content in this message"
    )
    
    if result.success:
        await message.reply_text(result.content)
    else:
        await message.reply_text(f"Error: {result.error}")
```

### 3. Document Processing

```python
from utils.docgen.document_processor import DocumentProcessor

# Initialize
doc_processor = DocumentProcessor(gemini_api)

# Process single document
result = await doc_processor.process_document(
    file_data=document_bytes,
    filename="contract.pdf",
    prompt="Extract key terms and conditions"
)

# Process multiple documents
files = [
    {"data": doc1_bytes, "filename": "contract.pdf"},
    {"data": doc2_bytes, "filename": "proposal.docx"},
    {"data": doc3_bytes, "filename": "budget.xlsx"}
]

result = await doc_processor.process_multiple_documents(
    files=files,
    prompt="Compare these documents and identify discrepancies"
)
```

### 4. Code Analysis

```python
# Analyze code files
result = await doc_processor.code_analysis(
    file_data=python_code_bytes,
    filename="app.py",
    analysis_type="comprehensive"  # or "security", "performance", "structure"
)
```

## 🔧 Integration with Existing Message Handlers

Update your `message_handlers.py` to use the new system:

```python
# In your _handle_image_message method
async def _handle_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_id = update.effective_user.id
        
        # Get conversation context
        context_messages = await self.conversation_manager.get_context(user_id)
        
        # Process with new multimodal system
        result = await self.multimodal_processor.process_telegram_message(
            message=update.message,
            context=context_messages
        )
        
        if result.success:
            formatted_response = await self.response_formatter.format_response(
                result.content, user_id, model_name="gemini-2.0-flash"
            )
            await update.message.reply_text(formatted_response, parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Error: {result.error}")
            
    except Exception as e:
        self.logger.error(f"Error in image message handler: {str(e)}")
        await self._error_handler(update, context)
```

## 📋 Configuration

The new system uses the same environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🎯 Benefits of New Implementation

### 1. **Single Request for Multiple Media Types**
- Old: Separate requests for image, text, document
- New: Combined request with all media types

### 2. **Better Performance**
- Fewer API calls = faster responses
- Optimized for Gemini 2.0 Flash model
- Intelligent rate limiting

### 3. **Maintainable Code**
- Clear separation of concerns
- Well-documented classes and methods
- Easy to extend for new media types

### 4. **Error Handling**
- Comprehensive error handling
- Retry logic for transient failures
- Graceful degradation

### 5. **Scalability**
- Modular design for easy extension
- Support for batch processing
- Efficient resource usage

## 🔄 Migration Guide

### From Old Implementation

1. **Replace imports**:
   ```python
   # Old
   from services.gemini_api import GeminiAPI
   
   # New (same import, but updated implementation)
   from services.gemini_api import GeminiAPI
   ```

2. **Update multimodal calls**:
   ```python
   # Old way (multiple separate calls)
   image_result = await gemini_api.analyze_image(image_data, prompt)
   doc_result = await gemini_api.process_document(doc_data, prompt)
   
   # New way (single combined call)
   media_inputs = [
       create_image_input(image_data, "image.jpg"),
       create_document_input(doc_data, "doc.pdf")
   ]
   result = await gemini_api.process_multimodal_input(prompt, media_inputs)
   ```

3. **Use new processors**:
   ```python
   # For Telegram messages
   telegram_processor = TelegramMultimodalProcessor(gemini_api)
   result = await telegram_processor.process_telegram_message(message)
   
   # For documents
   doc_processor = DocumentProcessor(gemini_api)
   result = await doc_processor.process_document(file_data, filename, prompt)
   ```

## 🧪 Testing

Run the example file to test the new implementation:

```bash
cd d:\Telegram-Gemini-Bot
python examples/gemini_2_flash_multimodal_usage.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're importing from the correct paths
2. **API Key**: Ensure `GEMINI_API_KEY` is set in your environment
3. **File Size**: Documents over 50MB are not supported
4. **Image Format**: Only JPEG, PNG, WEBP, GIF are supported

### Debug Logging

Enable debug logging to see detailed processing information:

```python
import logging
logging.getLogger("services.gemini_api").setLevel(logging.DEBUG)
```

## 📈 Performance Tips

1. **Batch Processing**: Use `process_multiple_documents()` for multiple files
2. **Image Optimization**: Images are automatically optimized for best performance
3. **Rate Limiting**: The system includes intelligent rate limiting
4. **Context Management**: Limit conversation context to recent messages

## 🎉 Summary

The new Gemini 2.0 Flash integration provides:

- ✅ **What you asked for**: Combined img_input + text_prompt + files_input in one request
- ✅ **Clean codebase**: Removed old, incomplete files and created maintainable new ones
- ✅ **Future-ready**: Scalable architecture for easy maintenance and extension
- ✅ **Full compatibility**: Existing code continues to work with minimal changes

The system is now ready for production use with your Telegram chatbot! 🚀
