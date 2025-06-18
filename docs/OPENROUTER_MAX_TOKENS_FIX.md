# OpenRouter API max_tokens Fix Summary

## Issue
The OpenRouter API was failing with 400 errors due to an excessively high `max_tokens` value (263,840) which exceeded the endpoint's maximum limit of 128,000 tokens.

**Error Message:**
```
ERROR:src.services.openrouter_api:OpenRouter API HTTP error: 400 - Bad Request
ERROR:src.services.openrouter_api:Response text: {"error":{"message":"This endpoint's maximum context length is 128000 tokens. However, you requested about 263862 tokens (22 of text input, 263840 in the output). Please reduce the length of either one, or use the \"middle-out\" transform to compress your prompt automatically.","code":400,"metadata":{"provider_name":null}}}
```

## Root Cause
The `max_tokens` parameter in both OpenRouter API methods was set to an unreasonably high default value:
- `generate_response()`: `max_tokens: int = 263840` ❌
- `generate_response_with_key()`: `max_tokens: int = 263840` ❌

This caused all OpenRouter API calls to fail when the model tried to generate responses, as the requested token limit far exceeded what most models can handle.

## Solution
Updated both methods in `/src/services/openrouter_api.py` to use a reasonable default `max_tokens` value:

### Changes Made:
1. **Line 137**: Changed `max_tokens: int = 263840` → `max_tokens: int = 4096`
2. **Line 297**: Changed `max_tokens: int = 263840` → `max_tokens: int = 4096`
3. **Line 10**: Fixed import path from `services.rate_limiter` → `src.services.rate_limiter`

### Why 4096 tokens?
- ✅ **Safe for all models**: Works with virtually all OpenRouter models
- ✅ **Reasonable response length**: Allows for substantial responses (≈3000-4000 words)
- ✅ **Performance efficient**: Reduces API costs and latency
- ✅ **Configurable**: Can still be overridden when calling the methods

## Testing Results
✅ **Before Fix**: 400 errors, no responses
✅ **After Fix**: Successful API calls, proper responses

**Test Results:**
```
INFO:src.services.openrouter_api:OpenRouter response length: 1722 characters, finish_reason: stop
INFO:__main__:✅ Success! Received response with default max_tokens=4096
INFO:__main__:Response length: 1722 characters
```

## Impact
- ✅ **Voice messages now work**: OpenRouter models can properly process transcribed voice messages
- ✅ **Model selection fixed**: The correct model key is now used for OpenRouter requests
- ✅ **Robust error handling**: Proper token limits prevent API failures
- ✅ **Maintained flexibility**: `max_tokens` can still be customized per request

## Files Modified
- `src/services/openrouter_api.py`: Updated `max_tokens` defaults and fixed import
- `src/handlers/message_handlers.py`: Fixed import path (multimodal_processor)

## Verification
Created test scripts to verify the fix:
- `test_openrouter_max_tokens.py`: Direct API testing
- `test_voice_flow.py`: Voice message flow testing

The integration is now complete and working correctly. Voice messages will use the centralized model system with proper OpenRouter API integration, using reasonable token limits that prevent errors while maintaining response quality.
