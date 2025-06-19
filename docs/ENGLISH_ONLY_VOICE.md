# English-Only Voice Processing

## Overview

To optimize the bot's performance and reduce resource usage, the voice processing system has been simplified to support only English language. This change:

1. Reduces model storage requirements
2. Simplifies processing logic
3. Improves response time for English speakers
4. Decreases memory usage

## Changes Made

The following changes were implemented:

### Voice Configuration

- Removed all non-English language configuration
- Simplified preprocessing settings to focus on English
- Removed high-resource language distinctions
- Removed language detection logic

### Voice Processor

- Removed all non-English language handling
- Simplified transcription to only use Faster-Whisper with English
- Removed Khmer-specific processing methods and logic
- Optimized model selection for English-only use cases
- Updated supported languages list to only include English

### Message Handlers

- Simplified voice message handling to only process English
- Removed language detection and mapping logic
- Updated status and error messages to reflect English-only support
- Improved logging for English transcription results

## Testing Notes

When testing the bot:

1. Only English voice messages will be processed correctly
2. Non-English voice messages will likely be transcribed as English with incorrect results
3. Users should be informed that only English is supported for voice messages

## Future Considerations

If multi-language support becomes necessary again in the future:

1. Restore language detection logic in message handlers
2. Update voice configuration with language-specific settings
3. Re-implement language-specific transcription methods in the voice processor
4. Consider using a more efficient language detection approach
