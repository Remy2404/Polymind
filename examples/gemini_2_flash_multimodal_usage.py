"""
Example Usage of the New Gemini 2.0 Flash Multimodal API
Demonstrates how to send combined img_input + text_prompt + files_input
"""

import sys

import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import asyncio
import io
from src.services.gemini_api import (
    GeminiAPI,
    MediaInput,
    MediaType,
    create_image_input,
    create_document_input,
    create_text_input,
)
from src.services.rate_limiter import RateLimiter
from src.services.multimodal_processor import TelegramMultimodalProcessor
from src.utils.docgen.document_processor import DocumentProcessor


async def example_combined_multimodal_request():
    """
    Example: Send image + text + document in one request to Gemini 2.0 Flash
    This is exactly what you asked for - combining all inputs in one content
    """

    # Initialize the API
    rate_limiter = RateLimiter(requests_per_minute=60)
    gemini_api = GeminiAPI(rate_limiter)

    # Example 1: Image + Text + Document in one request
    print("=== Example 1: Combined Multimodal Request ===")

    # Prepare media inputs
    media_inputs = []

    # Add an image (example with placeholder data)
    # In real usage, you'd have actual image bytes from Telegram
    image_data = b"fake_image_data_here"  # Replace with real image bytes
    # image_input = create_image_input(image_data, "screenshot.png")
    # media_inputs.append(image_input)

    # Add a document (example with text content)
    document_content = """
    # Project Report
    
    ## Overview
    This is a sample document for testing multimodal processing.
    
    ## Key Features
    - Feature 1: Advanced AI processing
    - Feature 2: Multimodal support
    - Feature 3: Scalable architecture
    
    ## Technical Details
    The system uses Python and integrates with Google's Gemini API.
    """
    doc_data = io.BytesIO(document_content.encode("utf-8"))
    doc_input = create_document_input(doc_data, "project_report.md")
    media_inputs.append(doc_input)

    # The main text prompt
    text_prompt = """
    Please analyze all the provided content (images and documents) and provide:
    
    1. A summary of each piece of content
    2. How they relate to each other
    3. Key insights and recommendations
    4. Any action items or next steps
    
    Focus on providing actionable insights based on all the provided information.
    """

    # Send combined request to Gemini 2.0 Flash
    try:
        result = await gemini_api.process_multimodal_input(
            text_prompt=text_prompt,
            media_inputs=media_inputs,
            context=None,  # Add conversation context if available
        )

        if result.success:
            print("✅ SUCCESS!")
            print(f"Response: {result.content}")
            print(f"Metadata: {result.metadata}")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.error}")

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")


async def example_telegram_message_processing():
    """
    Example: Process a Telegram message with multiple attachments
    This shows how the system handles real Telegram messages
    """

    print("\n=== Example 2: Telegram Message Processing ===")

    # Initialize components
    rate_limiter = RateLimiter(requests_per_minute=60)
    gemini_api = GeminiAPI(rate_limiter)
    telegram_processor = TelegramMultimodalProcessor(gemini_api)

    # In real usage, you'd have a real Telegram Message object
    # This is just to show the concept
    print("In real usage, you would:")
    print("1. Receive a Telegram message with photo + document + text")
    print(
        "2. Call: result = await telegram_processor.process_telegram_message(message)"
    )
    print("3. The processor automatically extracts all media and processes together")


async def example_document_processing():
    """
    Example: Advanced document processing with the new system
    """

    print("\n=== Example 3: Document Processing ===")

    # Initialize components
    rate_limiter = RateLimiter(requests_per_minute=60)
    gemini_api = GeminiAPI(rate_limiter)
    doc_processor = DocumentProcessor(gemini_api)

    # Example document processing
    sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"fib({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()
    '''

    code_data = io.BytesIO(sample_code.encode("utf-8"))

    try:
        # Analyze code
        result = await doc_processor.code_analysis(
            code_data, "fibonacci.py", "comprehensive"
        )

        if result.success:
            print("✅ Code Analysis Success!")
            print(f"Analysis: {result.content[:500]}...")  # Truncate for display
        else:
            print(f"❌ Code Analysis Failed: {result.error}")

    except Exception as e:
        print(f"❌ Exception in code analysis: {e}")


async def example_batch_processing():
    """
    Example: Process multiple files at once
    """

    print("\n=== Example 4: Batch Processing ===")

    # Initialize components
    rate_limiter = RateLimiter(requests_per_minute=60)
    gemini_api = GeminiAPI(rate_limiter)
    doc_processor = DocumentProcessor(gemini_api)

    # Prepare multiple files
    files = [
        {
            "data": io.BytesIO(b"# README\nThis is the main project documentation."),
            "filename": "README.md",
        },
        {"data": io.BytesIO(b"print('Hello, World!')"), "filename": "hello.py"},
        {
            "data": io.BytesIO(
                b"TODO:\n- Implement feature X\n- Fix bug Y\n- Update documentation"
            ),
            "filename": "TODO.txt",
        },
    ]

    try:
        result = await doc_processor.process_multiple_documents(
            files, "Analyze these project files and create a development roadmap."
        )

        if result.success:
            print("✅ Batch Processing Success!")
            print(f"Combined Analysis: {result.content[:500]}...")
        else:
            print(f"❌ Batch Processing Failed: {result.error}")

    except Exception as e:
        print(f"❌ Exception in batch processing: {e}")


def show_architecture_overview():
    """
    Show the new architecture overview
    """

    print("\n" + "=" * 60)
    print("🏗️  NEW GEMINI 2.0 FLASH ARCHITECTURE")
    print("=" * 60)
    print(
        """
    📁 src/services/
    ├── gemini_api.py           # 🆕 New clean Gemini 2.0 Flash API
    ├── multimodal_processor.py # 🔄 Updated for new API
    └── rate_limiter.py         # Rate limiting support
    
    📁 src/utils/docgen/
    └── document_processor.py   # 🆕 New document processor
    
    🔥 KEY FEATURES:
    ✅ Combined multimodal requests (img + text + files in ONE call)
    ✅ Clean, maintainable code structure
    ✅ Proper error handling and retries
    ✅ Support for all media types (images, docs, audio, video)
    ✅ Optimized for Gemini 2.0 Flash model
    ✅ Legacy compatibility for existing code
    ✅ Comprehensive document analysis
    ✅ Batch processing capabilities
    
    🚀 USAGE PATTERNS:
    1. Single multimodal request: gemini_api.process_multimodal_input()
    2. Telegram integration: telegram_processor.process_telegram_message()
    3. Document analysis: doc_processor.process_document()
    4. Batch processing: doc_processor.process_multiple_documents()
    5. Code analysis: doc_processor.code_analysis()
    """
    )


async def main():
    """Run all examples"""

    show_architecture_overview()

    print("\n🚀 Running Examples...")
    print(
        "Note: Some examples use placeholder data since we don't have real Telegram messages here"
    )

    # Run examples
    await example_combined_multimodal_request()
    await example_telegram_message_processing()
    await example_document_processing()
    await example_batch_processing()

    print("\n✅ All examples completed!")
    print("\nTo use in your Telegram bot:")
    print("1. Initialize: gemini_api = GeminiAPI(rate_limiter)")
    print(
        "2. Process messages: await telegram_processor.process_telegram_message(message)"
    )
    print("3. The system automatically handles combined inputs!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
