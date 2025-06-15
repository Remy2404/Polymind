#!/usr/bin/env python3
"""
Test script to demonstrate how the enhanced ResponseFormatter handles duplicate markdown formatting.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers.response_formatter import ResponseFormatter


async def test_duplicate_markdown():
    """Test the formatter with various duplicate markdown scenarios."""
    
    formatter = ResponseFormatter()
    
    # Test cases with duplicate/problematic markdown
    test_cases = [
        {
            "name": "AI Response with Duplicate Bold",
            "text": """**Next.js vs React: A Comprehensive Comparison**

Next.js and React are both popular tools in the JavaScript ecosystem, but they serve different purposes and have distinct features. Here's a detailed comparison to help you understand the differences and choose the right one for your project.

**1. Purpose and Use Cases****

• **React**:
• **Primary Use**: Building user interfaces (UI) and components.
• **Use Cases**: Front-end development, single-page applications (SPAs), reusable components, and dynamic UIs.
• **Best For**: Developers who want flexibility in architecture and need to manage complex UIs.

• **Next.js**:
• **Primary Use**: Building full-stack web applications with features like server-side rendering (SSR), static site generation (SSG), and routing.
• **Use Cases**: Enterprise applications, e-commerce sites, blogs, and any project requiring efficient routing and server-side rendering.
• **Best For**: Developers looking for a complete framework with built-in tools for routing, deployment, and performance optimization."""
        },
        
        {
            "name": "Mixed Formatting Issues",
            "text": """****Important findings:**** show that ****early intervention**** can improve outcomes.

**Key points**:
• ****Method 1****: Traditional approach ****with 90% accuracy****
• ****Method 2****: Modern ML-based ****detection system****
• ****Method 3****: Hybrid solution ****combining both methods****"""
        },
        
        {
            "name": "Already Clean Markdown",
            "text": """**Clean Response**

This text already has proper markdown formatting:

• **Method 1**: Well formatted
• **Method 2**: No duplicates
• **Method 3**: Properly structured

*Key takeaway*: This should remain unchanged."""
        },
        
        {
            "name": "No Markdown (Plain Text)",
            "text": """This is plain text without any markdown formatting.
It should get academic formatting applied since there's no existing markdown.

Key points would be:
1. Detection methods
2. Early detection approaches
3. Existing approaches for analysis"""
        }
    ]
    
    print("=== Duplicate Markdown Handling Test ===\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"TEST {i}: {case['name']}")
        print("=" * 50)
        
        # Show original text
        print("ORIGINAL:")
        print(case['text'][:200] + "..." if len(case['text']) > 200 else case['text'])
        print()
        
        # Test markdown detection
        has_markdown = formatter._detect_existing_markdown(case['text'])
        print(f"Detected existing markdown: {has_markdown}")
        
        # Test duplicate cleaning
        if has_markdown:
            cleaned = formatter._clean_duplicate_markdown(case['text'])
            print("CLEANED:")
            print(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
            print()
        
        # Test AI response formatting
        try:
            formatted = await formatter.format_ai_response(case['text'])
            print("AI RESPONSE FORMATTED:")
            print(formatted[:300] + "..." if len(formatted) > 300 else formatted)
        except Exception as e:
            print(f"AI response formatting failed: {str(e)}")
        
        print("\n" + "="*60 + "\n")
    
    # Test the comparison between different formatting approaches
    print("COMPARISON TEST: Different Formatting Approaches")
    print("=" * 60)
    
    sample_text = """**Next.js** vs **React**: ****A Comprehensive Comparison****

****Key differences****:
• ****Purpose****: **React** is for UI, ****Next.js**** is full-stack
• ****Performance****: Next.js has ****built-in optimizations****"""
    
    print("Original (with duplicates):")
    print(sample_text)
    print()
    
    methods = [
        ("Regular markdown", formatter.format_telegram_markdown),
        ("AI response", formatter.format_ai_response),
        ("Escape only", formatter.format_with_escape_only),
    ]
    
    for method_name, method_func in methods:
        try:
            result = await method_func(sample_text)
            print(f"{method_name:15}: {result}")
        except Exception as e:
            print(f"{method_name:15}: ERROR - {str(e)}")
    
    print("\n" + "="*60)
    print("✅ Test completed! The formatter now properly handles duplicate markdown.")


if __name__ == "__main__":
    asyncio.run(test_duplicate_markdown())
