#!/usr/bin/env python3
"""
Test script for Mermaid diagram rendering
"""
import sys
import os
sys.path.append('src')

from src.handlers.response_formatter import ResponseFormatter
import asyncio

async def test_mermaid():
    formatter = ResponseFormatter()
    
    # Test Mermaid content with AI-generated issues (comments and semicolons)
    problematic_mermaid = """flowchart TD
    A([Start]) --> B[User Initiates Registration]
    B --> C[User Enters Registration Data]; // This is a comment
    C --> D{Validation Successful?}
    D -->|Yes| E[Account Created]
    D -->|No| F[Show Error Message];
    E --> G[Confirmation Email Sent]
    F --> C
    G --> H[User Receives Email]
    H --> I[User Logs In]; // Assuming login follows
    I --> J([End])"""
    
    try:
        print("Testing Mermaid rendering with problematic syntax...")
        print(f"Original content (first 200 chars): {problematic_mermaid[:200]}...")
        
        # Test the cleaning function
        cleaned = formatter._clean_mermaid_syntax(problematic_mermaid)
        print(f"\nCleaned content (first 200 chars): {cleaned[:200]}...")
        
        img_file = formatter._render_mermaid_to_image(problematic_mermaid)
        print(f"✅ Success! Generated image file: {img_file.name}")
        img_file.close()
        
        # Check file size
        file_size = os.path.getsize(img_file.name)
        print(f"📊 Image file size: {file_size} bytes")
        
        if file_size > 0:
            print("🎉 Mermaid rendering with syntax cleanup is working correctly!")
        else:
            print("❌ Image file is empty")
            
    except Exception as e:
        print(f"❌ Mermaid rendering failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mermaid())
