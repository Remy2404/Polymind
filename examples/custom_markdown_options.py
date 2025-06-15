#!/usr/bin/env python3
"""
Demo script showing how to customize markdown formatting with ResponseFormatter.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.handlers.response_formatter import ResponseFormatter


async def demo_custom_markdown():
    """Demonstrate customized markdown formatting options."""
    
    # Sample markdown text with various formatting
    markdown_text = """
    # Custom Markdown Demo
    
    ## Formatting Examples
    
    This text has **bold**, *italic*, and __underlined__ formatting.
    
    > This is a block quote
    > With multiple lines
    
    ### Lists
    
    1. First item
    2. Second item
       - Nested unordered item
       - Another nested item
    
    ### Task Lists
    
    - [ ] Uncompleted task
    - [x] Completed task
    
    ### Code Example
    
    ```python
    def hello_telegram():
        print("Hello, Telegram Markdown!")
    ```
    
    ### Table Example
    
    | Feature | Status |
    |---------|--------|
    | Basic formatting | ✅ |
    | Advanced features | ✅ |
    
    [Link to Telegram API](https://core.telegram.org/bots/api)
    
    ||This is a spoiler text||
    """
    
    # Create formatter with default settings
    default_formatter = ResponseFormatter()
    default_result = await default_formatter.markdownify_text(markdown_text)
    
    print("===== Default Markdown Formatting =====")
    print(default_result[:500] + "...\n")
    
    # Create formatter with custom settings - enable underline
    underline_formatter = ResponseFormatter()
    underline_formatter.set_markdown_options(strict_markdown=False)  # Enable underline
    underline_result = await underline_formatter.markdownify_text(markdown_text)
    
    print("===== With Underline Enabled =====")
    print(underline_result[:500] + "...\n")
    
    # Create formatter with custom heading symbols
    custom_symbols_formatter = ResponseFormatter()
    custom_symbols_formatter.set_markdown_options(
        markdown_symbols={
            'head_level_1': '📌',
            'head_level_2': '📎',
            'head_level_3': '🔍',
            'link': '🌐'
        }
    )
    custom_symbols_result = await custom_symbols_formatter.markdownify_text(markdown_text)
    
    print("===== With Custom Symbols =====")
    print(custom_symbols_result[:500] + "...\n")
    
    # Create formatter with expandable citation and custom symbols
    fully_customized = ResponseFormatter()
    fully_customized.set_markdown_options(
        strict_markdown=False,
        cite_expandable=True,
        markdown_symbols={
            'head_level_1': '📝',
            'link': '🔗'
        }
    )
    fully_customized_result = await fully_customized.markdownify_text(markdown_text)
    
    print("===== Fully Customized Formatting =====")
    print(fully_customized_result[:500] + "...\n")


if __name__ == "__main__":
    asyncio.run(demo_custom_markdown())
