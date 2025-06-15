"""
Example usage of the enhanced response_formatter.py with escape_markdown and markdownify.
"""

import asyncio
from src.handlers.response_formatter import ResponseFormatter

async def demonstrate_formatting():
    formatter = ResponseFormatter()
    
    # Example markdown text with various formatting
    markdown_text = """
# Heading Level 1
## Heading Level 2

This is *italic* and this is **bold** text.
This should be __underlined__ in Telegram.

Here's a [link to Telegram](https://telegram.org).

```python
def sample_code():
    print("Hello, Telegram!")
```

- List item 1
- List item 2
  - Nested item
- List item 3

1. Ordered list item 1
2. Ordered list item 2

> This is a quote with *formatting* inside!

And here's some text with special characters: +, -, *, _, [, ], (, ), ~, `, >, #, =, |, {, }, ., !

||This is a spoiler||
"""
    
    print("===== Original Text =====")
    print(markdown_text)
    print("\n")
    
    # Using standard format_telegram_markdown
    standard_formatted = await formatter.format_telegram_markdown(markdown_text)
    print("===== Standard Formatting =====")
    print(standard_formatted)
    print("\n")
    
    # Using escape_markdown_text for simple text escaping
    escaped_text = await formatter.escape_markdown_text("Special characters: *, _, [, ], etc.")
    print("===== Escaped Text =====")
    print(escaped_text)
    print("\n")
    
    # Using markdownify_text for comprehensive markdown conversion
    markdownified = await formatter.markdownify_text(markdown_text)
    print("===== Markdownified Text =====")
    print(markdownified)
    print("\n")

if __name__ == "__main__":
    asyncio.run(demonstrate_formatting())
