
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers.response_formatter import ResponseFormatter


async def demo_formatter():
    """Demonstrate different formatting approaches."""
    
    formatter = ResponseFormatter()
    
    # Sample text with various formatting challenges
    test_texts = [
        # Academic content with markdown formatting
        """
        # Advanced Detection Methods
        
        **Early detection** of issues is crucial. Here are the main approaches:
        
        1. **Traditional methods**: Using basic detection (>90% accuracy)
        2. **Modern approaches**: ML-based detection with ~95% success rate
        3. **Hybrid solutions**: Combining both methods for optimal results
        
        *Key considerations*:
        - Cost-effectiveness
        - Implementation complexity
        - Scalability requirements
        
        > Note: Results may vary depending on implementation specifics.
        """,
        
        # Text with code blocks and complex formatting
        """
        # Important Research Findings
        
        The research shows that *early intervention* can improve outcomes by 40-60%.
        
        ```python
        def detect_issues(data):
            results = analyze_data(data)
            return {
                'anomalies': results.anomalies,
                'confidence': results.score
            }
        ```
        
        __Please review__ the attached documentation for more examples.
        
        ||This is spoiler content that will be hidden||
        """,
        
        # Text with special characters and links
        """
        # Special Characters & Links Test
        
        Special characters test: _underscore_ *asterisk* [brackets] (parentheses) 
        ~tilde~ `backtick` >greater #hash +plus -minus =equals |pipe {brace} .dot !exclamation
        
        [Check our documentation](https://docs.example.com) for more information.
        
        | Feature | Status | Notes |
        |---------|--------|-------|
        | Basic formatting | ✅ | Working well |
        | Advanced features | ⚠️ | In progress |
        """,
        
        # Mixed content
        """
        # Research Summary
        
        **Methodology**: We analyzed 1,000+ samples using:
        - Statistical analysis (p<0.05)
        - Machine learning models
        - Cross-validation techniques
        
        *Results*: The new approach shows 25% improvement over baseline.
        
        ```python
        def analyze_results():
            return calculate_metrics()
        ```
        """    ]
    
    print("=== ResponseFormatter Demo with telegramify_markdown ===\n")
    
    # Process each test text
    for i, text in enumerate(test_texts, 1):
        print(f"TEST {i}: Testing different formatting approaches")
        print(f"Original text:\n{text}")
        print("-" * 80)
        
        # 1. Basic markdown formatting using convert function
        formatted_md = await formatter.format_telegram_markdown(text)
        print(f"\n1. format_telegram_markdown result:\n{formatted_md[:200]}...")
        
        # 2. Escape special characters
        escaped_md = await formatter.escape_markdown_text(text)
        print(f"\n2. escape_markdown_text result:\n{escaped_md[:200]}...")
        
        # 3. Advanced markdown formatting with markdownify
        markdownified = await formatter.markdownify_text(text)
        print(f"\n3. markdownify_text result:\n{markdownified[:200]}...")
        
        # 4. HTML formatting
        html_formatted = await formatter.format_telegram_html(text)
        print(f"\n4. format_telegram_html result:\n{html_formatted[:200]}...")
        
        print("\n" + "="*80 + "\n")
    
    # Compare specific formatting features
    print("COMPARISON OF FORMATTING APPROACHES:\n")
    
    # Sample text with specific markdown features to highlight differences
    feature_demo_text = """
    # Heading Example
    
    This text has **bold**, *italic*, and `code` formatting.
    
    - List item 1
    - List item 2
    
    [Link example](https://example.com)
    
    > Quote block example
    
    ||Spoiler text||
    
    ```python
    def sample_function():
        return "This is a code block"
    ```
    """
    
    print("Original text:")
    print(feature_demo_text)
    print("-" * 80)
    
    # Compare the three methods on this specialized text
    print("\nMethod 1: format_telegram_markdown")
    method1 = await formatter.format_telegram_markdown(feature_demo_text)
    print(method1)
    
    print("\nMethod 2: escape_markdown_text (just escapes special chars)")
    method2 = await formatter.escape_markdown_text(feature_demo_text)
    print(method2)
    
    print("\nMethod 3: markdownify_text (full markdown to MarkdownV2)")
    method3 = await formatter.markdownify_text(feature_demo_text)
    print(method3)
    
    # Split long message demo
    print("\n" + "="*80)
    print("\nSPLIT LONG MESSAGE DEMO:")
    
    long_text = "This is a very long message.\n" * 100
    split_results = await formatter.split_long_message(long_text, 200)
    
    print(f"Original length: {len(long_text)} chars")
    print(f"Split into {len(split_results)} chunks")
    print(f"First chunk: {split_results[0][:50]}...")
    print(f"Last chunk: {split_results[-1][:50]}...")
    
    print("\n" + "="*80)
    print("Demo completed! The ResponseFormatter now provides multiple approaches")
    print("for handling telegramify_markdown formatting with enhanced capabilities.")


if __name__ == "__main__":
    asyncio.run(demo_formatter())
