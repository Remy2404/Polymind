
"""
Test script for enhanced table export functionality using Spire.Doc
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.handlers.commands.export_commands import SpireDocumentExporter

def test_table_export():
    """Test the enhanced table export functionality"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Create exporter
    exporter = SpireDocumentExporter(logger)
    
    # Test content with tables
    test_content = """# Test Document with Tables

Here is some content before the table.

| Name | Age | Department |
|------|-----|------------|
| john doe | 25 | engineering |
| jane smith | 30 | marketing |
| bob wilson | 35 | sales |

And here is content after the table.

## Another Section

| Product | Price | Stock |
|---------|-------|-------|
| laptop | $999 | 10 |
| mouse | $25 | 50 |
| keyboard | $75 | 25 |

Final paragraph with **bold** and *italic* text.
"""
    
    try:
        print("Creating DOCX with enhanced table support...")
        document_bytes = exporter.create_docx(test_content)
        
        # Save to file
        output_path = "d:\\Telegram-Gemini-Bot\\test_exports_fixed\\enhanced_table_test.docx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(document_bytes)
        
        print(f"✅ Enhanced table export test completed successfully!")
        print(f"📄 Document saved to: {output_path}")
        print(f"📊 Document size: {len(document_bytes)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced table export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_export()
    sys.exit(0 if success else 1)
