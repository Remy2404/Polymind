#!/usr/bin/env python3
"""
Script to remove all supports_tool_calls= lines from model_configs.py
"""

import re

def clean_model_configs():
    """Remove all supports_tool_calls= lines from the model configs file."""
    
    file_path = "src/services/model_handlers/model_configs.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove lines containing supports_tool_calls=
    # This regex matches the entire line including indentation
    pattern = r'^\s*supports_tool_calls=.*,?\n'
    cleaned_content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print("✅ Successfully removed all supports_tool_calls= lines from model_configs.py")
    
    # Count how many lines were removed
    original_lines = content.count('\n')
    cleaned_lines = cleaned_content.count('\n')
    removed_lines = original_lines - cleaned_lines
    
    print(f"📊 Removed {removed_lines} lines from the file")
    print(f"📄 File now has {cleaned_lines} lines (was {original_lines})")

if __name__ == "__main__":
    clean_model_configs()
