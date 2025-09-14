"""
remove comment from a line of code in python of dir src only symbol #
"""

import os


def remove_comments_from_file(filepath):
    """Remove comments (pure and inline) from a Python file, ignoring # inside strings."""

    def remove_inline_comment(line):
        # Remove # and everything after, unless inside quotes
        quote = None
        result = []
        i = 0
        while i < len(line):
            c = line[i]
            if c in ('"', "'"):
                if quote is None:
                    quote = c
                elif quote == c:
                    quote = None
            if c == "#" and quote is None:
                break
            result.append(c)
            i += 1
        return "".join(result).rstrip()

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        code = remove_inline_comment(line)
        if code.strip():  # Keep non-empty lines
            cleaned_lines.append(code + "\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)


def process_src_directory(src_dir):
    """Recursively process all .py files in the src directory."""
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                remove_comments_from_file(filepath)
                print(f"Processed: {filepath}")


if __name__ == "__main__":
    src_directory = "src"
    if os.path.exists(src_directory):
        process_src_directory(src_directory)
        print("Comment removal completed.")
    else:
        print("src directory not found.")
