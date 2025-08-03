import logging
import html
import re
import subprocess
import tempfile
import os
import platform
from typing import List, Optional, Any
from telegramify_markdown import convert, escape_markdown, markdownify, customize


class ResponseFormatter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # --- NEW FEATURE: Table conversion support ---
    def _convert_tables(self, text: str) -> str:
        pattern = r"(?:\|.*\|\s*\n)(?:\|[-: ]+\|\s*\n)(?:\|.*\|\s*\n?)+"

        def repl(match):
            rows = match.group(0).strip().splitlines()
            return "\n".join(f"```\n{row}\n```" for row in rows)

        return re.sub(pattern, repl, text)

    # --- NEW FEATURE: Spoiler & underline support ---
    def _handle_spoilers_and_underlines(self, text: str) -> str:
        text = re.sub(
            r"\|\|(.+?)\|\|", lambda m: f"||{escape_markdown(m.group(1))}||", text
        )
        text = re.sub(
            r"__(.+?)__", lambda m: f"__{escape_markdown(m.group(1))}__", text
        )
        return text

    # --- ENHANCED: Mermaid rendering with better error handling and syntax cleanup ---
    def _render_mermaid_to_image(self, mmd_text: str) -> Any:
        try:
            # Clean up the Mermaid syntax to handle AI-generated issues
            cleaned_mmd = self._clean_mermaid_syntax(mmd_text)

            # Create temp files
            with tempfile.NamedTemporaryFile(
                suffix=".mmd", delete=False, mode="w", encoding="utf-8"
            ) as src_file:
                src_file.write(cleaned_mmd)
                src_file.flush()
                src_path = src_file.name

            png_path = src_path.replace(".mmd", ".png")

            # Determine mmdc command based on platform
            if platform.system() == "Windows":
                # Try common Windows paths for mmdc
                possible_paths = [
                    "mmdc",
                    "mmdc.cmd",
                    os.path.join(os.environ.get("APPDATA", ""), "npm", "mmdc.cmd"),
                    os.path.join(
                        "C:",
                        "Users",
                        os.environ.get("USERNAME", ""),
                        "AppData",
                        "Roaming",
                        "npm",
                        "mmdc.cmd",
                    ),
                    # Additional common npm global paths
                    os.path.join(
                        os.environ.get("ProgramFiles", ""), "nodejs", "mmdc.cmd"
                    ),
                    os.path.join(
                        os.environ.get("ProgramFiles(x86)", ""), "nodejs", "mmdc.cmd"
                    ),
                    os.path.join(
                        os.environ.get("USERPROFILE", ""),
                        "AppData",
                        "Roaming",
                        "npm",
                        "mmdc.cmd",
                    ),
                ]

                # Try using npm to find the path
                try:
                    npm_result = subprocess.run(
                        ["npm", "bin", "-g"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if npm_result.returncode == 0:
                        npm_bin_path = npm_result.stdout.strip()
                        possible_paths.append(os.path.join(npm_bin_path, "mmdc.cmd"))
                        possible_paths.append(os.path.join(npm_bin_path, "mmdc"))
                except:
                    self.logger.warning("Failed to get npm global bin path")

                mmdc_cmd = None
                for path in possible_paths:
                    try:
                        result = subprocess.run(
                            [path, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if result.returncode == 0:
                            mmdc_cmd = path
                            self.logger.info(f"Found mmdc at: {path}")
                            break
                    except Exception as e:
                        self.logger.debug(f"Failed to run {path}: {str(e)}")
                        continue

                if not mmdc_cmd:
                    raise Exception(
                        "mmdc command not found. Please install @mermaid-js/mermaid-cli globally using: npm install -g @mermaid-js/mermaid-cli"
                    )
            else:
                mmdc_cmd = "mmdc"

            # Prepare the command with puppeteer config and optimization flags
            # Use absolute path for Docker container
            puppeteer_config = (
                "/app/puppeteer-config.json"
                if os.environ.get("INSIDE_DOCKER")
                else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "puppeteer-config.json",
                )
            )

            # Debug logging to verify config path
            self.logger.info(f"Using puppeteer config: {puppeteer_config}")
            if os.path.exists(puppeteer_config):
                self.logger.info("Puppeteer config file found")
            else:
                self.logger.warning(
                    f"Puppeteer config file not found at: {puppeteer_config}"
                )
                # Fallback to relative path
                puppeteer_config = "puppeteer-config.json"

            # Production-optimized command with performance improvements
            command = [
                mmdc_cmd,
                "-i",
                src_path,
                "-o",
                png_path,
                "--quiet",
                "-p",
                puppeteer_config,
                "-w",
                "1200",  # Increased width for better quality
                "-H",
                "800",  # Set height limit to prevent huge images
                "-b",
                "transparent",
                "--scale",
                "2",  # Better quality for high-DPI displays
                "--theme",
                "default",  # Explicitly set theme for consistency
            ]

            # Production-optimized timeout based on diagram complexity
            # Adaptive timeout based on input size
            input_size = len(cleaned_mmd)
            if input_size < 1000:
                production_timeout = 45  # Simple diagrams: 45s (increased from 30s)
            elif input_size < 5000:
                production_timeout = (
                    90  # Medium diagrams: 1.5 minutes (increased from 60s)
                )
            else:
                production_timeout = (
                    180  # Complex diagrams: 3 minutes (increased from 120s)
                )

            # Retry logic for better reliability
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    self.logger.info(
                        f"Rendering Mermaid diagram (attempt {attempt + 1}/{max_retries + 1}, timeout: {production_timeout}s, size: {input_size} chars)"
                    )

                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=production_timeout,
                        env={
                            **os.environ,
                            "NODE_OPTIONS": "--max-old-space-size=4096",
                        },  # Increase Node.js memory
                    )

                    # Check if PNG was created successfully
                    if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                        self.logger.info(
                            f"Mermaid diagram rendered successfully on attempt {attempt + 1}"
                        )
                        break
                    else:
                        raise Exception("PNG file was not created or is empty")

                except subprocess.TimeoutExpired:
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Attempt {attempt + 1} timed out, retrying with increased timeout..."
                        )
                        production_timeout = int(
                            production_timeout * 1.5
                        )  # Increase timeout for retry
                        continue
                    else:
                        raise
                except subprocess.CalledProcessError as e:
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed: {e.stderr}, retrying..."
                        )
                        continue
                    else:
                        raise
                except Exception as e:
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}, retrying..."
                        )
                        continue
                    else:
                        raise

            # If all retries failed and it's a complex diagram, try simplified rendering
            if input_size > 5000:
                self.logger.warning(
                    "Attempting simplified rendering for complex diagram..."
                )
                try:
                    # Simplified command for complex diagrams
                    simple_command = [
                        mmdc_cmd,
                        "-i",
                        src_path,
                        "-o",
                        png_path,
                        "--quiet",
                        "-p",
                        puppeteer_config,
                        "-w",
                        "800",  # Reduced width
                        "-H",
                        "600",  # Reduced height
                        "-b",
                        "white",  # Solid background
                        "--scale",
                        "1",  # Lower scale
                        "--theme",
                        "base",  # Simpler theme
                    ]

                    result = subprocess.run(
                        simple_command,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,  # Shorter timeout for simplified rendering
                        env={**os.environ, "NODE_OPTIONS": "--max-old-space-size=2048"},
                    )

                    if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                        self.logger.info("Simplified rendering succeeded")
                        # Return file handle
                        img_file = open(png_path, "rb")
                        # Clean up temp mermaid file
                        try:
                            os.unlink(src_path)
                        except:
                            pass
                        return img_file

                except Exception as simple_error:
                    self.logger.warning(
                        f"Simplified rendering also failed: {simple_error}"
                    )

            # Return file handle
            img_file = open(png_path, "rb")

            # Clean up temp mermaid file
            try:
                os.unlink(src_path)
            except:
                pass

            return img_file

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Mermaid CLI error: {e.stderr}")
            raise Exception(f"Mermaid rendering failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"Mermaid rendering timed out after {production_timeout}s - diagram complexity: {len(cleaned_mmd)} chars"
            )
            raise Exception(
                f"Mermaid rendering timed out after {production_timeout}s - the diagram might be too complex or your system is under heavy load"
            )
        except Exception as e:
            self.logger.error(f"Mermaid rendering error: {e}")
            raise

    def _clean_mermaid_syntax(self, mmd_text: str) -> str:
        """Clean up Mermaid syntax to handle AI-generated issues and improve performance"""
        lines = mmd_text.split("\n")
        cleaned_lines = []
        diagram_type = None

        for line_num, line in enumerate(lines):
            # Remove comments (// style comments are not supported in Mermaid)
            if "//" in line:
                line = line.split("//")[0].strip()

            # Skip empty lines after comment removal
            if not line.strip():
                continue

            # Fix common AI syntax issues
            line = line.strip()

            # Detect diagram type from first line
            if line_num == 0 or (
                not diagram_type
                and any(
                    x in line.lower()
                    for x in [
                        "graph",
                        "flowchart",
                        "sequencediagram",
                        "classDiagram",
                        "stateDiagram",
                        "erDiagram",
                        "journey",
                        "gantt",
                        "pie",
                    ]
                )
            ):
                diagram_type = line.lower().split()[0] if line.split() else None

            # Remove semicolons at the end of lines (not needed in Mermaid)
            if line.endswith(";"):
                line = line[:-1]

            # Fix common syntax issues for different diagram types
            if diagram_type:
                # For flowcharts and graphs
                if diagram_type in ["graph", "flowchart"]:
                    # Fix arrow syntax issues
                    line = re.sub(r"-->", "→", line)  # Convert --> to →
                    line = re.sub(r"→", "-->", line)  # Convert back to standard
                    # Fix node labeling issues
                    line = re.sub(
                        r"\[(.*?)\]", r"[\1]", line
                    )  # Ensure proper bracket syntax

                # For sequence diagrams
                elif diagram_type == "sequencediagram":
                    # Fix participant naming
                    line = re.sub(r"participant\s+([^:]+):", r"participant \1 as", line)

                # For class diagrams
                elif diagram_type == "classdiagram":
                    # Fix class method syntax
                    line = re.sub(r"(\+|\-|\#)\s*([^(]+)\(", r"\1\2(", line)

            # Limit line length to prevent rendering issues
            if len(line) > 200:
                self.logger.warning(
                    f"Long line detected ({len(line)} chars), truncating for performance"
                )
                line = line[:200] + "..."

            # Skip malformed lines that could cause parsing errors
            if line and not re.match(
                r"^[a-zA-Z0-9\s\-\>\<\[\]\(\)\{\}\|\+\-\#\:\;\.\,\=\%\"\'\_\→\←\↑\↓]*$",
                line,
            ):
                self.logger.warning(
                    f"Potentially malformed line skipped: {line[:50]}..."
                )
                continue

            cleaned_lines.append(line)

            # Limit total number of lines to prevent extremely complex diagrams
            if len(cleaned_lines) > 100:
                self.logger.warning(
                    "Diagram too complex (>100 lines), truncating for performance"
                )
                cleaned_lines.append("... (diagram truncated for performance)")
                break

        result = "\n".join(cleaned_lines)

        # Final validation - ensure the diagram has a valid structure
        if not result.strip():
            raise Exception("Empty diagram after cleaning")

        # Check for minimum valid content
        if len(result.strip()) < 10:
            raise Exception("Diagram too short or invalid")

        return result

    def _fix_mermaid_labels(self, line: str) -> str:
        """This method is kept for potential future use but simplified for now"""
        return line

    async def format_telegram_markdown(self, text: str) -> str:
        text = str(text)
        text = self._clean_unwanted_dashes(text)
        try:
            return convert(text)
        except Exception as e:
            self.logger.error(f"format_telegram_markdown error: {e}")
            return self._escape_all(text)

    async def escape_markdown_text(self, text: str) -> str:
        text = str(text)
        try:
            return escape_markdown(text)
        except Exception as e:
            self.logger.error(f"escape_markdown_text error: {e}")
            return self._escape_all(text)

    # --- OVERRIDDEN: Markdownify with table + spoiler/underline handling ---
    async def markdownify_text(
        self, md: str, normalize_whitespace: bool = False
    ) -> str:
        md = str(md)
        md = self._clean_unwanted_dashes(md)
        md = self._convert_tables(md)
        md = self._handle_spoilers_and_underlines(md)
        try:
            return markdownify(
                md, normalize_whitespace=normalize_whitespace, latex_escape=True
            )
        except Exception as e:
            self.logger.error(f"markdownify_text error: {e}")
            return await self.format_telegram_markdown(md)

    async def format_telegram_html(self, text: str) -> str:
        try:
            return html.escape(str(text))
        except Exception as e:
            self.logger.error(f"format_telegram_html error: {e}")
            return html.escape(str(text))

    def set_markdown_options(self, **opts) -> "ResponseFormatter":
        cfg = customize
        if "strict_markdown" in opts:
            cfg.strict_markdown = bool(opts["strict_markdown"])
        if "cite_expandable" in opts:
            cfg.cite_expandable = bool(opts["cite_expandable"])
        symbols = opts.get("markdown_symbols", {})
        if isinstance(symbols, dict):
            for lvl in range(1, 7):
                if f"head_level_{lvl}" in symbols:
                    setattr(
                        cfg.markdown_symbol,
                        f"head_level_{lvl}",
                        symbols[f"head_level_{lvl}"],
                    )
            for name in ("link", "image", "item", "task_list"):
                if name in symbols:
                    setattr(cfg.markdown_symbol, name, symbols[name])
        return self

    # --- OVERRIDDEN: Markdown-safe splitting + table pre-processing ---
    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """Split long messages into chunks that fit within Telegram's limits"""
        text = self._convert_tables(str(text or ""))

        if len(text) <= max_length:
            return [text]

        chunks = []

        # Try to split by paragraphs first (double newlines)
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            # If a single paragraph is too long, we need to split it further
            if len(paragraph) > max_length:
                # If we have content in current_chunk, save it first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split the long paragraph by sentences or lines
                lines = paragraph.split("\n")
                for line in lines:
                    if len(line) > max_length:
                        # If a single line is too long, split by characters
                        while len(line) > max_length:
                            split_point = max_length
                            # Try to find a good split point (space, comma, etc.)
                            for i in range(max_length - 100, max_length):
                                if i < len(line) and line[i] in " ,.;:":
                                    split_point = i + 1
                                    break

                            chunks.append(line[:split_point].strip())
                            line = line[split_point:]

                        if line.strip():
                            current_chunk = line.strip()
                    else:
                        # Check if adding this line would exceed the limit
                        if len(current_chunk) + len(line) + 1 > max_length:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = line
                        else:
                            if current_chunk:
                                current_chunk += "\n" + line
                            else:
                                current_chunk = line
            else:
                # Check if adding this paragraph would exceed the limit
                if len(current_chunk) + len(paragraph) + 2 > max_length:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph

        # Add the last chunk if there's any content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Ensure no chunk is empty and all are within limits
        final_chunks = []
        for chunk in chunks:
            if chunk.strip() and len(chunk) <= max_length:
                final_chunks.append(chunk)
            elif chunk.strip() and len(chunk) > max_length:
                # Emergency split if we still have oversized chunks
                while len(chunk) > max_length:
                    final_chunks.append(chunk[: max_length - 3] + "...")
                    chunk = "..." + chunk[max_length - 3 :]
                if chunk.strip():
                    final_chunks.append(chunk)

        return final_chunks if final_chunks else [text[:max_length]]

    def format_with_model_indicator(
        self, text: str, model: str, is_reply: bool = False
    ) -> str:
        text = str(text)
        header = model + ("\n↪️ Replying to message\n" if is_reply else "\n")
        return f"{header}{text}"

    def _escape_all(self, text: str) -> str:
        specials = r"_*[]()~`>#+\-=|{}.!".split()
        for ch in specials:
            text = text.replace(ch, f"\\{ch}")
        return text

    def _clean_unwanted_dashes(self, text: str) -> str:
        """Remove standalone dashes that appear at the end of lines"""
        # Remove lines that contain only dashes, spaces, or are empty
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip lines that are just dashes, asterisks, or empty
            if stripped and not re.match(r"^[-*\s]+$", stripped):
                cleaned_lines.append(line)
            elif not stripped:  # Keep empty lines for formatting
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    # --- ENHANCED: Detect mermaid blocks anywhere in message and render as image ---
    async def safe_send_message(
        self,
        message,
        text: str,
        reply_to_message_id: Optional[int] = None,
        disable_web_page_preview: bool = False,
    ) -> Optional[Any]:
        # Check for Mermaid blocks anywhere in the text
        mermaid_pattern = r"```mermaid\s*\n(.*?)\n```"
        mermaid_matches = re.findall(mermaid_pattern, text, re.DOTALL)

        # Debug logging
        self.logger.info(
            f"Checking text for Mermaid blocks. Found {len(mermaid_matches)} matches"
        )
        if mermaid_matches:
            self.logger.info(f"Mermaid content: {mermaid_matches[0][:100]}...")

        if mermaid_matches:
            # Extract the first Mermaid diagram
            mmd_content = mermaid_matches[0].strip()
            try:
                img = self._render_mermaid_to_image(mmd_content)

                # Remove the Mermaid block from text and send remaining text if any
                remaining_text = re.sub(
                    mermaid_pattern, "", text, count=1, flags=re.DOTALL
                ).strip()

                # Send the image first
                result = await message.reply_photo(
                    photo=img,
                    caption="Mermaid diagram",
                    reply_to_message_id=reply_to_message_id,
                )

                # Send remaining text if there's any meaningful content
                if (
                    remaining_text and len(remaining_text) > 10
                ):  # Only if substantial text remains
                    await self._send_single_message(
                        message, remaining_text, None, disable_web_page_preview
                    )

                return result

            except Exception as e:
                self.logger.error(f"Mermaid rendering failed: {e}")

                # If Mermaid rendering fails, provide helpful feedback and send the original text
                error_context = ""
                if "Parse error" in str(e):
                    error_context = "\n\n*Note: The Mermaid diagram contains syntax errors. Here's the original code:*"
                elif "mmdc command not found" in str(e):
                    error_context = "\n\n*Note: Mermaid CLI is not available. Here's the diagram code:*"

                # Send the original text with error context
                fallback_text = f"{text}{error_context}"
                return await self._send_single_message(
                    message,
                    fallback_text,
                    reply_to_message_id,
                    disable_web_page_preview,
                )

        # Check if message is too long and split it
        if len(text) > 4000:  # Leave some margin below Telegram's 4096 limit
            chunks = await self.split_long_message(text, max_length=4000)
            results = []
            for i, chunk in enumerate(chunks):
                try:
                    # Only use reply_to_message_id for the first chunk
                    reply_id = reply_to_message_id if i == 0 else None
                    result = await self._send_single_message(
                        message, chunk, reply_id, disable_web_page_preview
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to send chunk {i + 1}: {e}")
            return results[0] if results else None

        # For normal-length messages, try different formatting approaches
        return await self._send_single_message(
            message, text, reply_to_message_id, disable_web_page_preview
        )

    async def _send_single_message(
        self,
        message,
        text: str,
        reply_to_message_id: Optional[int] = None,
        disable_web_page_preview: bool = False,
    ) -> Optional[Any]:
        """Send a single message with multiple formatting fallbacks"""

        # Try MarkdownV2 first
        try:
            formatted_text = await self.format_telegram_markdown(text)
            return await message.reply_text(
                text=formatted_text,
                parse_mode="MarkdownV2",
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception:
            pass

        # Try HTML
        try:
            html_text = await self.format_telegram_html(text)
            return await message.reply_text(
                text=html_text,
                parse_mode="HTML",
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception:
            pass

        # Try escaped MarkdownV2
        try:
            escaped = await self.escape_markdown_text(text)
            return await message.reply_text(
                text=escaped,
                parse_mode="MarkdownV2",
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception:
            pass

        # Final fallback - plain text
        try:
            return await message.reply_text(
                text=text,
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception as e:
            self.logger.error(f"All send attempts failed: {e}")
            return None

    async def format_response(
        self, content: str, user_id: int = None, model_name: str = None
    ) -> str:
        try:
            content = str(content) if content else ""
            if model_name:
                model_badges = {
                    "gemini-2.0-flash": "🤖 *Gemini 2\\.0 Flash*",
                    "gemini": "🤖 *Gemini*",
                    "deepseek": "🧠 *DeepSeek*",
                    "openrouter": "🔀 *OpenRouter*",
                    "gpt": "🔥 *GPT*",
                    "claude": "🌟 *Claude*",
                    "moonshot": "🌙 *Moonshot Kimi*",
                    "kimi": "🌙 *Kimi*",
                }
                badge = next(
                    (b for k, b in model_badges.items() if k in model_name.lower()),
                    None,
                )
                if not badge:
                    disp = model_name.replace("-", "\\-").replace(".", "\\.")
                    badge = f"🤖 *{disp}*"
                content = f"{badge}\n\n{content}"
            return await self.format_telegram_markdown(content)
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return await self.escape_markdown_text(str(content or ""))
