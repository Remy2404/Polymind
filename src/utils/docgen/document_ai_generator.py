import logging
from typing import Optional, Tuple
import re
import traceback

from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from src.services.model_handlers.model_configs import ModelConfigurations
from src.services.gemini_api import GeminiAPI
from src.services.openrouter_api import OpenRouterAPI
from src.services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM
from .document_generator import DocumentGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class AIDocumentGenerator:
    def __init__(self, gemini_api: GeminiAPI = None, openrouter_api: OpenRouterAPI = None, deepseek_api: DeepSeekLLM = None, api_manager=None):
        # Use provided API manager or create a new one
        if api_manager:
            self.api_manager = api_manager
        else:
            # Initialize the unified API manager
            self.api_manager = SuperSimpleAPIManager(
                gemini_api=gemini_api,
                openrouter_api=openrouter_api,
                deepseek_api=deepseek_api
            )
        self.document_generator = DocumentGenerator()
        self.logger = logging.getLogger(__name__)

    async def generate_ai_document(
        self,
        prompt: str,
        output_format: str = "pdf",
        document_type: str = "article",
        model: str = "gemini",
        additional_context: str = None,
        max_tokens: int = 32000,
    ) -> Tuple[bytes, str]:
        try:
            system_prompt = self._get_document_prompt(document_type)
            system_prompt += "\n\nFormat your response using Markdown syntax with:\n"
            system_prompt += "- # for the main title (use only once at the beginning)\n"
            system_prompt += "- ## for section headings\n"
            system_prompt += "- ### for subsection headings\n"
            system_prompt += "- * or - for bullet points\n"
            system_prompt += "- 1. 2. 3. for ordered lists\n"
            system_prompt += "- ``` for code blocks\n"
            system_prompt += "- **text** for bold text\n"
            system_prompt += "- *text* for italic text\n"
            system_prompt += "- > for blockquotes\n"
            system_prompt += "- | column1 | column2 | for tables with header row and separator row\n\n"
            system_prompt += "Use a professional tone and structure with:\n"
            system_prompt += (
                "- A clear introduction that states the purpose and provides context\n"
            )
            system_prompt += (
                "- Logically organized sections with descriptive headings\n"
            )
            system_prompt += "- Appropriate use of formatting to highlight key points\n"
            system_prompt += "- Tables to organize comparative data\n"
            system_prompt += "- A comprehensive 'Summary' section at the end that ties together all key points\n\n"
            system_prompt += "IMPORTANT: Document Structure Requirements:\n"
            system_prompt += "1. Begin with a detailed introduction that clearly states the purpose and scope\n"
            system_prompt += (
                "2. Include a table when presenting comparative data or metrics\n"
            )
            system_prompt += "3. Use consistent formatting for all bullet points and numbered lists\n"
            system_prompt += "4. End with both a conclusion AND a separate summary section that highlights key takeaways\n"
            system_prompt += "5. Do not leave empty sections - each heading must have substantial content\n"

            user_prompt = (
                f"Create a comprehensive, professional document about: {prompt}"
            )
            if additional_context:
                user_prompt += f"\n\nAdditional context: {additional_context}"

            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Use the unified API manager to generate content with the selected model
            try:
                self.logger.info(f"Generating document using model: {model}")
                ai_response = await self.api_manager.chat(
                    model_id=model,
                    prompt=full_prompt,
                    temperature=0.7,
                    max_tokens=max_tokens
                ) 
                self.logger.info(f"API response received from {model}")
            except Exception as api_error:
                self.logger.error(f"API error with {model}: {str(api_error)}")
                # Try fallback to gemini if it's not the model we just tried
                if model != "gemini":
                    try:
                        self.logger.info("Attempting fallback to Gemini model")
                        ai_response = await self.api_manager.chat(
                            model_id="gemini",
                            prompt=full_prompt,
                            temperature=0.7,
                            max_tokens=max_tokens
                        )
                        self.logger.info("Fallback to Gemini successful")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback API error: {str(fallback_error)}")
                        ai_response = "# Error Generating Document\n\nThere was a problem generating your document with the AI model. Please try again with a different model or prompt."
                else:
                    ai_response = "# Error Generating Document\n\nThere was a problem generating your document with the AI model. Please try again with a different model or prompt."

            # The SuperSimpleAPIManager returns a string directly
            content = ai_response if isinstance(ai_response, str) else str(ai_response)
            self.logger.info(f"Content generated: {len(content)} characters")

            if not content or len(content.strip()) < 10:
                self.logger.error("Empty or too short content generated")
                content = "# Document Generation Failed\n\nThe AI model did not generate sufficient content for your document. Please try again with a more specific prompt or a different document type."

            title = self._extract_title(content) or f"AI Document: {prompt[:50]}"
            self.logger.info(f"Using title: {title}")

            if content:
                if not content.strip().startswith("# "):
                    content = f"# {title}\n\n{content}"
                content = self._process_empty_sections(content)
                content = self._clean_markdown_formatting(content)
                content = self._ensure_section_spacing(content)
                self.logger.info(
                    "Content processed and formatted for document generation"
                )

            if output_format.lower() == "pdf":
                document_bytes = await self.document_generator.create_pdf(
                    content=content, title=title, author="DeepGem AI"
                )
            else:
                document_bytes = await self.document_generator.create_docx(
                    content=content, title=title, author="DeepGem AI"
                )

            return document_bytes, title

        except Exception as e:
            self.logger.error(f"Error generating AI document: {str(e)}")
            traceback.print_exc()
            raise

    def _process_empty_sections(self, content: str) -> str:
        lines = content.splitlines()
        result_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]
            result_lines.append(current_line)

            if re.match(r"^#{1,6}\s+", current_line):
                heading_level = len(re.match(r"^(#{1,6})\s+", current_line).group(1))
                heading_text = current_line.strip("#").strip()

                next_non_empty = i + 1
                while next_non_empty < len(lines) and not lines[next_non_empty].strip():
                    next_non_empty += 1

                if next_non_empty >= len(lines) or re.match(
                    r"^#{1,6}\s+", lines[next_non_empty]
                ):
                    section_type = ""
                    if "introduction" in heading_text.lower():
                        section_type = "an introductory overview"
                    elif "conclusion" in heading_text.lower():
                        section_type = "a summary of key points and future outlook"
                    elif (
                        "economic" in heading_text.lower()
                        or "growth" in heading_text.lower()
                    ):
                        section_type = "economic analysis and growth projections"
                    elif "challenge" in heading_text.lower():
                        section_type = "key challenges and potential solutions"
                    elif "opportunit" in heading_text.lower():
                        section_type = "emerging opportunities and strategic advantages"
                    elif (
                        "driver" in heading_text.lower()
                        or "sector" in heading_text.lower()
                    ):
                        section_type = (
                            "analysis of key economic sectors and growth drivers"
                        )
                    elif (
                        "government" in heading_text.lower()
                        or "polic" in heading_text.lower()
                    ):
                        section_type = "government policies and regulatory framework"
                    else:
                        section_type = "detailed information related to this topic"

                    title = self._extract_title(content) or "the main topic"
                    placeholder = f"This section provides {section_type} for {title}. "
                    placeholder += f"It includes relevant data, analysis, and insights about {heading_text.lower()} "
                    placeholder += f"in the context of {title}."

                    result_lines.append("")
                    result_lines.append(placeholder)
                    result_lines.append("")

            i += 1

        return "\n".join(result_lines)

    def _get_document_prompt(self, document_type: str) -> str:
        document_prompts = {
            "article": "You are an expert content writer. Create a well-structured article with an introduction, body sections, and conclusion.",
            "report": "You are a professional report writer. Create a detailed report with executive summary, findings, analysis, and recommendations.",
            "guide": "You are a technical writer. Create a step-by-step guide with clear instructions, examples, and tips.",
            "summary": "You are a professional summarizer. Create a concise summary highlighting the key points and insights.",
            "essay": "You are an academic writer. Create a well-structured essay with introduction, arguments, and conclusion.",
            "analysis": "You are a data analyst. Create an in-depth analysis with observations, trends, and actionable insights.",
            "proposal": "You are a business consultant. Create a compelling proposal with overview, objectives, methods, and benefits.",
        }

        return document_prompts.get(document_type.lower(), document_prompts["article"])

    def _extract_title(self, content: str) -> Optional[str]:
        if not content:
            return None
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("# "):
                return line[2:].strip()
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line and not cleaned_line.startswith("```"):
                cleaned_title = re.sub(r'[*_#\[\]\(\)`]', '', cleaned_line)
                if cleaned_title:
                    return cleaned_title[:100]
        return None

    def _clean_markdown_formatting(self, content: str) -> str:
        lines = content.splitlines()
        cleaned_lines = []

        for i, line in enumerate(lines):
            if re.match(r"^#{1,6}", line):
                for j in range(6, 0, -1):
                    pattern = r"^#{" + str(j) + r"}([^#\s]?)(.*)"
                    if re.match(pattern, line):
                        match = re.match(pattern, line)
                        heading_text = match.group(2)
                        if heading_text and heading_text[0].islower():
                            heading_text = heading_text[0].upper() + heading_text[1:]
                        line = "#" * j + " " + match.group(1) + heading_text
                        break

            if re.match(r"^\s*[\*\-•]([^\s]|$)", line):
                indentation = re.match(r"^(\s*)", line).group(1)
                content_match = re.match(r"^\s*[\*\-•]\s*(.*)", line)
                if content_match:
                    content = content_match.group(1)
                    if content and content[0].islower():
                        content = content[0].upper() + content[1:]
                    line = f"{indentation}* {content}"
                else:
                    line = f"{indentation}* "

            if re.match(r"^\s*\d+\.[^\s]", line):
                indentation = re.match(r"^(\s*)", line).group(1)
                number = re.match(r"^\s*(\d+)\.", line).group(1)
                content_match = re.match(r"^\s*\d+\.\s*(.*)", line)
                if content_match:
                    content = content_match.group(1)
                    if content and content[0].islower():
                        content = content[0].upper() + content[1:]
                    line = f"{indentation}{number}. {content}"

            if "**" in line or "*" in line:
                line = re.sub(r"\s+\*\*", r" **", line)
                line = re.sub(r"\*\*\s+", r"** ", line)
                line = re.sub(r"\s+\*([^\*])", r" *\1", line)
                line = re.sub(r"([^\*])\*\s+", r"\1* ", line)

                if line.count("**") % 2 != 0:
                    next_line_has_marker = False
                    for j in range(i + 1, min(len(lines), i + 3)):
                        if "**" in lines[j]:
                            next_line_has_marker = True
                            break
                    if not next_line_has_marker:
                        line += "**"

                if line.count("*") % 2 != 0 and line.count("**") * 2 != line.count("*"):
                    next_line_has_marker = False
                    for j in range(i + 1, min(len(lines), i + 3)):
                        if "*" in lines[j] and "**" not in lines[j]:
                            next_line_has_marker = True
                            break
                    if not next_line_has_marker:
                        line += "*"

            if re.match(r"^\s*\|.*\|\s*$", line):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                formatted_cells = []
                for cell in cells:
                    if not re.match(r"^[-:]+$", cell) and cell and cell[0].islower():
                        cell = cell[0].upper() + cell[1:]
                    formatted_cells.append(cell)
                line = "| " + " | ".join(formatted_cells) + " |"

            cleaned_lines.append(line)

        result_lines = []
        i = 0
        while i < len(cleaned_lines):
            current_line = cleaned_lines[i]
            result_lines.append(current_line)

            if re.match(r"^#{1,6}\s", current_line) and i < len(cleaned_lines) - 1:
                next_non_empty = i + 1
                while (
                    next_non_empty < len(cleaned_lines)
                    and not cleaned_lines[next_non_empty].strip()
                ):
                    next_non_empty += 1

                if next_non_empty < len(cleaned_lines) and re.match(
                    r"^#{1,6}\s", cleaned_lines[next_non_empty]
                ):
                    result_lines.append("")
                    i = next_non_empty - 1

            i += 1

        result = "\n".join(result_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = re.sub(r"```(\w+)?\s+", r"```\1\n", result)
        result = re.sub(r"\s+```", r"\n```", result)

        return result

    def _ensure_section_spacing(self, content: str) -> str:
        lines = content.splitlines()
        result_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]
            result_lines.append(current_line)

            if re.match(r"^#{1,6}\s+", current_line) and i < len(lines) - 1:
                next_heading_index = i + 1
                found_content = False

                while next_heading_index < len(lines):
                    next_line = lines[next_heading_index]
                    if next_line.strip() and not re.match(r"^#{1,6}\s+", next_line):
                        found_content = True
                        break
                    elif re.match(r"^#{1,6}\s+", next_line):
                        break
                    next_heading_index += 1

                current_heading_level = len(re.match(r"^(#{1,6})\s+", current_line).group(1))
                if next_heading_index < len(lines) and re.match(r"^#{1,6}\s+", lines[next_heading_index]):
                    next_heading_level = len(re.match(r"^(#{1,6})\s+", lines[next_heading_index]).group(1))
                    if not found_content:
                        if next_heading_index > i + 1:
                            i = next_heading_index - 1
                        else:
                            result_lines.append("")

            i += 1

        result = "\n".join(result_lines)
        result = re.sub(r"(^#{1,6}\s+.+)\n([^#\s])", r"\1\n\n\2", result, flags=re.MULTILINE)
        result = re.sub(r"\n{3,}", r"\n\n", result)

        return result
