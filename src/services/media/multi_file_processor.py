"""
Multi-file processor module for handling multiple files of different types simultaneously.
Intelligently processes files based on their types and user intent.
"""

import io
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from services.gemini_api import GeminiAPI
import mimetypes
import re

logger = logging.getLogger(__name__)


class MultiFileProcessor:
    """
    Process multiple files of different types simultaneously and determine their purpose.
    """

    def __init__(self, gemini_api: GeminiAPI, document_processor=None):
        """
        Initialize the multi-file processor.

        Args:
            gemini_api: Instance of GeminiAPI for processing media
            document_processor: Optional document processor instance
        """
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
        self.document_processor = document_processor

        # Register common file extensions with MIME types if not already registered
        self._register_mime_types()

    def _register_mime_types(self):
        """Register additional MIME types not in the standard library"""
        # Code file types
        mimetypes.add_type("text/x-python", ".py")
        mimetypes.add_type("text/x-java", ".java")
        mimetypes.add_type("text/x-c++", ".cpp")
        mimetypes.add_type("text/x-c", ".c")
        mimetypes.add_type("text/x-javascript", ".js")
        mimetypes.add_type("text/x-typescript", ".ts")
        mimetypes.add_type("text/x-csharp", ".cs")
        mimetypes.add_type("text/x-php", ".php")
        mimetypes.add_type("text/x-go", ".go")
        mimetypes.add_type("text/x-ruby", ".rb")
        mimetypes.add_type("text/x-rust", ".rs")
        mimetypes.add_type("text/x-swift", ".swift")
        mimetypes.add_type("text/x-kotlin", ".kt")
        mimetypes.add_type("text/x-scala", ".scala")

        # Document types
        if not mimetypes.guess_type(".docx")[0]:
            mimetypes.add_type(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".docx",
            )
            mimetypes.add_type(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xlsx",
            )
            mimetypes.add_type(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".pptx",
            )

    async def process_multiple_files(
        self, files: List[Dict], prompt: str
    ) -> Dict[str, Any]:
        """
        Process multiple files simultaneously, determining purpose for each.

        Args:
            files: List of file dictionaries, each with:
                - data: BytesIO object containing file data
                - filename: Original filename
                - type: File type indicator from Telegram (optional)
            prompt: User's prompt or caption

        Returns:
            Dict with processed results and file purposes
        """
        if not files:
            return {"error": "No files provided"}

        # Categorize files by type
        categorized_files = self._categorize_files(files)

        # Determine intent from prompt
        intent, specific_files = self._determine_intent_from_prompt(
            prompt, categorized_files
        )

        # Process files based on intent
        result = await self._process_files_by_intent(
            intent, specific_files or categorized_files, prompt
        )

        return result

    def _categorize_files(self, files: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize files based on their types.

        Returns:
            Dict with categories: 'images', 'documents', 'code', 'audio', 'video', 'other'
        """
        result = {
            "images": [],
            "documents": [],
            "code": [],
            "audio": [],
            "video": [],
            "other": [],
        }

        for file in files:
            filename = file.get("filename", "")
            file_ext = os.path.splitext(filename)[1].lower()
            mime_type = mimetypes.guess_type(filename)[0] or file.get(
                "mime", "application/octet-stream"
            )

            # Copy file dict and ensure it has mime_type
            file_dict = dict(file)
            file_dict["mime"] = mime_type

            # Categorize based on mime type and extension
            if mime_type and mime_type.startswith("image/"):
                result["images"].append(file_dict)
            elif mime_type and mime_type.startswith("audio/"):
                result["audio"].append(file_dict)
            elif mime_type and mime_type.startswith("video/"):
                result["video"].append(file_dict)
            elif (
                mime_type in ("application/pdf", "application/msword")
                or "document" in mime_type
            ):
                result["documents"].append(file_dict)
            elif file_ext in (
                ".py",
                ".java",
                ".cpp",
                ".c",
                ".js",
                ".ts",
                ".html",
                ".css",
                ".php",
                ".go",
                ".rb",
            ):
                result["code"].append(file_dict)
            elif file_ext in (".txt", ".md", ".csv", ".json", ".xml"):
                # Could be code or document depending on content
                # We'll check content to determine
                if self._is_code_file(file["data"]):
                    result["code"].append(file_dict)
                else:
                    result["documents"].append(file_dict)
            else:
                result["other"].append(file_dict)

        return {k: v for k, v in result.items() if v}  # Remove empty categories

    def _is_code_file(self, file_data: io.BytesIO) -> bool:
        """Check if a file is likely code based on content"""
        try:
            # Reset position to start
            file_data.seek(0)

            # Read first 1000 bytes to check
            content = file_data.read(1000).decode("utf-8", errors="ignore")

            # Reset position again
            file_data.seek(0)

            # Check for code indicators
            code_patterns = [
                r"import\s+[\w.]+",  # Python/Java imports
                r"from\s+[\w.]+\s+import",  # Python imports
                r"def\s+\w+\s*\(",  # Python functions
                r"function\s+\w+\s*\(",  # JavaScript functions
                r"class\s+\w+",  # Class definitions
                r"public\s+[static\s+]?\w+\s+\w+\(",  # Java methods
                r"<\?php",  # PHP
                r"^\s*#include",  # C/C++
                r"package\s+[\w.]+",  # Java/Golang packages
                r"using\s+namespace",  # C++
                r"using\s+[\w.]+;",  # C#
            ]

            for pattern in code_patterns:
                if re.search(pattern, content):
                    return True

            # Check for high density of symbols common in code
            symbols = ["=", "{", "}", "(", ")", ";", ":", "[", "]"]
            symbol_count = sum(content.count(sym) for sym in symbols)

            # If high density of symbols, likely code
            return symbol_count > len(content) * 0.05

        except Exception as e:
            self.logger.warning(f"Error determining if file is code: {e}")
            return False

    def _determine_intent_from_prompt(
        self, prompt: str, categorized_files: Dict[str, List[Dict]]
    ) -> Tuple[str, Optional[Dict[str, List[Dict]]]]:
        """
        Determine the user's intent from the prompt and available files.

        Returns:
            Tuple of (intent, specific_files)
            intent: 'analyze', 'compare', 'extract', 'translate', etc.
            specific_files: Dict of files specifically referenced in prompt, or None for all files
        """
        prompt_lower = prompt.lower()

        # Initialize intent as analyze (default)
        intent = "analyze"
        specific_files = None

        # Check for comparison intent
        if any(
            word in prompt_lower
            for word in [
                "compare",
                "difference",
                "similarities",
                "versus",
                "vs",
                "between",
            ]
        ):
            intent = "compare"

        # Check for extraction intent
        elif any(
            word in prompt_lower
            for word in ["extract", "pull out", "get", "find", "retrieve"]
        ):
            intent = "extract"

            # Extract what needs to be extracted
            extraction_target = None
            for target in [
                "text",
                "data",
                "information",
                "content",
                "code",
                "tables",
                "images",
            ]:
                if target in prompt_lower:
                    extraction_target = target
                    break

            if extraction_target:
                intent = f"extract_{extraction_target}"

        # Check for translation intent
        elif any(
            word in prompt_lower for word in ["translate", "translation", "language"]
        ):
            intent = "translate"

        # Check for summarization intent
        elif any(
            word in prompt_lower
            for word in ["summarize", "summary", "brief", "overview"]
        ):
            intent = "summarize"

        # Check for code-specific intents
        elif "code" in categorized_files:
            if any(
                word in prompt_lower for word in ["explain", "what does", "how does"]
            ):
                intent = "explain_code"
            elif any(
                word in prompt_lower for word in ["optimize", "improve", "better"]
            ):
                intent = "optimize_code"
            elif any(
                word in prompt_lower
                for word in ["debug", "fix", "error", "issue", "problem"]
            ):
                intent = "debug_code"
            elif any(
                word in prompt_lower for word in ["review", "critique", "feedback"]
            ):
                intent = "review_code"

        # Check if specific files are referenced in the prompt
        specific_files_dict = {}
        for category, files in categorized_files.items():
            for file in files:
                filename = file.get("filename", "")
                basename = os.path.basename(filename)

                if basename.lower() in prompt_lower:
                    if category not in specific_files_dict:
                        specific_files_dict[category] = []
                    specific_files_dict[category].append(file)

        if specific_files_dict:
            specific_files = specific_files_dict

        return intent, specific_files

    async def _process_files_by_intent(
        self, intent: str, categorized_files: Dict[str, List[Dict]], prompt: str
    ) -> Dict[str, Any]:
        """
        Process files based on determined intent.

        Args:
            intent: Determined processing intent
            categorized_files: Dict of categorized files
            prompt: Original user prompt

        Returns:
            Dict with processed results
        """
        results = {"intent": intent, "results": {}}
        processing_tasks = []

        # Handle different intents
        if intent == "compare":
            # Compare files - we need at least 2 files
            file_lists = list(categorized_files.values())
            if sum(len(files) for files in file_lists) >= 2:
                if len(file_lists) == 1 and len(file_lists[0]) >= 2:
                    # Compare files within the same category
                    processing_tasks.append(self._compare_files(file_lists[0], prompt))
                else:
                    # Compare across categories
                    processing_tasks.append(
                        self._compare_across_categories(categorized_files, prompt)
                    )

        elif intent.startswith("extract_"):
            # Extract specific content from files
            extraction_type = intent.split("_")[1]
            for category, files in categorized_files.items():
                processing_tasks.append(
                    self._extract_from_files(files, extraction_type, prompt)
                )

        elif intent == "translate":
            # Translate text content in files
            for category, files in categorized_files.items():
                if category in ["documents", "code"]:
                    processing_tasks.append(self._translate_files(files, prompt))

        elif (
            intent in ["explain_code", "optimize_code", "debug_code", "review_code"]
            and "code" in categorized_files
        ):
            # Process code with specific intent
            processing_tasks.append(
                self._process_code_files(categorized_files["code"], intent, prompt)
            )

        else:
            # Default: analyze each file by its type
            for category, files in categorized_files.items():
                if category == "images":
                    for file in files:
                        processing_tasks.append(self._process_image(file, prompt))

                elif category == "documents":
                    for file in files:
                        processing_tasks.append(self._process_document(file, prompt))

                elif category == "code":
                    for file in files:
                        processing_tasks.append(self._process_code_file(file, prompt))

                elif category == "audio":
                    for file in files:
                        processing_tasks.append(self._process_audio(file, prompt))

                elif category == "video":
                    for file in files:
                        processing_tasks.append(self._process_video(file, prompt))

                else:  # 'other'
                    for file in files:
                        processing_tasks.append(self._process_other_file(file, prompt))

        # Execute all tasks and collect results
        if processing_tasks:
            processed_results = await asyncio.gather(*processing_tasks)
            for result in processed_results:
                if result:
                    for key, value in result.items():
                        results["results"][key] = value

        return results

    async def _process_image(self, image_file: Dict, prompt: str) -> Dict[str, str]:
        """Process an image file"""
        try:
            filename = image_file.get("filename", "image")
            file_data = image_file["data"]
            file_data.seek(0)  # Reset position to start

            # Use Gemini to analyze the image
            analysis = await self.gemini_api.analyze_image(file_data, prompt)

            return {filename: analysis}
        except Exception as e:
            self.logger.error(
                f"Error processing image {image_file.get('filename', 'unknown')}: {e}"
            )
            return {
                image_file.get("filename", "image"): f"Error processing image: {str(e)}"
            }

    async def _process_document(
        self, document_file: Dict, prompt: str
    ) -> Dict[str, str]:
        """Process a document file"""
        try:
            filename = document_file.get("filename", "document")
            file_data = document_file["data"]
            file_data.seek(0)  # Reset position to start
            file_extension = os.path.splitext(filename)[1].lower()

            # Use document processor if available
            if self.document_processor:
                response = await self.document_processor.process_document_from_file(
                    file=file_data,
                    file_extension=file_extension.lstrip("."),
                    prompt=prompt,
                )

                if isinstance(response, dict) and "result" in response:
                    return {filename: response["result"]}
                elif isinstance(response, str):
                    return {filename: response}
                else:
                    return {
                        filename: "Processed document successfully but response format was unexpected."
                    }
            else:
                # Fallback to text extraction and analysis
                return {
                    filename: f"Please install document processing capabilities to analyze {file_extension} files."
                }
        except Exception as e:
            self.logger.error(
                f"Error processing document {document_file.get('filename', 'unknown')}: {e}"
            )
            return {
                document_file.get(
                    "filename", "document"
                ): f"Error processing document: {str(e)}"
            }

    async def _process_code_file(self, code_file: Dict, prompt: str) -> Dict[str, str]:
        """Process a code file"""
        try:
            filename = code_file.get("filename", "code_file")
            file_data = code_file["data"]
            file_data.seek(0)  # Reset position to start

            # Read the code content
            code_content = file_data.read().decode("utf-8", errors="ignore")
            file_data.seek(0)  # Reset position

            # Create an enhanced prompt with the code content
            enhanced_prompt = (
                f"{prompt}\n\nCode from {filename}:\n```\n{code_content}\n```"
            )

            # Use Gemini to analyze the code
            response = await self.gemini_api.generate_response(enhanced_prompt)

            return {filename: response or f"Error analyzing code in {filename}"}
        except Exception as e:
            self.logger.error(
                f"Error processing code file {code_file.get('filename', 'unknown')}: {e}"
            )
            return {
                code_file.get("filename", "code"): f"Error processing code: {str(e)}"
            }

    async def _process_audio(self, audio_file: Dict, prompt: str) -> Dict[str, str]:
        """Process an audio file"""
        filename = audio_file.get("filename", "audio")
        return {
            filename: f"Audio analysis is not fully supported yet. For {filename}, consider uploading as a voice message for transcription."
        }

    async def _process_video(self, video_file: Dict, prompt: str) -> Dict[str, str]:
        """Process a video file"""
        filename = video_file.get("filename", "video")
        return {
            filename: f"Video analysis is not fully supported yet. For {filename}, I can only analyze individual frames."
        }

    async def _process_other_file(
        self, other_file: Dict, prompt: str
    ) -> Dict[str, str]:
        """Process other file types"""
        filename = other_file.get("filename", "file")
        return {
            filename: "This file type is not directly supported for detailed analysis. I can try to analyze as text if you'd like."
        }

    async def _compare_files(self, files: List[Dict], prompt: str) -> Dict[str, str]:
        """Compare multiple files of the same type"""
        try:
            file_contents = []
            filenames = []

            for file in files[:2]:  # Limit to first 2 files for simplicity
                filename = file.get("filename", "file")
                file_data = file["data"]
                file_data.seek(0)  # Reset position to start

                # Read the file content
                try:
                    content = file_data.read().decode("utf-8", errors="ignore")
                except Exception as e:
                    content = f"[Binary content from {filename} due to error: {e}]"

                file_data.seek(0)  # Reset position
                file_contents.append(content)
                filenames.append(filename)

            # Create an enhanced prompt with the file contents
            enhanced_prompt = f"{prompt}\n\nContent from {filenames[0]}:\n```\n{file_contents[0][:3000]}\n```\n\nContent from {filenames[1]}:\n```\n{file_contents[1][:3000]}\n```"

            # Use Gemini to compare
            response = await self.gemini_api.generate_response(enhanced_prompt)

            return {"comparison": response or "Error comparing files"}
        except Exception as e:
            self.logger.error(f"Error comparing files: {e}")
            return {"comparison": f"Error comparing files: {str(e)}"}

    async def _compare_across_categories(
        self, categorized_files: Dict[str, List[Dict]], prompt: str
    ) -> Dict[str, str]:
        """Compare files across different categories"""
        try:
            # Get one file from each of the first 2 categories
            categories = list(categorized_files.keys())[:2]
            file1 = categorized_files[categories[0]][0]
            file2 = categorized_files[categories[1]][0]

            filename1 = file1.get("filename", f"{categories[0]}_file")
            filename2 = file2.get("filename", f"{categories[1]}_file")

            enhanced_prompt = f"{prompt}\n\nI'm analyzing these files for comparison: {filename1} and {filename2}"

            # Process each file type
            result1 = await self._get_file_content_or_description(file1)
            result2 = await self._get_file_content_or_description(file2)

            # Create comparison prompt
            comparison_prompt = f"{enhanced_prompt}\n\nContent/description of {filename1}: {result1}\n\nContent/description of {filename2}: {result2}"

            # Use Gemini for comparison
            response = await self.gemini_api.generate_response(comparison_prompt)

            return {"comparison": response or "Error comparing files across categories"}
        except Exception as e:
            self.logger.error(f"Error comparing across categories: {e}")
            return {"comparison": f"Error comparing files across categories: {str(e)}"}

    async def _extract_from_files(
        self, files: List[Dict], extraction_type: str, prompt: str
    ) -> Dict[str, str]:
        """Extract specific content from files"""
        results = {}

        for file in files:
            filename = file.get("filename", "file")
            file_data = file["data"]
            file_data.seek(0)  # Reset position to start

            enhanced_prompt = (
                f"{prompt}\n\nExtract {extraction_type} from this content:"
            )

            try:
                # Get file content based on file type
                mime_type = file.get("mime", "application/octet-stream")

                if mime_type.startswith("image/"):
                    # For images, use image analysis
                    extraction = await self.gemini_api.analyze_image(
                        file_data, enhanced_prompt
                    )
                else:
                    # For text-based files
                    try:
                        content = file_data.read().decode("utf-8", errors="ignore")
                        file_data.seek(0)  # Reset position

                        # Enhance prompt with content
                        extraction_prompt = (
                            f"{enhanced_prompt}\n```\n{content[:4000]}\n```"
                        )
                        extraction = await self.gemini_api.generate_response(
                            extraction_prompt
                        )
                    except Exception as e:
                        extraction = f"Could not extract {extraction_type} from binary file {filename} due to error: {e}"

                results[filename] = (
                    extraction or f"No {extraction_type} found in {filename}"
                )

            except Exception as e:
                self.logger.error(f"Error extracting from {filename}: {e}")
                results[filename] = f"Error extracting {extraction_type}: {str(e)}"

        return results

    async def _translate_files(self, files: List[Dict], prompt: str) -> Dict[str, str]:
        """Translate text content in files"""
        results = {}

        # Try to determine target language from prompt
        target_language = self._extract_target_language(prompt)

        for file in files:
            filename = file.get("filename", "file")
            file_data = file["data"]
            file_data.seek(0)  # Reset position to start

            try:
                # Read text content
                content = file_data.read().decode("utf-8", errors="ignore")
                file_data.seek(0)  # Reset position

                # Create translation prompt
                if target_language:
                    translation_prompt = f"Translate the following text to {target_language}:\n```\n{content[:4000]}\n```"
                else:
                    translation_prompt = f"{prompt}:\n```\n{content[:4000]}\n```"

                # Use Gemini for translation
                translation = await self.gemini_api.generate_response(
                    translation_prompt
                )

                results[filename] = translation or f"Error translating {filename}"

            except Exception as e:
                self.logger.error(f"Error translating {filename}: {e}")
                results[filename] = f"Error translating file: {str(e)}"

        return results

    def _extract_target_language(self, prompt: str) -> Optional[str]:
        """Extract target language from prompt"""
        prompt_lower = prompt.lower()

        # Common languages
        languages = {
            "english": "English",
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "italian": "Italian",
            "portuguese": "Portuguese",
            "russian": "Russian",
            "japanese": "Japanese",
            "chinese": "Chinese",
            "korean": "Korean",
            "arabic": "Arabic",
            "hindi": "Hindi",
            "turkish": "Turkish",
            "vietnamese": "Vietnamese",
        }

        # Check for patterns like "translate to X" or "in X"
        for lang_key, lang_value in languages.items():
            if (
                f"to {lang_key}" in prompt_lower
                or f"into {lang_key}" in prompt_lower
                or f"in {lang_key}" in prompt_lower
            ):
                return lang_value

        return None

    async def _process_code_files(
        self, code_files: List[Dict], intent: str, prompt: str
    ) -> Dict[str, str]:
        """Process code files with specific intent"""
        results = {}

        intent_descriptions = {
            "explain_code": "Explain in detail what this code does:",
            "optimize_code": "Optimize this code for better performance and readability:",
            "debug_code": "Find and fix any bugs or issues in this code:",
            "review_code": "Review this code and provide feedback on quality and potential improvements:",
        }

        intent_description = intent_descriptions.get(intent, "Analyze this code:")

        for file in code_files:
            filename = file.get("filename", "code_file")
            file_data = file["data"]
            file_data.seek(0)  # Reset position to start

            try:
                # Read code content
                code_content = file_data.read().decode("utf-8", errors="ignore")
                file_data.seek(0)  # Reset position

                # Create enhanced prompt
                enhanced_prompt = f"{intent_description}\n```\n{code_content}\n```\n\nUser prompt: {prompt}"

                # Process with Gemini
                response = await self.gemini_api.generate_response(enhanced_prompt)

                results[filename] = response or f"Error processing {filename}"

            except Exception as e:
                self.logger.error(f"Error processing code file {filename}: {e}")
                results[filename] = f"Error processing code: {str(e)}"

        return results

    async def _get_file_content_or_description(self, file: Dict) -> str:
        """Get file content as text or a description for non-text files"""
        filename = file.get("filename", "file")
        file_data = file["data"]
        file_data.seek(0)  # Reset position to start

        mime_type = file.get("mime", "application/octet-stream")

        if mime_type.startswith("image/"):
            # For images, get a description
            try:
                description = await self.gemini_api.analyze_image(
                    file_data, "Describe this image in detail"
                )
                return f"[Image description: {description}]"
            except Exception as e:
                self.logger.error(f"Error describing image {filename}: {e}")
                return f"[Image file: {filename}]"

        elif mime_type.startswith(("text/", "application/json", "application/xml")):
            # For text files, return content
            try:
                content = file_data.read().decode("utf-8", errors="ignore")
                file_data.seek(0)  # Reset position
                return content[:4000]  # Limit length
            except Exception as e:
                return (
                    f"[Could not read text content from {filename} due to error: {e}]"
                )

        else:
            # For other files, just return a placeholder
            return f"[Binary file: {filename}]"
