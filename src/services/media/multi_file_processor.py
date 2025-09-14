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
        self._register_mime_types()
    def _register_mime_types(self):
        """Register additional MIME types not in the standard library"""
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
        categorized_files = self._categorize_files(files)
        intent, specific_files = self._determine_intent_from_prompt(
            prompt, categorized_files
        )
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
            file_dict = dict(file)
            file_dict["mime"] = mime_type
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
                if self._is_code_file(file["data"]):
                    result["code"].append(file_dict)
                else:
                    result["documents"].append(file_dict)
            else:
                result["other"].append(file_dict)
        return {k: v for k, v in result.items() if v}
    def _is_code_file(self, file_data: io.BytesIO) -> bool:
        """Check if a file is likely code based on content"""
        try:
            file_data.seek(0)
            content = file_data.read(1000).decode("utf-8", errors="ignore")
            file_data.seek(0)
            code_patterns = [
                r"import\s+[\w.]+",
                r"from\s+[\w.]+\s+import",
                r"def\s+\w+\s*\(",
                r"function\s+\w+\s*\(",
                r"class\s+\w+",
                r"public\s+[static\s+]?\w+\s+\w+\(",
                r"<\?php",
                r"^\s*#include",
                r"package\s+[\w.]+",
                r"using\s+namespace",
                r"using\s+[\w.]+;",
            ]
            for pattern in code_patterns:
                if re.search(pattern, content):
                    return True
            symbols = ["=", "{", "}", "(", ")", ";", ":", "[", "]"]
            symbol_count = sum(content.count(sym) for sym in symbols)
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
        intent = "analyze"
        specific_files = None
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
        elif any(
            word in prompt_lower
            for word in ["extract", "pull out", "get", "find", "retrieve"]
        ):
            intent = "extract"
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
        elif any(
            word in prompt_lower for word in ["translate", "translation", "language"]
        ):
            intent = "translate"
        elif any(
            word in prompt_lower
            for word in ["summarize", "summary", "brief", "overview"]
        ):
            intent = "summarize"
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
        if intent == "compare":
            file_lists = list(categorized_files.values())
            if sum(len(files) for files in file_lists) >= 2:
                if len(file_lists) == 1 and len(file_lists[0]) >= 2:
                    processing_tasks.append(self._compare_files(file_lists[0], prompt))
                else:
                    processing_tasks.append(
                        self._compare_across_categories(categorized_files, prompt)
                    )
        elif intent.startswith("extract_"):
            extraction_type = intent.split("_")[1]
            for category, files in categorized_files.items():
                processing_tasks.append(
                    self._extract_from_files(files, extraction_type, prompt)
                )
        elif intent == "translate":
            for category, files in categorized_files.items():
                if category in ["documents", "code"]:
                    processing_tasks.append(self._translate_files(files, prompt))
        elif (
            intent in ["explain_code", "optimize_code", "debug_code", "review_code"]
            and "code" in categorized_files
        ):
            processing_tasks.append(
                self._process_code_files(categorized_files["code"], intent, prompt)
            )
        else:
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
                else:
                    for file in files:
                        processing_tasks.append(self._process_other_file(file, prompt))
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
            file_data.seek(0)
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
            file_data.seek(0)
            file_extension = os.path.splitext(filename)[1].lower()
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
            file_data.seek(0)
            code_content = file_data.read().decode("utf-8", errors="ignore")
            file_data.seek(0)
            enhanced_prompt = (
                f"{prompt}\n\nCode from {filename}:\n```\n{code_content}\n```"
            )
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
            for file in files[:2]:
                filename = file.get("filename", "file")
                file_data = file["data"]
                file_data.seek(0)
                try:
                    content = file_data.read().decode("utf-8", errors="ignore")
                except Exception as e:
                    content = f"[Binary content from {filename} due to error: {e}]"
                file_data.seek(0)
                file_contents.append(content)
                filenames.append(filename)
            enhanced_prompt = f"{prompt}\n\nContent from {filenames[0]}:\n```\n{file_contents[0][:3000]}\n```\n\nContent from {filenames[1]}:\n```\n{file_contents[1][:3000]}\n```"
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
            categories = list(categorized_files.keys())[:2]
            file1 = categorized_files[categories[0]][0]
            file2 = categorized_files[categories[1]][0]
            filename1 = file1.get("filename", f"{categories[0]}_file")
            filename2 = file2.get("filename", f"{categories[1]}_file")
            enhanced_prompt = f"{prompt}\n\nI'm analyzing these files for comparison: {filename1} and {filename2}"
            result1 = await self._get_file_content_or_description(file1)
            result2 = await self._get_file_content_or_description(file2)
            comparison_prompt = f"{enhanced_prompt}\n\nContent/description of {filename1}: {result1}\n\nContent/description of {filename2}: {result2}"
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
            file_data.seek(0)
            enhanced_prompt = (
                f"{prompt}\n\nExtract {extraction_type} from this content:"
            )
            try:
                mime_type = file.get("mime", "application/octet-stream")
                if mime_type.startswith("image/"):
                    extraction = await self.gemini_api.analyze_image(
                        file_data, enhanced_prompt
                    )
                else:
                    try:
                        content = file_data.read().decode("utf-8", errors="ignore")
                        file_data.seek(0)
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
        target_language = self._extract_target_language(prompt)
        for file in files:
            filename = file.get("filename", "file")
            file_data = file["data"]
            file_data.seek(0)
            try:
                content = file_data.read().decode("utf-8", errors="ignore")
                file_data.seek(0)
                if target_language:
                    translation_prompt = f"Translate the following text to {target_language}:\n```\n{content[:4000]}\n```"
                else:
                    translation_prompt = f"{prompt}:\n```\n{content[:4000]}\n```"
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
            file_data.seek(0)
            try:
                code_content = file_data.read().decode("utf-8", errors="ignore")
                file_data.seek(0)
                enhanced_prompt = f"{intent_description}\n```\n{code_content}\n```\n\nUser prompt: {prompt}"
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
        file_data.seek(0)
        mime_type = file.get("mime", "application/octet-stream")
        if mime_type.startswith("image/"):
            try:
                description = await self.gemini_api.analyze_image(
                    file_data, "Describe this image in detail"
                )
                return f"[Image description: {description}]"
            except Exception as e:
                self.logger.error(f"Error describing image {filename}: {e}")
                return f"[Image file: {filename}]"
        elif mime_type.startswith(("text/", "application/json", "application/xml")):
            try:
                content = file_data.read().decode("utf-8", errors="ignore")
                file_data.seek(0)
                return content[:4000]
            except Exception as e:
                return (
                    f"[Could not read text content from {filename} due to error: {e}]"
                )
        else:
            return f"[Binary file: {filename}]"
