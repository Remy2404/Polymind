import logging
import asyncio
import io
import os
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from PIL import Image
from src.services.rate_limiter import RateLimiter
from src.services.mcp import MCPManager
from src.services.model_handlers.model_configs import ModelConfigurations
from src.utils.log.telegramlog import telegram_logger
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY is required")


class MediaType(Enum):
    """Supported media types for multimodal processing"""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


@dataclass
class MediaInput:
    """Represents a media input for processing"""

    type: MediaType
    data: Union[bytes, str, io.BytesIO]
    mime_type: str
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Represents a tool/function call from the model"""

    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of multimodal processing"""

    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_calls: Optional[List[ToolCall]] = None


class MediaProcessor:
    """Handles processing of different media types for Gemini"""

    MAX_IMAGE_SIZE = 4096
    MAX_FILE_SIZE = 20 * 1024 * 1024
    SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
    IMAGE_QUALITY = 85
    DOCUMENT_MIME_TYPES = {
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "txt": "text/plain",
        "csv": "text/csv",
        "md": "text/markdown",
        "html": "text/html",
        "json": "application/json",
        "xml": "text/xml",
        "py": "text/plain",
        "js": "text/plain",
        "ts": "text/plain",
        "java": "text/plain",
        "cpp": "text/plain",
        "c": "text/plain",
        "cs": "text/plain",
        "php": "text/plain",
        "rb": "text/plain",
        "go": "text/plain",
        "rs": "text/plain",
        "sql": "text/plain",
        "sh": "text/plain",
        "yaml": "text/plain",
        "yml": "text/plain",
    }

    @staticmethod
    def validate_image(image_data: Union[bytes, io.BytesIO]) -> bool:
        """Validate image format and size"""
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data
            if len(img_bytes) > MediaProcessor.MAX_FILE_SIZE:
                return False
            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.format not in MediaProcessor.SUPPORTED_IMAGE_FORMATS:
                    return False
                if img.size[0] * img.size[1] > 25000000:
                    return False
                return True
        except Exception:
            return False

    @staticmethod
    def optimize_image(image_data: Union[bytes, io.BytesIO]) -> io.BytesIO:
        """Optimize image for Gemini processing"""
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data
            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.mode in ("RGBA", "LA", "P"):
                    if img.mode == "P" and "transparency" in img.info:
                        img = img.convert("RGBA")
                    if img.mode in ("RGBA", "LA"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "RGBA":
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img, mask=img.split()[1])
                        img = background
                    else:
                        img = img.convert("RGB")
                if max(img.size) > MediaProcessor.MAX_IMAGE_SIZE:
                    ratio = MediaProcessor.MAX_IMAGE_SIZE / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                output = io.BytesIO()
                img.save(
                    output,
                    format="JPEG",
                    quality=MediaProcessor.IMAGE_QUALITY,
                    optimize=True,
                )
                output.seek(0)
                return output
        except Exception as e:
            logging.error(f"Image optimization failed: {e}")
            raise ValueError(f"Image processing failed: {e}")

    @staticmethod
    def get_image_mime_type(image_data: Union[bytes, io.BytesIO]) -> str:
        """Get MIME type for image"""
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data
            with Image.open(io.BytesIO(img_bytes)) as img:
                format_map = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "WEBP": "image/webp",
                    "GIF": "image/gif",
                }
                return format_map.get(img.format, "image/jpeg")
        except Exception:
            return "image/jpeg"

    @staticmethod
    def get_document_mime_type(filename: str) -> str:
        """Get MIME type from filename extension"""
        if not filename or "." not in filename:
            return "application/octet-stream"
        ext = filename.split(".")[-1].lower()
        return MediaProcessor.DOCUMENT_MIME_TYPES.get(ext, "application/octet-stream")

    @staticmethod
    def validate_document(file_data: Union[bytes, io.BytesIO], filename: str) -> bool:
        """Validate document for processing"""
        try:
            if isinstance(file_data, io.BytesIO):
                size = len(file_data.getvalue())
            else:
                size = len(file_data)
            if size > 50 * 1024 * 1024:
                return False
            return True
        except Exception:
            return False


class GeminiAPI:
    """
    Modern Gemini 2.5 Flash API client with multimodal and tool calling support
    Uses the latest Google Gen AI SDK for enhanced capabilities
    """

    def __init__(self, rate_limiter: RateLimiter, mcp_config_path: str = "mcp.json"):
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = rate_limiter
        self.media_processor = MediaProcessor()
        self.mcp_manager = MCPManager(mcp_config_path)
        self.mcp_tools_loaded = False
        self._tool_unsupported_models = set()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=32768,
        )
        self.logger.info(
            "Gemini 2.5 Flash API initialized with Google Gen AI SDK and MCP support"
        )

    async def initialize_mcp_tools(self) -> bool:
        """
        Initialize and load MCP tools from configured servers.
        Returns:
            True if MCP tools were loaded successfully
        """
        try:
            self.logger.info("Initializing MCP tools for Gemini...")
            telegram_logger.log_message("Initializing MCP tools for Gemini...", 0)
            success = await self.mcp_manager.load_servers()
            if success:
                self.mcp_tools_loaded = True
                server_info = self.mcp_manager.get_server_info()
                self.logger.info(
                    f"MCP tools initialized successfully for Gemini: {server_info}"
                )
                telegram_logger.log_message(
                    f"MCP tools initialized for Gemini: {len(server_info)} servers", 0
                )
                return True
            else:
                self.logger.warning("Failed to initialize MCP tools for Gemini")
                telegram_logger.log_message(
                    "Failed to initialize MCP tools for Gemini", 0
                )
                return False
        except Exception as e:
            self.logger.error(f"Error initializing MCP tools for Gemini: {str(e)}")
            telegram_logger.log_error(
                f"Error initializing MCP tools for Gemini: {str(e)}", 0
            )
            return False

    async def generate_response_with_mcp_tools(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response using Gemini with MCP tools available.
        Args:
            prompt: The user prompt
            context: Conversation context
            model: Optional model override (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
        Returns:
            Generated response or None if failed
        """
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()
        actual_model = model if model is not None else "gemini-2.5-flash"
        mcp_tools = (
            await self.mcp_manager.get_all_tools() if self.mcp_tools_loaded else []
        )
        gemini_tools = []
        for tool in mcp_tools:
            try:
                gemini_tool = self._convert_mcp_tool_to_gemini(tool)
                if gemini_tool:
                    gemini_tools.append(gemini_tool)
            except Exception as e:
                self.logger.warning(
                    f"Failed to convert MCP tool {tool.get('function', {}).get('name', 'unknown')}: {e}"
                )
        if gemini_tools:
            self.logger.info(
                f"Using {len(gemini_tools)} MCP tools for Gemini generation"
            )
            return await self._generate_with_tools(
                prompt=prompt,
                tools=gemini_tools,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            self.logger.info(
                "No MCP tools available for Gemini, using standard generation"
            )
            return await self.generate_response(
                prompt=prompt,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _convert_mcp_tool_to_gemini(self, mcp_tool: Dict[str, Any]) -> Optional[Any]:
        """
        Convert MCP tool format to Gemini-compatible tool format.
        Args:
            mcp_tool: Tool in MCP format
        Returns:
            Gemini-compatible tool or None if conversion fails
        """
        try:
            from google.genai import types

            function = mcp_tool.get("function", {})
            if not function:
                return None
            gemini_function = types.FunctionDeclaration(
                name=function.get("name", ""),
                description=function.get("description", ""),
                parameters=self._convert_mcp_parameters_to_gemini(
                    function.get("parameters", {})
                ),
            )
            return types.Tool(function_declarations=[gemini_function])
        except Exception as e:
            self.logger.error(f"Failed to convert MCP tool to Gemini format: {e}")
            return None

    def _convert_mcp_parameters_to_gemini(
        self, mcp_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert MCP parameter schema to Gemini parameter schema.
        Args:
            mcp_parameters: Parameters in MCP format
        Returns:
            Parameters in Gemini format
        """
        return mcp_parameters

    async def _generate_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        context: Optional[List[Dict]] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate content with tool calling support using Gemini.
        Args:
            prompt: The user prompt
            tools: List of tools in Gemini format
            context: Conversation context
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
        Returns:
            Generated response or None if failed
        """
        try:
            await self.rate_limiter.acquire()
            system_message = self._build_system_message(model, context, tools)
            self.logger.info(
                f"📋 System message for tool usage: {system_message[:300]}..."
            )
            content_parts = [system_message, prompt]
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
                max_output_tokens=max_tokens
                or self.generation_config.max_output_tokens,
                tools=tools,
            )
            contents = self._build_conversation_context(context, content_parts)
            response = await self._generate_with_retry(contents, model, config)
            if (
                not response
                or not hasattr(response, "candidates")
                or not response.candidates
            ):
                return None
            candidate = response.candidates[0]
            if (
                hasattr(candidate, "content")
                and candidate.content
                and hasattr(candidate.content, "parts")
            ):
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        self.logger.info(
                            f"Gemini requested tool call: {function_call.name}"
                        )
                        tool_result = await self._execute_mcp_tool(function_call)
                        if tool_result:
                            contents.append(candidate.content)
                            contents.append(
                                types.Content(
                                    role="user",
                                    parts=[
                                        types.Part.from_text(
                                            text=f"Tool result: {tool_result}"
                                        )
                                    ],
                                )
                            )
                            final_config = types.GenerateContentConfig(
                                temperature=temperature,
                                top_p=self.generation_config.top_p,
                                top_k=self.generation_config.top_k,
                                max_output_tokens=max_tokens
                                or self.generation_config.max_output_tokens,
                            )
                            final_response = await self._generate_with_retry(
                                contents, model, final_config
                            )
                            if (
                                final_response
                                and hasattr(final_response, "candidates")
                                and final_response.candidates
                            ):
                                return self._extract_response_text(
                                    final_response.candidates[0]
                                )
            return self._extract_response_text(candidate)
        except Exception as e:
            self.logger.error(f"Error in _generate_with_tools: {e}")
            return None

    async def _execute_mcp_tool(self, function_call: Any) -> Optional[str]:
        """
        Execute an MCP tool based on Gemini's function call.
        Args:
            function_call: Gemini function call object
        Returns:
            Tool execution result or None if failed
        """
        try:
            if not hasattr(function_call, "name") or not hasattr(function_call, "args"):
                return None
            tool_name = function_call.name
            tool_args = (
                dict(function_call.args) if hasattr(function_call, "args") else {}
            )
            self.logger.info(f"Executing MCP tool: {tool_name} with args: {tool_args}")
            result = await self.mcp_manager.execute_tool(tool_name, tool_args)
            if result:
                self.logger.info(f"MCP tool {tool_name} executed successfully")
                return str(result)
            else:
                self.logger.warning(f"MCP tool {tool_name} execution failed")
                return None
        except Exception as e:
            self.logger.error(
                f"Error executing MCP tool {getattr(function_call, 'name', 'unknown')}: {e}"
            )
            return None

    def _extract_response_text(self, candidate: Any) -> Optional[str]:
        """
        Extract text response from Gemini candidate.
        Args:
            candidate: Gemini response candidate
        Returns:
            Extracted text or None
        """
        try:
            if (
                not candidate
                or not hasattr(candidate, "content")
                or not candidate.content
            ):
                return None
            response_text = ""
            if hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text
            return response_text.strip() if response_text else None
        except Exception as e:
            self.logger.error(f"Error extracting response text: {e}")
            return None

    async def get_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available MCP tools in MCP format.
        Returns:
            List of available MCP tools
        """
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()
        if self.mcp_tools_loaded:
            return await self.mcp_manager.get_all_tools()
        else:
            return []

    def get_mcp_server_info(self) -> Dict[str, Any]:
        """
        Get information about connected MCP servers.
        Returns:
            Dictionary with server information
        """
        if self.mcp_tools_loaded:
            return self.mcp_manager.get_server_info()
        else:
            return {}

    def _build_system_message(
        self,
        model_id: str,
        context: Optional[List[Dict]] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Build system message with dynamic tool usage instructions for Gemini."""
        self.logger.info(
            f"🔧 Building system message for model {model_id} with {len(tools) if tools else 0} tools"
        )
        if tools:
            tool_names = []
            for tool in tools:
                if (
                    hasattr(tool, "function_declarations")
                    and tool.function_declarations
                ):
                    tool_names.extend(
                        [decl.name for decl in tool.function_declarations]
                    )
            self.logger.info(f"🔧 Available tool names: {tool_names}")
        model_config = ModelConfigurations.get_all_models().get(model_id)
        if model_config and model_config.system_message:
            base_message = model_config.system_message
        else:
            base_message = "You are Gemini, Google's advanced multimodal AI assistant. You can analyze text, images, documents, and other media types. Provide helpful, accurate, and detailed responses based on all provided content."
        context_hint = (
            " Use conversation history/context when relevant." if context else ""
        )
        tool_instructions = ""
        if tools:
            tool_categories = self._categorize_tools(tools)
            tool_names = [
                tool.function_declarations[0].name
                for tool in tools
                if hasattr(tool, "function_declarations") and tool.function_declarations
            ]
            tool_instructions = f"""
You have access to the following tools: {', '.join(tool_names)}
- **When user asks to "summary this link", "analyze this URL", "get content from", "fetch this page"**:
  - **MUST use fetch_html** to get the webpage content first
  - Then provide summary/analysis based on the fetched content
  - DO NOT use sequentialthinking for URL fetching
- **When user asks to "search for", "find information about", "look up"**:
  - Use web_search_exa or other search tools
- **When user asks about libraries, frameworks, APIs, or needs code examples**:
  - Use resolve-library-id and get-library-docs from Context7
- **When user asks complex analytical questions requiring step-by-step thinking**:
  - Use sequentialthinking for multi-step reasoning
  - NOT for content fetching or web access
1. **URL/Link Requests = fetch_html FIRST** - This is the most important rule
2. **Identify the Right Tool**: Choose the most appropriate tool based on the user's request
3. **Provide Complete Arguments**: Ensure all required parameters are included in your function calls
4. **Handle Results**: Use the tool results to provide comprehensive, accurate responses
5. **Combine Tools**: Use multiple tools when needed to provide complete answers
{chr(10).join([f"- **{category}**: {', '.join(category_tools)}" for category, category_tools in tool_categories.items()])}
- **URLs/Links → fetch_html** (ALWAYS for web content)
- **Search queries → web_search_exa**
- **Library docs → resolve-library-id + get-library-docs**
- **Complex reasoning → sequentialthinking**
- **Company research → company_research_exa**
- Always use tools when they can provide more accurate or current information
- For URLs, ALWAYS use fetch_html to get actual content before summarizing
- Provide detailed, helpful responses based on tool results
- If a tool fails, try alternative approaches or inform the user
- Do not mention tool internal details in your final response
Focus on providing the most helpful and accurate response possible using the available tools."""
        return base_message + context_hint + tool_instructions

    def _categorize_tools(self, tools: List[Any]) -> Dict[str, List[str]]:
        """
        Categorize tools by their functionality for better organization.
        Args:
            tools: List of Gemini tool objects
        Returns:
            Dictionary mapping categories to tool names
        """
        categories = {
            "Content Fetching": [],
            "Documentation": [],
            "Search & Research": [],
            "Development": [],
            "Analysis": [],
            "Communication": [],
            "Other": [],
        }
        for tool in tools:
            if hasattr(tool, "function_declarations") and tool.function_declarations:
                tool_name = tool.function_declarations[0].name.lower()
                description = (
                    tool.function_declarations[0].description.lower()
                    if tool.function_declarations[0].description
                    else ""
                )
                if any(
                    keyword in tool_name or keyword in description
                    for keyword in [
                        "fetch",
                        "html",
                        "markdown",
                        "txt",
                        "json",
                        "url",
                        "webpage",
                        "content",
                        "crawl",
                    ]
                ):
                    categories["Content Fetching"].append(
                        tool.function_declarations[0].name
                    )
                elif any(
                    keyword in tool_name or keyword in description
                    for keyword in [
                        "doc",
                        "docs",
                        "documentation",
                        "library",
                        "api",
                        "guide",
                        "tutorial",
                        "reference",
                    ]
                ):
                    categories["Documentation"].append(
                        tool.function_declarations[0].name
                    )
                elif any(
                    keyword in tool_name or keyword in description
                    for keyword in [
                        "search",
                        "find",
                        "query",
                        "lookup",
                        "research",
                        "web",
                        "browse",
                    ]
                ):
                    categories["Search & Research"].append(
                        tool.function_declarations[0].name
                    )
                elif any(
                    keyword in tool_name or keyword in description
                    for keyword in [
                        "code",
                        "dev",
                        "build",
                        "compile",
                        "test",
                        "debug",
                        "git",
                    ]
                ):
                    categories["Development"].append(tool.function_declarations[0].name)
                elif any(
                    keyword in tool_name or keyword in description
                    for keyword in [
                        "analyze",
                        "process",
                        "calculate",
                        "data",
                        "metrics",
                        "stats",
                        "thinking",
                        "sequential",
                    ]
                ):
                    categories["Analysis"].append(tool.function_declarations[0].name)
                elif any(
                    keyword in tool_name or keyword in description
                    for keyword in [
                        "chat",
                        "message",
                        "email",
                        "notify",
                        "communication",
                    ]
                ):
                    categories["Communication"].append(
                        tool.function_declarations[0].name
                    )
                else:
                    categories["Other"].append(tool.function_declarations[0].name)
        return {k: v for k, v in categories.items() if v}

    async def process_multimodal_input(
        self,
        text_prompt: str,
        media_inputs: Optional[List[MediaInput]] = None,
        context: Optional[List[Dict]] = None,
        model_name: str = "gemini-2.5-flash",
        tools: Optional[List[Union[Callable, types.Tool]]] = None,
        auto_function_calling: bool = True,
    ) -> ProcessingResult:
        """
        Process combined multimodal input with tool calling support
        Args:
            text_prompt: The main text prompt
            media_inputs: List of media inputs (images, documents, etc.)
            context: Conversation context
            model_name: Gemini model to use
            tools: List of tools/functions the model can call
            auto_function_calling: Whether to automatically execute function calls
        Returns:
            ProcessingResult with the generated response and any tool calls
        """
        try:
            await self.rate_limiter.acquire()
            content_parts = []
            if media_inputs:
                for media in media_inputs:
                    processed_content = await self._process_media_input(media)
                    if processed_content:
                        content_parts.extend(processed_content)
            content_parts.append(text_prompt)
            config = types.GenerateContentConfig(
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
                max_output_tokens=self.generation_config.max_output_tokens,
            )
            if tools:
                config.tools = tools
                if not auto_function_calling:
                    config.automatic_function_calling = (
                        types.AutomaticFunctionCallingConfig(disable=True)
                    )
            contents = self._build_conversation_context(context, content_parts)
            response = await self._generate_with_retry(contents, model_name, config)
            if response and hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                response_text = ""
                if (
                    candidate.content
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text
                tool_calls = []
                if hasattr(candidate, "function_calls") and candidate.function_calls:
                    for fc in candidate.function_calls:
                        tool_calls.append(
                            ToolCall(
                                name=fc.name,
                                args=dict(fc.args) if hasattr(fc, "args") else {},
                                id=getattr(fc, "id", None),
                            )
                        )
                return ProcessingResult(
                    success=True,
                    content=response_text.strip() if response_text else None,
                    tool_calls=tool_calls,
                    function_calls=tool_calls,
                    metadata={
                        "model": model_name,
                        "media_count": len(media_inputs) if media_inputs else 0,
                        "token_count": (
                            len(response_text.split()) if response_text else 0
                        ),
                        "has_tool_calls": len(tool_calls) > 0,
                    },
                )
            else:
                return ProcessingResult(
                    success=False, error="Empty or invalid response from Gemini API"
                )
        except Exception as e:
            self.logger.error(f"Multimodal processing failed: {e}")
            return ProcessingResult(success=False, error=f"Processing failed: {str(e)}")

    async def _process_media_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process individual media input based on its type"""
        try:
            if media.type == MediaType.IMAGE:
                return await self._process_image_input(media)
            elif media.type == MediaType.DOCUMENT:
                return await self._process_document_input(media)
            elif media.type == MediaType.AUDIO:
                return [
                    f"[Audio file: {media.filename or 'audio'} - audio processing not yet implemented]"
                ]
            elif media.type == MediaType.VIDEO:
                return [
                    f"[Video file: {media.filename or 'video'} - video processing not yet implemented]"
                ]
            else:
                return [f"[Unknown media type: {media.type.value}]"]
        except Exception as e:
            self.logger.error(f"Failed to process {media.type.value}: {e}")
            return [
                f"[Error processing {media.type.value}: {media.filename or 'unknown'}]"
            ]

    async def _process_image_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process image input for Gemini using new SDK"""
        try:
            from google.genai import types

            if not self.media_processor.validate_image(media.data):
                return [f"[Invalid image file: {media.filename or 'unknown'}]"]
            optimized_image = self.media_processor.optimize_image(media.data)
            mime_type = self.media_processor.get_image_mime_type(optimized_image)
            optimized_image.seek(0)
            image_bytes = optimized_image.getvalue()
            return [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return [f"[Image processing failed: {media.filename or 'unknown'}]"]

    async def _process_document_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process document input for Gemini using new SDK"""
        try:
            from google.genai import types

            if not media.filename:
                return ["[Document file uploaded without filename]"]
            if not self.media_processor.validate_document(media.data, media.filename):
                return [f"[Document too large or invalid: {media.filename}]"]
            if isinstance(media.data, io.BytesIO):
                media.data.seek(0)
                doc_bytes = media.data.getvalue()
            else:
                doc_bytes = media.data
            mime_type = self.media_processor.get_document_mime_type(media.filename)
            if mime_type in [
                "application/pdf",
                "text/plain",
                "text/markdown",
                "application/json",
                "text/html",
                "text/csv",
            ]:
                try:
                    uploaded_file = await self._upload_file_to_gemini_new_sdk(
                        doc_bytes, mime_type, media.filename
                    )
                    return [uploaded_file]
                except Exception as upload_error:
                    self.logger.error(f"File upload failed: {upload_error}")
                    return [types.Part.from_bytes(data=doc_bytes, mime_type=mime_type)]
            else:
                return [types.Part.from_bytes(data=doc_bytes, mime_type=mime_type)]
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return [f"[Document processing failed: {media.filename or 'unknown'}]"]

    async def _upload_file_to_gemini_new_sdk(
        self, file_bytes: bytes, mime_type: str, filename: str
    ) -> Any:
        """Upload file to Gemini using new SDK"""
        try:
            file_data = io.BytesIO(file_bytes)
            uploaded_file = await asyncio.to_thread(
                self.client.files.upload,
                file=file_data,
                mime_type=mime_type,
                display_name=filename,
            )
            return uploaded_file
        except Exception as e:
            self.logger.error(f"New SDK file upload failed: {e}")
            raise

    def _build_conversation_context(
        self, context: Optional[List[Dict]], content_parts: List[Any]
    ) -> List[Any]:
        """Build conversation context using new SDK patterns"""
        from google.genai import types

        contents = []
        if context:
            for msg in context[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    if role == "user":
                        contents.append(
                            types.Content(
                                role="user", parts=[types.Part.from_text(text=content)]
                            )
                        )
                    elif role == "assistant" or role == "model":
                        contents.append(
                            types.Content(
                                role="model", parts=[types.Part.from_text(text=content)]
                            )
                        )
        if content_parts:
            parts = []
            for part in content_parts:
                if isinstance(part, str):
                    parts.append(types.Part.from_text(text=part))
                else:
                    parts.append(part)
            contents.append(types.Content(role="user", parts=parts))
        return contents if contents else content_parts

    def get_system_message(self) -> str:
        """
        Return the system message for Gemini models.
        This is used by the prompt formatter for consistent system prompts.
        """
        return (
            "You are Gemini, Google's advanced multimodal AI assistant. You can analyze "
            "text, images, documents, and other media types. Provide helpful, accurate, "
            "and detailed responses based on all provided content."
        )

    async def _generate_with_retry(
        self, contents: List[Any], model_name: str, config: Any, max_retries: int = 3
    ) -> Any:
        """Generate content with retry logic using new SDK"""
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise e
            except ServiceUnavailable as e:
                if attempt < max_retries - 1:
                    wait_time = 1 + attempt
                    self.logger.warning(
                        f"Service unavailable, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise e
            except Exception as e:
                self.logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise e
        raise Exception("All retry attempts failed")

    async def generate_content_with_tools(
        self,
        prompt: str,
        tools: List[Union[Callable, Any]],
        context: Optional[List[Dict]] = None,
        auto_execute: bool = True,
        model_name: str = "gemini-2.5-flash",
    ) -> ProcessingResult:
        """
        Generate content with tool calling capabilities
        Args:
            prompt: The text prompt
            tools: List of functions or tool declarations
            context: Conversation context
            auto_execute: Whether to automatically execute function calls
            model_name: Model to use
        Returns:
            ProcessingResult with content and tool calls
        """
        return await self.process_multimodal_input(
            text_prompt=prompt,
            context=context,
            model_name=model_name,
            tools=tools,
            auto_function_calling=auto_execute,
        )

    async def stream_content(
        self,
        prompt: str,
        media_inputs: Optional[List[MediaInput]] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Stream content generation using new SDK"""
        try:
            from google.genai import types

            await self.rate_limiter.acquire()
            content_parts = []
            if media_inputs:
                for media in media_inputs:
                    processed_content = await self._process_media_input(media)
                    if processed_content:
                        content_parts.extend(processed_content)
            content_parts.append(types.Part.from_text(text=prompt))
            async for chunk in await asyncio.to_thread(
                self.client.models.generate_content_stream,
                model=model_name,
                contents=content_parts,
            ):
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            yield part.text
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"

    async def create_chat_session(
        self,
        model_name: str = "gemini-2.5-flash",
        tools: Optional[List[Union[Callable, Any]]] = None,
    ):
        """Create a chat session using new SDK"""
        try:
            config = None
            if tools:
                config = types.GenerateContentConfig(tools=tools)
            chat = self.client.chats.create(model=model_name, config=config)
            return chat
        except Exception as e:
            self.logger.error(f"Failed to create chat session: {e}")
            raise

    async def generate_content(
        self, prompt: str, context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate content method for backward compatibility
        Returns a dictionary with status and content for compatibility with existing tests
        """
        try:
            result = await self.process_multimodal_input(
                text_prompt=prompt, context=context
            )
            if result.success:
                return {"status": "success", "content": result.content}
            else:
                return {
                    "status": "error",
                    "content": f"Error: {result.error}",
                    "error": result.error,
                }
        except Exception as e:
            self.logger.error(f"Error in generate_content: {e}")
            return {"status": "error", "content": f"Error: {str(e)}", "error": str(e)}

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        image_context: Optional[str] = None,
        document_context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 32768,
    ) -> Optional[str]:
        """Legacy method for backward compatibility with temperature and max_tokens support"""
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            max_output_tokens=max_tokens,
        )
        content_parts = [prompt]
        if context:
            context_parts = []
            for msg in context[-5:]:
                if msg.get("role") in ["user", "assistant"]:
                    content = msg.get("content", "")
                    if content:
                        context_parts.append(f"{msg['role'].title()}: {content}")
            if context_parts:
                context_text = "\n".join(context_parts)
                content_parts.insert(0, f"Context:\n{context_text}")
        contents = []
        for part in content_parts:
            if isinstance(part, str):
                contents.append(types.Part.from_text(text=part))
            else:
                contents.append(part)
        contents = [types.Content(role="user", parts=contents)]
        try:
            response = await self._generate_with_retry(
                contents, "gemini-2.5-flash", config
            )
            if response and hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                response_text = ""
                if (
                    candidate.content
                    and hasattr(candidate.content, "parts")
                    and candidate.content.parts
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text
                return response_text.strip() if response_text else None
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error in generate_response: {e}")
            return None

    async def close(self):
        """Clean up resources and close MCP connections."""
        self.logger.info("Gemini API client closed")
        if self.mcp_tools_loaded:
            await self.mcp_manager.disconnect_all()

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name for Gemini models."""
        return "✨ Gemini"


def create_image_input(
    image_data: Union[bytes, io.BytesIO], filename: Optional[str] = None
) -> MediaInput:
    """
    Create a MediaInput object for image data.
    Args:
        image_data: Image data as bytes or BytesIO
        filename: Optional filename for the image
    Returns:
        MediaInput object for the image
    """
    if isinstance(image_data, io.BytesIO):
        image_data.seek(0)
        data = image_data.getvalue()
    else:
        data = image_data
    mime_type = "image/jpeg"
    if filename:
        if filename.lower().endswith((".png", ".PNG")):
            mime_type = "image/png"
        elif filename.lower().endswith((".webp", ".WEBP")):
            mime_type = "image/webp"
        elif filename.lower().endswith((".gif", ".GIF")):
            mime_type = "image/gif"
    return MediaInput(
        type=MediaType.IMAGE,
        data=data,
        mime_type=mime_type,
        filename=filename,
    )


def create_document_input(
    document_data: Union[bytes, io.BytesIO], filename: str
) -> MediaInput:
    """
    Create a MediaInput object for document data.
    Args:
        document_data: Document data as bytes or BytesIO
        filename: Filename of the document (used to determine MIME type)
    Returns:
        MediaInput object for the document
    """
    if isinstance(document_data, io.BytesIO):
        document_data.seek(0)
        data = document_data.getvalue()
    else:
        data = document_data
    mime_type = "application/octet-stream"
    if filename:
        ext = filename.lower().split(".")[-1]
        mime_mapping = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "txt": "text/plain",
            "csv": "text/csv",
            "md": "text/markdown",
            "html": "text/html",
            "json": "application/json",
            "xml": "text/xml",
            "py": "text/plain",
            "js": "text/plain",
            "ts": "text/plain",
            "java": "text/plain",
            "cpp": "text/plain",
            "c": "text/plain",
            "cs": "text/plain",
            "php": "text/plain",
            "rb": "text/plain",
            "go": "text/plain",
            "rs": "text/plain",
            "sql": "text/plain",
            "sh": "text/plain",
            "yaml": "text/plain",
            "yml": "text/plain",
        }
        mime_type = mime_mapping.get(ext, mime_type)
    return MediaInput(
        type=MediaType.DOCUMENT,
        data=data,
        mime_type=mime_type,
        filename=filename,
    )
