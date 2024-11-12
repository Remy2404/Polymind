import logging
import google.generativeai as genai
from typing import Optional, List, Dict
import asyncio
from config import GEMINI_API_KEY
from .rate_limiter import RateLimiter

class GeminiAPI:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.model.generation_config = genai.types.GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        self._initialize_safety_settings()

    def _initialize_safety_settings(self):
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

    async def get_text_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        await self.rate_limiter.acquire()
        try:
            chat = self.model.start_chat(history=context or [])
            response = await asyncio.to_thread(chat.send_message, prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise

    async def generate_code(self, language: str, prompt: str, include_explanations: bool = True) -> Dict[str, str]:
        code_prompt = f"""
        Generate {language} code for: {prompt}
        Requirements:
        - Include error handling
        - Follow best practices
        - Add comments for clarity
        - Make it production-ready
        {' Include detailed explanations.' if include_explanations else ''}
        """
        await self.rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(self.model.generate_content, code_prompt)
            return {
                'code': self._extract_code(response.text),
                'explanation': self._extract_explanation(response.text)
            }
        except Exception as e:
            logging.error(f"Code generation error: {e}")
            raise

    def _extract_code(self, response: str) -> str:
        code_blocks = []
        lines = response.split('\n')
        in_code_block = False
        current_block = []
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_block.append(line)
        return '\n\n'.join(code_blocks)

    def _extract_explanation(self, response: str) -> str:
        explanation = []
        lines = response.split('\n')
        in_code_block = False
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
            elif not in_code_block and line.strip():
                explanation.append(line)
        return '\n'.join(explanation)
