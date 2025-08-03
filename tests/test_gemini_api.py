import sys
import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import logging
from io import BytesIO
from PIL import Image

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the GeminiAPI class and dependencies
from src.services.gemini_api import GeminiAPI
from src.services.rate_limiter import RateLimiter
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TestGeminiAPI(unittest.TestCase):
    """Test case for the GeminiAPI class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Configure Gemini with API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        # Create a mock vision model and rate limiter
        self.vision_model = genai.GenerativeModel("gemini-2.0-flash")
        self.rate_limiter = RateLimiter(requests_per_minute=20)

        # Create a GeminiAPI instance
        self.gemini_api = GeminiAPI(
            vision_model=self.vision_model, rate_limiter=self.rate_limiter
        )

        # Create a small test image
        self.test_image = self._create_test_image()

    def _create_test_image(self):
        """Create a small test image for testing image analysis."""
        img = Image.new("RGB", (100, 100), color="red")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # Clean up the API object
        if hasattr(self, "gemini_api"):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.gemini_api.close())
            else:
                loop.run_until_complete(self.gemini_api.close())

    def test_gemini_api_initialization(self):
        """Test that the GeminiAPI initializes correctly."""
        self.assertIsNotNone(self.gemini_api)
        self.assertIsNotNone(self.gemini_api.vision_model)
        self.assertIsNotNone(self.gemini_api.rate_limiter)

    def test_generate_response(self):
        """Test the generate_response method."""

        async def _test():
            prompt = "Write a short haiku about coding."
            response = await self.gemini_api.generate_response(prompt)

            # Verify we got a non-empty response
            self.assertIsNotNone(response)
            self.assertTrue(len(response) > 0)
            logger.info(f"Generate response result: {response}")

            return response

        # Run the async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_test())
        print(f"\nGenerated Response: {result}")

    def test_generate_content(self):
        """Test the generate_content method."""

        async def _test():
            prompt = "Explain what artificial intelligence is in one paragraph."
            response = await self.gemini_api.generate_content(prompt)

            # Verify we got a successful response
            self.assertIsNotNone(response)
            self.assertEqual(response.get("status"), "success")
            self.assertIsNotNone(response.get("content"))
            logger.info(f"Generate content result: {response}")

            return response.get("content")

        # Run the async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_test())
        print(f"\nGenerated Content: {result}")

    def test_analyze_image(self):
        """Test the analyze_image method."""

        async def _test():
            prompt = "Describe what you see in this image."
            response = await self.gemini_api.analyze_image(self.test_image, prompt)

            # Verify we got a non-empty response
            self.assertIsNotNone(response)
            self.assertTrue(len(response) > 0)
            logger.info(f"Image analysis result: {response}")

            return response

        # Run the async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_test())
        print(f"\nImage Analysis: {result}")

    def test_call_with_circuit_breaker(self):
        """Test the circuit breaker functionality."""

        async def _test():
            # Mock an API function that succeeds
            async def mock_success(*args, **kwargs):
                return "Success"

            result = await self.gemini_api.call_with_circuit_breaker(
                "test_api", mock_success, "arg1", arg2="value"
            )
            self.assertEqual(result, "Success")

            # Mock an API function that fails
            async def mock_failure(*args, **kwargs):
                raise ValueError("API Error")

            # Should handle the exception and track the failure
            with self.assertRaises(ValueError):
                await self.gemini_api.call_with_circuit_breaker(
                    "test_api", mock_failure, "arg1", arg2="value"
                )

            # Check that the failure was tracked
            self.assertEqual(self.gemini_api.test_api_failures, 1)

            return True

        # Run the async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_test())
        self.assertTrue(result)


def run_tests():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
