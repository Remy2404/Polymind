import asyncio
import unittest
from io import BytesIO
from PIL import Image
from types import SimpleNamespace
from unittest.mock import patch

from src.ai.providers.gemini_api import GeminiAPI
from src.services.rate_limiter import RateLimiter


class TestGeminiAPI(unittest.TestCase):
    """Test case for the GeminiAPI class."""

    def setUp(self):
        """Use the real provider adapter with a deterministic SDK client."""
        self.vision_model = object()
        self.rate_limiter = RateLimiter(requests_per_minute=20)
        self.generate_calls = []

        def generate_content(**kwargs):
            self.generate_calls.append(kwargs)
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[SimpleNamespace(text="A generated answer")]),
                        function_calls=[],
                    )
                ]
            )

        fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch("src.ai.providers.gemini_api.genai.Client", return_value=fake_client):
            self.gemini_api = GeminiAPI(
                vision_model=self.vision_model,
                rate_limiter=self.rate_limiter,
                api_key="test-key",
            )
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
            asyncio.run(self.gemini_api.close())

    def test_gemini_api_initialization(self):
        """Test that the GeminiAPI initializes correctly."""
        self.assertIsNotNone(self.gemini_api)
        self.assertIsNotNone(self.gemini_api.vision_model)
        self.assertIsNotNone(self.gemini_api.rate_limiter)

    def test_generate_response(self):
        result = asyncio.run(self.gemini_api.generate_response("Write a haiku"))
        self.assertEqual(result, "A generated answer")
        self.assertEqual(self.generate_calls[0]["model"], "gemini-3.8-flash")

    def test_generate_content(self):
        result = asyncio.run(self.gemini_api.generate_content("Explain AI"))
        self.assertEqual(result, {"status": "success", "content": "A generated answer"})
        self.assertEqual(self.generate_calls[0]["model"], "gemini-3.8-flash")

    def test_analyze_image(self):
        result = asyncio.run(
            self.gemini_api.analyze_image(self.test_image, "Describe this image")
        )
        self.assertEqual(result, "A generated answer")
        self.assertEqual(self.generate_calls[0]["model"], "gemini-3.8-flash")
        parts = self.generate_calls[0]["contents"][-1].parts
        self.assertTrue(any(part.inline_data for part in parts))

    def test_analyze_image_rejects_invalid_data_before_provider_call(self):
        result = asyncio.run(self.gemini_api.analyze_image(b"invalid", "Describe this"))
        self.assertEqual(result, "Image analysis unavailable. Invalid image.")
        self.assertEqual(self.generate_calls, [])

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
        result = asyncio.run(_test())
        self.assertTrue(result)


def run_tests():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
