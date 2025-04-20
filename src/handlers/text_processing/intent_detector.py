"""
Intent detection module for Telegram bot conversations.
Detects user intentions from message content using context and natural language understanding.
"""

from typing import List, Dict, Any, Union, Tuple, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)


class IntentDetector:
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the intent detector with intent recognition capabilities.

        Args:
            debug_mode: Enable debug logging of intent detection decisions
        """
        self.debug_mode = debug_mode

        # We'll keep these as fallbacks but rely more on pattern recognition
        self.image_generation_keywords = [
            "generate image",
            "create image",
            "make an image",
            "generate a picture",
            "create a picture",
            "draw",
            "create an illustration",
            "generate art",
            "create art",
            "make a photo",
            "generate a photo",
            "image of",
            "picture of",
        ]

        self.analysis_keywords = [
            "analyze",
            "what's in",
            "what is in",
            "describe",
            "tell me about",
            "explain",
            "what do you see",
            "what can you tell me about",
            "identify",
            "recognize",
            "detect",
            "extract",
            "summarize",
        ]

        # Image generation patterns in natural language
        self.image_generation_patterns = [
            r"^(show|create|generate|make|draw|design|produce|give me|render)(\s+me)?(\s+a|\s+an)?(\s+image|\s+picture|\s+photo)?(\s+of)?(.+)$",
            r"^(.+)(\s+in|\s+with|\s+using)(\s+the\s+style\s+of|\s+style\s+of)(.+)$",
            r"^(?:imagine|visualize|paint|illustrate)(\s+a|\s+an)?(.+)$",
            r"^(?:what\s+would)(\s+a|\s+an)?(.+)(\s+look\s+like)(\?)?$",
            r"^(?:can\s+you\s+show\s+me)(\s+a|\s+an|\s+the)?(.+)(\?)?$",
        ]

        # Terms that suggest visual content
        self.visual_phrases = [
            "in the style of",
            "art style",
            "photorealistic",
            "digital art",
            "painting of",
            "drawing of",
            "illustration of",
            "render of",
            "realistic image",
            "picture with",
            "visually",
            "depicted as",
            "portrait",
            "scene",
            "view of",
            "landscape",
            "drawing",
            "painting",
            "render",
            "visualization",
            "artwork",
            "illustration",
            "design",
        ]

        # Typical artwork descriptors that suggest image generation intent
        self.art_descriptors = [
            "colorful",
            "vibrant",
            "3d",
            "high resolution",
            "4k",
            "8k",
            "hdr",
            "photorealistic",
            "anime",
            "cartoon",
            "watercolor",
            "oil painting",
            "fantasy",
            "sci-fi",
            "cinematic",
            "dramatic lighting",
            "digital",
            "surreal",
            "abstract",
            "minimalist",
            "detailed",
            "realistic",
            "stylized",
            "concept art",
            "futuristic",
            "vintage",
            "neon",
            "cyberpunk",
            "steampunk",
            "pixel art",
            "low poly",
            "isometric",
            "gothic",
            "baroque",
            "impressionist",
            "expressionist",
            "renaissance",
        ]

        # Scenes or subjects commonly requested in image generation
        self.scene_descriptions = [
            "sunset over",
            "mountain landscape",
            "portrait of",
            "cityscape",
            "futuristic city",
            "medieval castle",
            "forest scene",
            "beach scene",
            "galaxy",
            "space",
            "underwater scene",
            "animals in",
            "character wearing",
            "dragon",
            "robot",
            "spaceship",
            "mansion",
            "ruins",
            "ancient temple",
            "skyscraper",
            "village",
            "desert",
            "jungle",
            "snow",
            "rain",
            "storm",
            "night scene",
            "day scene",
            "aerial view",
            "close-up of",
            "panorama of",
        ]

        # Combined visual vocabulary for matching
        self.visual_vocabulary = set(
            [word for phrase in self.visual_phrases for word in phrase.split()]
            + [word for phrase in self.art_descriptors for word in phrase.split()]
            + [word for phrase in self.scene_descriptions for word in phrase.split()]
        )
        # Remove very common words that could cause false positives
        self.visual_vocabulary -= {
            "of",
            "in",
            "with",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "from",
            "by",
            "at",
            "on",
        }

    async def detect_user_intent(
        self, message_text: str, has_attached_media: bool
    ) -> Union[str, Dict[str, float]]:
        """
        Detects user intent using natural language understanding and context clues.

        Args:
            message_text: The user's message text
            has_attached_media: Whether the message contains media attachments

        Returns:
            Either a single intent string or a dict of intents with confidence scores
        """
        message_lower = message_text.lower()

        # Initialize confidence scores for different intents
        intent_scores = {
            "generate_image": 0.0,
            "analyze": (
                0.0 if has_attached_media else -1.0
            ),  # Only consider analysis with media
            "chat": 0.1,  # Small base score for chat as default
        }

        # Check explicit keywords first
        if any(keyword in message_lower for keyword in self.image_generation_keywords):
            intent_scores[
                "generate_image"
            ] += 0.8  # High confidence for explicit keywords

        # Check for analysis keywords with attached media
        if has_attached_media and any(
            keyword in message_lower for keyword in self.analysis_keywords
        ):
            intent_scores[
                "analyze"
            ] += 0.8  # High confidence for explicit analysis keywords

        # Check pattern matches for image generation
        if self._matches_image_generation_pattern(message_lower):
            intent_scores["generate_image"] += 0.7

        # Analyze visual vocabulary density for image generation
        visual_word_count = self._count_visual_vocabulary_matches(message_lower)
        total_words = len(message_lower.split())

        if total_words > 0:
            visual_density = visual_word_count / total_words
            intent_scores["generate_image"] += min(
                visual_density * 2.0, 0.6
            )  # Cap at 0.6 boost

        # Handle combined intents
        # For example, "analyze this image and then generate a similar one"
        if any(kw in message_lower for kw in self.analysis_keywords) and any(
            kw in message_lower for kw in self.image_generation_keywords
        ):
            # Detect combined intent
            if has_attached_media:
                intent_scores["analyze_and_generate"] = 0.9

        # Short but highly visual phrases might be image prompts
        if len(message_lower.split()) <= 6:
            descriptors_count = sum(
                1 for desc in self.art_descriptors if desc in message_lower
            )
            scenes_count = sum(
                1 for scene in self.scene_descriptions if scene in message_lower
            )

            # If it's a short message with multiple visual descriptors, likely an image prompt
            if descriptors_count + scenes_count >= 2:
                intent_scores["generate_image"] += 0.5

        # Default to media analysis if media is attached and no other strong intent
        if has_attached_media and message_text.strip() == "":
            intent_scores["analyze"] = (
                0.9  # Very high confidence for empty text + media
            )

        # Get highest scoring intent
        max_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[max_intent]

        if self.debug_mode:
            logger.debug(
                f"Intent scores: {intent_scores} | Selected: {max_intent} ({max_score:.2f}) | Message: '{message_text}'"
            )

        # We could return the full scores dictionary, but for compatibility return just the string
        # Remove this condition to get more detailed intent information
        return max_intent

    def _count_visual_vocabulary_matches(self, message_text: str) -> int:
        """Count how many words from our visual vocabulary appear in the message"""
        words = set(message_text.lower().split())
        return len(words.intersection(self.visual_vocabulary))

    def _matches_image_generation_pattern(self, message_text: str) -> bool:
        """
        Check if the message matches natural language patterns for image generation.

        Args:
            message_text: Lowercase message text

        Returns:
            True if the message likely requests image generation
        """
        # Don't skip short messages anymore - they can be valid image prompts
        # Instead, we'll assess them based on visual content

        # Check if the message starts with typical image creation verbs
        for pattern in self.image_generation_patterns:
            match = re.match(pattern, message_text)
            if match:
                # For certain patterns, we need more evidence
                # For example "show me a..." could be requesting information not an image
                if pattern.startswith("^(?:can\\s+you\\s+show\\s+me)"):
                    # If asking to show something visual or artistic, it's likely an image request
                    visual_subjects = [
                        "picture",
                        "image",
                        "photo",
                        "drawing",
                        "artwork",
                        "landscape",
                        "portrait",
                        "sunset",
                        "character",
                        "scene",
                        "visualization",
                        "design",
                    ]
                    if any(subject in message_text for subject in visual_subjects):
                        return True
                    # Otherwise it might be asking for information
                    if any(
                        info_word in message_text
                        for info_word in ["information", "data", "stats", "statistics"]
                    ):
                        return False
                else:
                    # Other patterns are more clearly image generation related
                    return True

        # Check for visual subject matter that implies image generation
        if any(phrase in message_text for phrase in self.visual_phrases):
            return True

        # If the message contains multiple art descriptors, it's likely an image request
        descriptor_count = sum(
            1 for desc in self.art_descriptors if desc in message_text
        )
        if descriptor_count >= 2:
            return True

        # Consider phrases describing visual scenes
        if any(desc in message_text for desc in self.scene_descriptions):
            # Additional check to avoid false positives with informational requests
            if not any(
                info_word in message_text
                for info_word in [
                    "tell me about",
                    "explain",
                    "what is",
                    "information on",
                    "facts",
                ]
            ):
                return True

        # If this is a short phrase (3-7 words) with high visual vocabulary density, treat as image prompt
        # This helps catch subject-first phrasing like "A knight on horseback"
        words = message_text.split()
        if 3 <= len(words) <= 7:
            visual_words = sum(1 for word in words if word in self.visual_vocabulary)
            # If 30% or more of words are visual, consider it an image prompt
            if visual_words / len(words) >= 0.3:
                return True

        return False

    def is_image_generation_request(self, message_text: str) -> bool:
        """Check if the message is specifically requesting image generation."""
        message_lower = message_text.lower()
        return any(
            keyword in message_lower for keyword in self.image_generation_keywords
        ) or self._matches_image_generation_pattern(message_lower)

    def is_analysis_request(self, message_text: str) -> bool:
        """Check if the message is specifically requesting analysis."""
        message_lower = message_text.lower()
        return any(keyword in message_lower for keyword in self.analysis_keywords)

    def extract_image_prompt(self, message_text: str) -> str:
        """
        Extract the actual image prompt from a message that's requesting image generation.
        Cleans up the prompt to focus on what should be generated.

        Args:
            message_text: Original message text

        Returns:
            Cleaned prompt for image generation
        """
        message_lower = message_text.lower()

        # Remove common prefixes that aren't part of what should be generated
        prefixes_to_remove = [
            "generate an image of",
            "generate image of",
            "create an image of",
            "create image of",
            "make an image of",
            "show me an image of",
            "can you generate",
            "can you create",
            "please generate",
            "please create",
            "please make",
            "please show me",
            "i want an image of",
            "i want a picture of",
            "i need an image of",
            "generate a picture of",
            "create a picture of",
            "draw",
            "make me",
            "show me",
            "generate",
            "create",
            "imagine",
            "visualize",
            "render",
            "paint",
            "illustrate",
            "can you draw",
            "can you show",
            "i would like to see",
            "can i see",
            "let me see",
            "could you make",
        ]

        cleaned_prompt = message_text
        for prefix in sorted(prefixes_to_remove, key=len, reverse=True):
            if message_lower.startswith(prefix):
                cleaned_prompt = message_text[len(prefix) :].strip()
                break

        # Check patterns for more complex extraction
        for pattern in self.image_generation_patterns:
            match = re.match(pattern, message_lower)
            if match:
                # Extract the relevant part based on the pattern
                # Different patterns need different group extraction
                if "style of" in pattern:
                    # For style patterns like "X in the style of Y"
                    groups = match.groups()
                    if len(groups) >= 4:
                        subject = groups[0].strip()
                        style = groups[3].strip()
                        cleaned_prompt = f"{subject} in the style of {style}"
                elif "what would" in pattern:
                    # For patterns like "what would X look like"
                    groups = match.groups()
                    if len(groups) >= 2:
                        cleaned_prompt = groups[1].strip()
                elif "can you show me" in pattern:
                    # For patterns like "can you show me X"
                    groups = match.groups()
                    if len(groups) >= 2:
                        cleaned_prompt = groups[1].strip()
                else:
                    # For standard patterns like "create X"
                    groups = match.groups()
                    if len(groups) >= 5 and groups[5]:  # The subject is in group 5
                        cleaned_prompt = groups[5].strip()
                    elif len(groups) >= 2:  # For simpler patterns
                        cleaned_prompt = groups[1].strip()

                break

        # Remove question marks from the end
        cleaned_prompt = cleaned_prompt.rstrip("?.,!")

        # If we couldn't extract a better prompt (or the cleaning didn't change anything),
        # but the message has visual characteristics, use the whole message
        if not cleaned_prompt or cleaned_prompt == message_text:
            # Count visual descriptors and scene elements
            score = sum(1 for desc in self.art_descriptors if desc in message_lower)
            score += sum(
                1 for scene in self.scene_descriptions if scene in message_lower
            )

            # If the message contains multiple visual elements, it's probably a good prompt as-is
            if score >= 2 or self._count_visual_vocabulary_matches(message_lower) >= 3:
                cleaned_prompt = message_text.strip()

        if self.debug_mode:
            logger.debug(
                f"Extracted image prompt: '{cleaned_prompt}' from '{message_text}'"
            )

        return cleaned_prompt

    def detect_combined_intent(
        self, message_text: str, has_attached_media: bool
    ) -> Dict[str, float]:
        """
        Detect if the message has multiple intents (like analyze AND generate)

        Args:
            message_text: The user's message text
            has_attached_media: Whether the message contains media attachments

        Returns:
            Dict with intent types and confidence scores
        """
        message_lower = message_text.lower()
        intents = {}

        # Check for analyzing attached media
        if has_attached_media:
            if (
                any(kw in message_lower for kw in self.analysis_keywords)
                or message_lower.strip() == ""
            ):
                intents["analyze"] = 0.8

        # Check for image generation request
        if self._matches_image_generation_pattern(message_lower) or any(
            kw in message_lower for kw in self.image_generation_keywords
        ):
            intents["generate_image"] = 0.7

        # Check for combined intents
        if "analyze" in intents and "generate_image" in intents:
            # Words that suggest sequence like "then", "after"
            if any(
                word in message_lower
                for word in [
                    "then",
                    "after",
                    "next",
                    "and",
                    "also",
                    "then",
                    "additionally",
                ]
            ):
                intents["analyze_and_generate"] = 0.9

        # Default to chat if no other intents detected
        if not intents:
            intents["chat"] = 0.6

        return intents
