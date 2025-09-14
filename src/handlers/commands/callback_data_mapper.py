"""
Callback Data Mapper for Telegram Bot
Handles mapping between long model IDs and short callback data
to comply with Telegram's 64-character limit for callback data.
"""

import hashlib
from typing import Dict, Optional


class CallbackDataMapper:
    """Maps long model IDs to short callback data and vice versa."""

    def __init__(self):
        self._id_to_short: Dict[str, str] = {}
        self._short_to_id: Dict[str, str] = {}
        self._counter = 0

    def get_callback_data(self, model_id: str, prefix: str = "model") -> str:
        """
        Get short callback data for a model ID.
        Args:
            model_id: The full model ID
            prefix: The callback prefix (e.g., "model", "category")
        Returns:
            Short callback data that fits within Telegram's limits
        """
        if not model_id:
            return f"{prefix}_unknown"
        if model_id in self._id_to_short:
            short_id = self._id_to_short[model_id]
        else:
            short_id = self._create_short_id(model_id)
            self._id_to_short[model_id] = short_id
            self._short_to_id[short_id] = model_id
        callback_data = f"{prefix}_{short_id}"
        if len(callback_data) > 64:
            hash_id = hashlib.md5(model_id.encode()).hexdigest()[:8]
            callback_data = f"{prefix}_{hash_id}"
            self._id_to_short[model_id] = hash_id
            self._short_to_id[hash_id] = model_id
        return callback_data

    def get_model_id(self, callback_data: str) -> Optional[str]:
        """
        Get the original model ID from callback data.
        Args:
            callback_data: The callback data received from Telegram
        Returns:
            The original model ID, or None if not found
        """
        parts = callback_data.split("_", 1)
        if len(parts) < 2:
            return None
        short_id = parts[1]
        return self._short_to_id.get(short_id)

    def _create_short_id(self, model_id: str) -> str:
        """Create a short, unique ID for the model."""
        if "/" in model_id:
            parts = model_id.split("/")
            provider = parts[0]
            model_part = parts[1] if len(parts) > 1 else ""
            model_part = (
                model_part.replace(":free", "")
                .replace("-instruct", "")
                .replace("-preview", "")
            )
            short_provider = provider[:4] if len(provider) > 4 else provider
            short_model = model_part[:10] if len(model_part) > 10 else model_part
            short_id = f"{short_provider}_{short_model}".replace("-", "")
        else:
            short_id = model_id[:15].replace("-", "").replace(":", "")
        base_short_id = short_id
        counter = 0
        while short_id in self._short_to_id:
            counter += 1
            short_id = f"{base_short_id}_{counter}"
        return short_id

    def get_all_mappings(self) -> Dict[str, str]:
        """Get all current mappings for debugging."""
        return self._id_to_short.copy()


callback_mapper = CallbackDataMapper()
