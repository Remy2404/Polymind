import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SemanticSearchManager:
    """
    Simplified SemanticSearchManager.
    Legacy vector logic removed. Now a lightweight stub or simple keyword matcher if needed.
    Most search logic is now handled directly in MemoryManager via keyword matching.
    """

    def __init__(self):
        # No heavy initialization
        pass

    async def store_message_vector(
        self, conversation_id: str, content: str, message_index: int
    ):
        """No-op: Vector storage removed for performance."""
        pass

    async def store_group_message_vector(
        self, group_id: str, content: str, message_index: int
    ):
        """No-op: Vector storage removed for performance."""
        pass

    async def semantic_search(
        self, cache_key: str, query: str, is_group: bool = False
    ) -> List[Tuple[int, float]]:
        """
        No-op: Semantic search is now handled by simple keyword matching in MemoryManager.
        Returns empty list to gracefully fallback to recent history.
        """
        return []

    def calculate_message_importance(
        self,
        message: Dict[str, Any],
        relevance_score: float,
        importance_factors: Dict[str, float],
    ) -> float:
        """
        Simple importance calculation.
        """
        try:
            base_importance = message.get("importance", 0.5)
            # Simplified formula: mostly rely on base importance + relevance
            return (base_importance * 0.7) + (relevance_score * 0.3)
        except Exception:
            return 0.5
