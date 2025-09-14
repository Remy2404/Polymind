import logging
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
logger = logging.getLogger(__name__)
class SemanticSearchManager:
    """Manages semantic search and vector operations for memory"""
    def __init__(self):
        self.message_vectors = {}
        self.group_message_vectors = {}
    async def store_message_vector(
        self, conversation_id: str, content: str, message_index: int
    ):
        """Store message vector for semantic search"""
        try:
            if conversation_id not in self.message_vectors:
                self.message_vectors[conversation_id] = {}
            words = re.findall(r"\w+", content.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            self.message_vectors[conversation_id][message_index] = {
                "content": content,
                "words": dict(word_freq),
                "length": len(content),
            }
        except Exception as e:
            logger.error(f"Error storing message vector: {e}")
    async def store_group_message_vector(
        self, group_id: str, content: str, message_index: int
    ):
        """Store group message vector for semantic search"""
        try:
            if group_id not in self.group_message_vectors:
                self.group_message_vectors[group_id] = {}
            words = re.findall(r"\w+", content.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            self.group_message_vectors[group_id][message_index] = {
                "content": content,
                "words": dict(word_freq),
                "length": len(content),
            }
        except Exception as e:
            logger.error(f"Error storing group message vector: {e}")
    async def semantic_search(
        self, cache_key: str, query: str, is_group: bool = False
    ) -> List[Tuple[int, float]]:
        """Perform semantic search on messages"""
        try:
            vector_cache = (
                self.group_message_vectors if is_group else self.message_vectors
            )
            if cache_key not in vector_cache:
                return []
            query_words = set(re.findall(r"\w+", query.lower()))
            similarities = []
            for msg_idx, vector_data in vector_cache[cache_key].items():
                msg_words = set(vector_data["words"].keys())
                intersection = len(query_words & msg_words)
                union = len(query_words | msg_words)
                if union > 0:
                    similarity = intersection / union
                    length_boost = min(vector_data["length"] / 200, 1.5)
                    final_similarity = similarity * length_boost
                    if final_similarity > 0.1:
                        similarities.append((msg_idx, final_similarity))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    def calculate_message_importance(
        self,
        message: Dict[str, Any],
        relevance_score: float,
        importance_factors: Dict[str, float],
    ) -> float:
        """Calculate combined importance score for a message"""
        import time
        current_time = time.time()
        message_time = message.get("timestamp", current_time)
        time_diff = current_time - message_time
        recency_score = max(0, 1 - (time_diff / (7 * 24 * 3600)))
        base_importance = message.get("importance", 0.5)
        media_bonus = 0.2 if message.get("message_type") != "text" else 0
        final_score = (
            relevance_score * importance_factors["relevance"]
            + recency_score * importance_factors["recency"]
            + base_importance * importance_factors["interaction"]
            + media_bonus * importance_factors["media"]
        )
        return final_score
