import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
import json
import asyncio
import re
from datetime import datetime, timedelta
import aiofiles
import os
import spacy
import uuid
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KnowledgeGraph:
    """Maintains a knowledge graph of entities from processed documents and conversations"""

    def __init__(
        self, db=None, storage_path="./data/knowledge_graph", memory_manager=None
    ):
        self.logger = logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.db = db
        self.storage_path = storage_path
        self.memory_manager = memory_manager

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)

        # Initialize entity extraction patterns
        self._init_extraction_patterns()

        # Try to load NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            # Ensure the sentencizer is in the pipeline for sentence boundary detection
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
                self.logger.info("Added 'sentencizer' to spaCy pipeline.")
            self.logger.info(
                "SpaCy model loaded successfully for advanced entity extraction"
            )
        except Exception as e:
            self.logger.warning(
                f"SpaCy model not available: {str(e)}. Using regex patterns only."
            )
            self.spacy_available = False

        # Initialize TF-IDF vectorizer for context similarity
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.context_vectors = {}

    def _init_extraction_patterns(self):
        """Initialize regex patterns for entity extraction"""
        self.patterns = {
            # People names (simplified pattern)
            "people": r"\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b",
            # Organizations
            "organizations": r"\b(?:[A-Z][a-z]*\.? )*[A-Z][a-z]*\b(?:\.com|\.org|\.gov|\.net|\.edu|Inc\.?|Corp\.?|LLC|Company|Group|Organization|Foundation|Association)\b",
            # Locations
            "locations": r"\b(?:[A-Z][a-z]+ )*[A-Z][a-z]+(?:,[ ]?[A-Z][a-z]+)?\b",
            # Dates
            "dates": r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            # Technologies
            "technologies": r"\b(?:AI|ML|NLP|API|GPT-\d+|LLM|BERT|Transformer|Python|JavaScript|TypeScript|Java|C\+\+|Kubernetes|Docker|Blockchain|TensorFlow|PyTorch|React|Angular|Vue\.js)\b",
            # Concepts
            "concepts": r"\b(?:Machine Learning|Artificial Intelligence|Natural Language Processing|Deep Learning|Computer Vision|Cloud Computing|Data Science|Big Data|Internet of Things|Blockchain|Cybersecurity|Digital Transformation|User Experience|DevOps|Agile Development)\b",
        }

        # Initialize relationship extraction patterns
        self.relationship_patterns = [
            # Person and organization
            (
                r"(?P<person>[A-Z][a-z]+ [A-Z][a-z]+) (?:works|worked) (?:for|at) (?P<organization>[A-Z][a-zA-Z ]+(?:Inc\.?|Corp\.?|LLC|Company|Group))",
                "works_for",
                "person",
                "organization",
            ),
            # Person and location
            (
                r"(?P<person>[A-Z][a-z]+ [A-Z][a-z]+) (?:lives|lived) in (?P<location>[A-Z][a-zA-Z ]+)",
                "lives_in",
                "person",
                "location",
            ),
            # Person and date
            (
                r"(?P<person>[A-Z][a-z]+ [A-Z][a-z]+) (?:was born on|born) (?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                "born_on",
                "person",
                "date",
            ),
            # Organization and technology
            (
                r"(?P<organization>[A-Z][a-zA-Z ]+(?:Inc\.?|Corp\.?|LLC|Company|Group)) (?:uses|used|developed) (?P<technology>AI|ML|NLP|API|GPT-\d+|BERT|TensorFlow|PyTorch)",
                "uses",
                "organization",
                "technology",
            ),
            # General connection pattern
            (
                r"(?P<entity1>[A-Z][a-zA-Z ]+) is (?:related to|connected to|associated with) (?P<entity2>[A-Z][a-zA-Z ]+)",
                "related_to",
                "entity1",
                "entity2",
            ),
        ]

    async def add_document_entities(
        self,
        document_id: str,
        document_content: str,
        user_id: str,
        document_type: str = "document",
    ):
        """Add document entities to the knowledge graph"""
        try:
            # Add document node
            doc_node_id = f"doc:{document_id}"
            self.graph.add_node(
                doc_node_id,
                type="document",
                doc_type=document_type,
                user_id=user_id,
                timestamp=datetime.now().isoformat(),
                content_preview=(
                    document_content[:200] + "..."
                    if len(document_content) > 200
                    else document_content
                ),
            )

            # Extract entities from document
            entities = await self.extract_entities(document_content)

            # Extract relationships between entities
            relationships = await self.extract_relationships(document_content, entities)

            # Add entity nodes and relationships
            for entity_type, items in entities.items():
                for item in items:
                    # Create unique entity ID
                    entity_id = f"{entity_type}:{self._normalize_entity_name(item)}"

                    # Check if entity already exists
                    if not self.graph.has_node(entity_id):
                        self.graph.add_node(
                            entity_id,
                            type=entity_type,
                            name=item,
                            normalized_name=self._normalize_entity_name(item),
                            first_seen=datetime.now().isoformat(),
                            documents=[document_id],
                            contexts=[document_content[:300]],
                        )
                    else:
                        # Update existing entity
                        docs = self.graph.nodes[entity_id].get("documents", [])
                        if document_id not in docs:
                            docs.append(document_id)

                        contexts = self.graph.nodes[entity_id].get("contexts", [])
                        context_preview = document_content[:300]
                        if len(contexts) < 5:  # Limit stored contexts
                            contexts.append(context_preview)

                        self.graph.nodes[entity_id]["documents"] = docs
                        self.graph.nodes[entity_id]["contexts"] = contexts
                        self.graph.nodes[entity_id][
                            "last_seen"
                        ] = datetime.now().isoformat()

                    # Connect document to entity
                    self.graph.add_edge(
                        doc_node_id,
                        entity_id,
                        relationship="contains",
                        timestamp=datetime.now().isoformat(),
                    )

            # Add relationships between entities
            for rel in relationships:
                if self.graph.has_node(rel["source"]) and self.graph.has_node(
                    rel["target"]
                ):
                    self.graph.add_edge(
                        rel["source"],
                        rel["target"],
                        relationship=rel["relationship"],
                        confidence=rel.get("confidence", 0.8),
                        source_document=document_id,
                        timestamp=datetime.now().isoformat(),
                    )

            # Update vector representation for context similarity
            await self._update_context_vectors()

            # Save to storage
            await self._save_graph()
            self.logger.info(
                f"Added entities for document {document_id} to knowledge graph"
            )

            # Return entity summary
            return self._get_entity_summary(entities, relationships)

        except Exception as e:
            self.logger.error(f"Error adding document entities: {str(e)}")
            return None

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using SpaCy if available, fallback to regex patterns"""
        entities = {}

        # Initialize entity categories
        for entity_type in self.patterns.keys():
            entities[entity_type] = []

        # Try SpaCy first if available
        if self.spacy_available:
            try:
                # Process text with SpaCy
                doc = self.nlp(text)

                # Extract entities by type
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        entities["people"].append(ent.text)
                    elif ent.label_ in ["ORG", "NORP"]:
                        entities["organizations"].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities["locations"].append(ent.text)
                    elif ent.label_ in ["DATE", "TIME"]:
                        entities["dates"].append(ent.text)

                # Keep unique entities only
                for entity_type in entities:
                    entities[entity_type] = list(set(entities[entity_type]))

                # Fallback to regex for specific types
                self._apply_regex_extraction(
                    text, entities, ["technologies", "concepts"]
                )

            except Exception as e:
                self.logger.error(
                    f"SpaCy entity extraction failed: {str(e)}, falling back to regex"
                )
                # Fallback to regex patterns
                self._apply_regex_extraction(text, entities)
        else:
            # Use regex patterns for all entity types
            self._apply_regex_extraction(text, entities)

        return {k: v for k, v in entities.items() if v}  # Remove empty categories

    def _apply_regex_extraction(
        self,
        text: str,
        entities: Dict[str, List[str]],
        entity_types: Optional[List[str]] = None,
    ):
        """Apply regex patterns to extract entities"""
        patterns_to_use = self.patterns
        if entity_types:
            patterns_to_use = {
                k: v for k, v in self.patterns.items() if k in entity_types
            }

        # Apply each pattern and collect unique matches
        for entity_type, pattern in patterns_to_use.items():
            matches = re.finditer(pattern, text)
            found_entities = list(set(match.group(0) for match in matches))

            # Add to existing list
            if entity_type in entities:
                current = set(entities[entity_type])
                current.update(found_entities)
                entities[entity_type] = list(current)
            else:
                entities[entity_type] = found_entities

    def _has_pos_tagger(self) -> bool:
        """Check if the loaded spaCy model has a POS tagger."""
        if not self.spacy_available:
            return False
        return "tagger" in self.nlp.pipe_names or "morphologizer" in self.nlp.pipe_names

    async def extract_relationships(
        self, text: str, entities: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities from text"""
        relationships = []

        # Check for matching relationship patterns
        for pattern, rel_type, src_type, tgt_type in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    match_dict = match.groupdict()
                    if src_type in match_dict and tgt_type in match_dict:
                        src_text = match_dict[src_type]
                        tgt_text = match_dict[tgt_type]

                        # Find matching entities in our extracted set
                        src_entity = self._find_best_entity_match(src_text, entities)
                        tgt_entity = self._find_best_entity_match(tgt_text, entities)

                        if src_entity and tgt_entity:
                            relationships.append(
                                {
                                    "source": src_entity["id"],
                                    "target": tgt_entity["id"],
                                    "relationship": rel_type,
                                    "confidence": src_entity["score"]
                                    * tgt_entity["score"],
                                    "text_evidence": match.group(0),
                                }
                            )
                except Exception as e:
                    self.logger.error(f"Error extracting relationship: {str(e)}")

        # If SpaCy is available, use dependency parsing for additional relationships
        if self.spacy_available:
            try:                # Process text with SpaCy
                doc = self.nlp(text)
                
                # Check if model has POS tagging capability
                has_pos = self._has_pos_tagger()

                # Extract subject-verb-object relationships
                for sent in doc.sents:
                    for token in sent:
                        # Look for verb predicates
                        if has_pos and token.pos_ == "VERB":
                            subj = None
                            obj = None

                            # Find subject and object
                            for child in token.children:
                                if child.dep_ in ["nsubj", "nsubjpass"] and not subj:
                                    subj = child
                                elif child.dep_ in ["dobj", "pobj"] and not obj:
                                    obj = child

                            if subj and obj:
                                # Get the full noun phrases
                                subj_span = self._get_span_for_token(subj, has_pos)
                                obj_span = self._get_span_for_token(obj, has_pos)

                                # Find matching entities
                                subj_entity = self._find_best_entity_match(
                                    subj_span.text, entities
                                )
                                obj_entity = self._find_best_entity_match(
                                    obj_span.text, entities
                                )

                                if subj_entity and obj_entity:
                                    relationships.append(
                                        {
                                            "source": subj_entity["id"],
                                            "target": obj_entity["id"],
                                            "relationship": token.lemma_,
                                            "confidence": 0.7,
                                            "text_evidence": sent.text,
                                        }
                                    )
                        elif not has_pos:
                            # Fallback: basic pattern matching without POS tags
                            # Look for common verb patterns in the lemma
                            if token.lemma_ in ["be", "have", "do", "say", "get", "make", "go", "know", "take", "see", "come", "think", "look", "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call"]:
                                # Simple relationship extraction without dependency parsing
                                # This is a basic fallback that won't be as accurate
                                pass
            except Exception as e:
                self.logger.error(f"SpaCy relationship extraction failed: {str(e)}")

        return relationships

    def _get_span_for_token(self, token, has_pos=True):
        """Get the complete noun phrase for a token"""
        if has_pos and token.pos_ in ["NOUN", "PROPN"]:
            # Check if token is part of a noun phrase
            for np in token.doc.noun_chunks:
                if token in np:
                    return np
        return token

    def _find_best_entity_match(
        self, text: str, entities: Dict[str, List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching entity from extracted entities"""
        best_match = None
        best_score = 0

        text_norm = self._normalize_entity_name(text)

        for entity_type, items in entities.items():
            for item in items:
                item_norm = self._normalize_entity_name(item)

                # Calculate similarity score
                if text_norm == item_norm:
                    score = 1.0
                elif text_norm in item_norm or item_norm in text_norm:
                    # One is substring of the other
                    score = 0.9
                else:
                    # Calculate Jaccard similarity between tokens
                    text_tokens = set(text_norm.split())
                    item_tokens = set(item_norm.split())

                    if not text_tokens or not item_tokens:
                        continue

                    intersection = text_tokens.intersection(item_tokens)
                    union = text_tokens.union(item_tokens)

                    score = len(intersection) / len(union)

                if score > best_score:
                    best_score = score
                    best_match = {
                        "id": f"{entity_type}:{self._normalize_entity_name(item)}",
                        "type": entity_type,
                        "name": item,
                        "score": score,
                    }

        # Only return if score is above threshold
        if best_match and best_score >= 0.5:
            return best_match

        return None

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistent identification"""
        # Convert to lowercase, remove excess whitespace and punctuation
        normalized = re.sub(r"[^\w\s]", "", name.lower())
        normalized = re.sub(r"\s+", "_", normalized.strip())
        return normalized

    async def query_related_documents(
        self, entity_name: str, user_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents related to a specific entity"""
        related_docs = []

        # Check all entity nodes for matches
        entity_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") not in ["document"] and entity_name.lower() in (
                data.get("normalized_name", "").lower() or node.lower()
            ):
                entity_nodes.append(node)

        # Find connected document nodes for each matching entity
        for entity_node in entity_nodes:
            for doc_node, edge_data in self.graph.in_edges(entity_node, data=True):
                if doc_node.startswith("doc:"):
                    # Get document data
                    doc_data = self.graph.nodes[doc_node]

                    # Filter by user_id if specified
                    if user_id and doc_data.get("user_id") != user_id:
                        continue

                    # Add to results
                    related_docs.append(
                        {
                            "document_id": doc_node.replace("doc:", ""),
                            "entity_type": self.graph.nodes[entity_node].get(
                                "type", "unknown"
                            ),
                            "entity_name": self.graph.nodes[entity_node].get(
                                "name", entity_name
                            ),
                            "timestamp": doc_data.get("timestamp", ""),
                            "user_id": doc_data.get("user_id", ""),
                            "preview": doc_data.get("content_preview", ""),
                            "doc_type": doc_data.get("doc_type", "document"),
                        }
                    )

        # Sort by timestamp (newest first) and limit results
        related_docs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return related_docs[:limit]

    async def get_entity_network(
        self, entity_name: str, depth: int = 2
    ) -> Dict[str, Any]:
        """Get a network of entities connected to the specified entity"""
        entity_nodes = []

        # Find matching entity nodes
        for node, data in self.graph.nodes(data=True):
            if data.get("type") not in ["document"] and entity_name.lower() in (
                data.get("normalized_name", "").lower() or node.lower()
            ):
                entity_nodes.append(node)

        if not entity_nodes:
            return {"nodes": [], "edges": []}

        # Start with the first matching entity
        root_node = entity_nodes[0]

        # BFS to find connected nodes up to specified depth
        nodes_to_include = set([root_node])
        current_nodes = set([root_node])

        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                # Get neighbors in both directions
                neighbors = set(
                    list(self.graph.predecessors(node))
                    + list(self.graph.successors(node))
                )
                next_nodes.update(neighbors)

            nodes_to_include.update(next_nodes)
            current_nodes = next_nodes

        # Build subgraph
        subgraph = self.graph.subgraph(nodes_to_include)

        # Convert to dictionary format
        nodes = []
        for node, data in subgraph.nodes(data=True):
            nodes.append(
                {
                    "id": node,
                    "type": data.get("type", "unknown"),
                    "name": data.get(
                        "name", node.split(":", 1)[1] if ":" in node else node
                    ),
                    "first_seen": data.get("first_seen", ""),
                    "last_seen": data.get("last_seen", data.get("first_seen", "")),
                    "doc_count": (
                        len(data.get("documents", [])) if "documents" in data else 0
                    ),
                }
            )

        edges = []
        for source, target, data in subgraph.edges(data=True):
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relationship": data.get("relationship", "related_to"),
                    "confidence": data.get("confidence", 1.0),
                    "timestamp": data.get("timestamp", ""),
                }
            )

        return {"nodes": nodes, "edges": edges}

    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get a summary of entities contained in a document"""
        doc_node = f"doc:{document_id}"
        if not self.graph.has_node(doc_node):
            return {"document_id": document_id, "entities": {}}

        # Collect entities connected to this document
        entities = {}
        for _, target in self.graph.out_edges(doc_node):
            target_data = self.graph.nodes[target]
            entity_type = target_data.get("type", "unknown")

            if entity_type not in entities:
                entities[entity_type] = []

            entities[entity_type].append(
                target_data.get(
                    "name", target.split(":", 1)[1] if ":" in target else target
                )
            )

        # Collect relationships between entities in this document
        relationships = []
        for source, target, data in self.graph.edges(data=True):
            if data.get("source_document") == document_id:
                source_data = self.graph.nodes[source]
                target_data = self.graph.nodes[target]

                relationships.append(
                    {
                        "source": source_data.get("name", source),
                        "source_type": source_data.get("type", "unknown"),
                        "relationship": data.get("relationship", "related_to"),
                        "target": target_data.get("name", target),
                        "target_type": target_data.get("type", "unknown"),
                        "confidence": data.get("confidence", 1.0),
                    }
                )

        return {
            "document_id": document_id,
            "entities": entities,
            "relationships": relationships,
            "timestamp": self.graph.nodes[doc_node].get("timestamp", ""),
            "user_id": self.graph.nodes[doc_node].get("user_id", ""),
            "doc_type": self.graph.nodes[doc_node].get("doc_type", "document"),
        }

    async def _save_graph(self):
        """Save the knowledge graph to storage"""
        try:
            # Convert graph to dictionary
            graph_data = {"nodes": [], "edges": []}

            # Save nodes
            for node, data in self.graph.nodes(data=True):
                node_data = dict(data)
                node_data["id"] = node
                graph_data["nodes"].append(node_data)

            # Save edges
            for source, target, data in self.graph.edges(data=True):
                edge_data = dict(data)
                edge_data["source"] = source
                edge_data["target"] = target
                graph_data["edges"].append(edge_data)

            # Save to file
            file_path = os.path.join(self.storage_path, "knowledge_graph.json")
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(graph_data, indent=2))

            # Save to database if available
            if self.db is not None:
                try:
                    await asyncio.to_thread(
                        self.db.knowledge_graph.update_one,
                        {"id": "main_graph"},
                        {
                            "$set": {
                                "data": graph_data,
                                "updated_at": datetime.now().isoformat(),
                            }
                        },
                        upsert=True,
                    )
                except Exception as db_error:
                    self.logger.error(f"Failed to save to database: {str(db_error)}")

        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {str(e)}")

    async def load_graph(self):
        """Load the knowledge graph from storage"""
        try:
            file_path = os.path.join(self.storage_path, "knowledge_graph.json")

            # Try loading from file first
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, "r") as f:
                    graph_data = json.loads(await f.read())

                    # Create a new graph
                    self.graph = nx.DiGraph()

                    # Add nodes
                    for node_data in graph_data["nodes"]:
                        node_id = node_data.pop("id")
                        self.graph.add_node(node_id, **node_data)

                    # Add edges
                    for edge_data in graph_data["edges"]:
                        source = edge_data.pop("source")
                        target = edge_data.pop("target")
                        self.graph.add_edge(source, target, **edge_data)

                # Update vector representation for context similarity
                await self._update_context_vectors()

                self.logger.info("Loaded knowledge graph from file storage")
                return

            # If file doesn't exist, try loading from database
            if self.db is not None:
                graph_doc = await asyncio.to_thread(
                    self.db.knowledge_graph.find_one, {"id": "main_graph"}
                )

                if graph_doc and "data" in graph_doc:
                    graph_data = graph_doc["data"]

                    # Create a new graph
                    self.graph = nx.DiGraph()

                    # Add nodes
                    for node_data in graph_data["nodes"]:
                        node_id = node_data.pop("id")
                        self.graph.add_node(node_id, **node_data)

                    # Add edges
                    for edge_data in graph_data["edges"]:
                        source = edge_data.pop("source")
                        target = edge_data.pop("target")
                        self.graph.add_edge(source, target, **edge_data)

                    # Update vector representation for context similarity
                    await self._update_context_vectors()

                    self.logger.info("Loaded knowledge graph from database")
                    return

            self.logger.info(
                "No existing knowledge graph found, starting with empty graph"
            )

        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {str(e)}")
            self.logger.info("Starting with empty knowledge graph")

    def _get_entity_summary(
        self,
        entities: Dict[str, List[str]],
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Get a summary of extracted entities and relationships"""
        entity_counts = {}
        for entity_type, items in entities.items():
            entity_counts[entity_type] = len(items)

        total_entities = sum(entity_counts.values())

        # Get top entities for each type (up to 3)
        top_entities = {}
        for entity_type, items in entities.items():
            if items:
                top_entities[entity_type] = items[: min(3, len(items))]

        result = {
            "total_entities": total_entities,
            "entity_counts": entity_counts,
            "top_entities": top_entities,
        }

        # Add relationship summary if available
        if relationships:
            relationship_types = defaultdict(int)
            for rel in relationships:
                rel_type = rel.get("relationship", "related_to")
                relationship_types[rel_type] += 1

            result["total_relationships"] = len(relationships)
            result["relationship_types"] = dict(relationship_types)

            # Get top relationships (up to 3)
            top_rels = []
            for rel in relationships[: min(3, len(relationships))]:
                source_name = self.graph.nodes[rel["source"]].get("name", rel["source"])
                target_name = self.graph.nodes[rel["target"]].get("name", rel["target"])
                rel_type = rel.get("relationship", "related_to")

                top_rels.append(f"{source_name} {rel_type} {target_name}")

            result["top_relationships"] = top_rels

        return result

    async def find_connections(self, entity1: str, entity2: str) -> List[List[str]]:
        """Find all paths connecting two entities"""
        # Find matching nodes for both entities
        nodes1 = []
        nodes2 = []

        for node, data in self.graph.nodes(data=True):
            if data.get("type") not in ["document"]:
                # Check both normalized name and node ID
                norm_name = data.get("normalized_name", "").lower()
                if entity1.lower() in norm_name or entity1.lower() in node.lower():
                    nodes1.append(node)
                if entity2.lower() in norm_name or entity2.lower() in node.lower():
                    nodes2.append(node)

        # Find all paths between any matching nodes
        all_paths = []
        for node1 in nodes1:
            for node2 in nodes2:
                try:
                    paths = list(
                        nx.all_simple_paths(self.graph, node1, node2, cutoff=4)
                    )
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    # No path found, try reverse direction
                    try:
                        paths = list(
                            nx.all_simple_paths(self.graph, node2, node1, cutoff=4)
                        )
                        all_paths.extend(
                            [path[::-1] for path in paths]
                        )  # Reverse paths
                    except nx.NetworkXNoPath:
                        continue

        # Format paths with node names and relationships
        formatted_paths = []
        for path in all_paths:
            formatted_path = []
            for i, node in enumerate(path):
                node_data = self.graph.nodes[node]
                name = node_data.get(
                    "name", node.split(":", 1)[1] if ":" in node else node
                )
                formatted_path.append(name)

                # Add relationship if not the last node
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    if self.graph.has_edge(node, next_node):
                        rel = self.graph.get_edge_data(node, next_node).get(
                            "relationship", "related_to"
                        )
                        formatted_path.append(f"--{rel}-->")
                    else:
                        # Must be reverse direction
                        rel = self.graph.get_edge_data(next_node, node).get(
                            "relationship", "related_to"
                        )
                        formatted_path.append(f"<--{rel}--")

            formatted_paths.append(formatted_path)

        return formatted_paths

    async def add_conversation_entities(
        self,
        conversation_id: str,
        user_id: str,
        message_text: str,
        is_user_message: bool,
    ):
        """Extract and add entities from conversation messages"""
        try:
            # Generate a document-like ID for this conversation message
            message_id = f"{conversation_id}_{uuid.uuid4().hex[:8]}"

            # Add as a special document type
            doc_type = "user_message" if is_user_message else "bot_message"
            await self.add_document_entities(
                message_id, message_text, user_id, doc_type
            )

            # If connected to memory manager, add relationship to conversation memory
            if self.memory_manager is not None:
                # Get conversation context from memory manager
                conversation_memory = await self.memory_manager.get_memory_for_prompt(
                    conversation_id, user_id
                )

                if conversation_memory:
                    # Add edge connecting this knowledge to the conversation context
                    conversation_context_id = f"context:{conversation_id}"

                    # Create or update the conversation context node
                    if not self.graph.has_node(conversation_context_id):
                        self.graph.add_node(
                            conversation_context_id,
                            type="conversation_context",
                            user_id=user_id,
                            conversation_id=conversation_id,
                            last_updated=datetime.now().isoformat(),
                        )
                    else:
                        self.graph.nodes[conversation_context_id][
                            "last_updated"
                        ] = datetime.now().isoformat()

                    # Connect message to conversation context
                    self.graph.add_edge(
                        f"doc:{message_id}",
                        conversation_context_id,
                        relationship="part_of",
                        timestamp=datetime.now().isoformat(),
                    )

            # Return the document ID for reference
            return message_id

        except Exception as e:
            self.logger.error(f"Error adding conversation entities: {str(e)}")
            return None

    async def find_contextually_relevant_entities(
        self, query_text: str, user_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find entities that are contextually relevant to the query text"""
        try:
            # Extract entities from query
            query_entities = await self.extract_entities(query_text)
            relevant_entities = []

            # Find directly matching entities
            for entity_type, items in query_entities.items():
                for item in items:
                    entity_nodes = []
                    norm_item = self._normalize_entity_name(item)

                    for node, data in self.graph.nodes(data=True):
                        if data.get("type") not in ["document", "conversation_context"]:
                            # Check if this entity matches or is similar
                            if (
                                data.get("normalized_name", "") == norm_item
                                or data.get("normalized_name", "").startswith(norm_item)
                                or norm_item in data.get("normalized_name", "")
                            ):
                                entity_nodes.append((node, data))

                    # Add matching entities to results
                    for node, data in entity_nodes:
                        # Filter by user_id if specified
                        if user_id and "documents" in data:
                            # Check if any documents belong to this user
                            doc_nodes = [
                                f"doc:{doc_id}" for doc_id in data.get("documents", [])
                            ]
                            user_docs = [
                                n
                                for n in doc_nodes
                                if self.graph.has_node(n)
                                and self.graph.nodes[n].get("user_id") == user_id
                            ]

                            if not user_docs:
                                continue

                        relevant_entities.append(
                            {
                                "id": node,
                                "name": data.get(
                                    "name",
                                    node.split(":", 1)[1] if ":" in node else node,
                                ),
                                "type": data.get("type", "unknown"),
                                "relevance": 1.0,  # Direct match
                                "doc_count": (
                                    len(data.get("documents", []))
                                    if "documents" in data
                                    else 0
                                ),
                                "contexts": data.get("contexts", [])[
                                    :1
                                ],  # Include one context example
                            }
                        )

            # Find contextually similar entities using TF-IDF if we have context vectors
            if self.context_vectors:
                # Vectorize the query
                query_vector = self.vectorizer.transform([query_text])

                # Calculate similarity with all context vectors
                similarities = {}
                for entity_id, vector in self.context_vectors.items():
                    sim = cosine_similarity(query_vector, vector)[0][0]
                    if sim > 0.3:  # Threshold for similarity
                        similarities[entity_id] = sim

                # Sort by similarity and get top results
                sorted_sims = sorted(
                    similarities.items(), key=lambda x: x[1], reverse=True
                )

                # Add to results if not already included
                for entity_id, sim in sorted_sims[:10]:  # Consider top 10
                    if not any(e["id"] == entity_id for e in relevant_entities):
                        if self.graph.has_node(entity_id):
                            data = self.graph.nodes[entity_id]

                            # Filter by user_id if specified
                            if user_id and "documents" in data:
                                # Check if any documents belong to this user
                                doc_nodes = [
                                    f"doc:{doc_id}"
                                    for doc_id in data.get("documents", [])
                                ]
                                user_docs = [
                                    n
                                    for n in doc_nodes
                                    if self.graph.has_node(n)
                                    and self.graph.nodes[n].get("user_id") == user_id
                                ]

                                if not user_docs:
                                    continue

                            relevant_entities.append(
                                {
                                    "id": entity_id,
                                    "name": data.get(
                                        "name",
                                        (
                                            entity_id.split(":", 1)[1]
                                            if ":" in entity_id
                                            else entity_id
                                        ),
                                    ),
                                    "type": data.get("type", "unknown"),
                                    "relevance": float(sim),  # Context similarity score
                                    "doc_count": (
                                        len(data.get("documents", []))
                                        if "documents" in data
                                        else 0
                                    ),
                                    "contexts": data.get("contexts", [])[
                                        :1
                                    ],  # Include one context example
                                }
                            )

            # Sort by relevance and limit results
            relevant_entities.sort(key=lambda x: x["relevance"], reverse=True)
            return relevant_entities[:limit]

        except Exception as e:
            self.logger.error(f"Error finding contextually relevant entities: {str(e)}")
            return []

    async def _update_context_vectors(self):
        """Update TF-IDF vectors for context similarity matching"""
        try:
            # Collect context texts for all entities
            contexts = {}
            for node, data in self.graph.nodes(data=True):
                if (
                    data.get("type") not in ["document", "conversation_context"]
                    and "contexts" in data
                ):
                    if data["contexts"]:
                        # Combine all contexts into one text
                        contexts[node] = " ".join(data["contexts"])

            if not contexts:
                return

            # Create corpus and fit vectorizer
            corpus = list(contexts.values())
            self.vectorizer.fit(corpus)

            # Transform each entity's context
            for entity_id, text in contexts.items():
                self.context_vectors[entity_id] = self.vectorizer.transform([text])

        except Exception as e:
            self.logger.error(f"Error updating context vectors: {str(e)}")

    async def get_entity_suggestions(
        self, text: str, user_id: Optional[str] = None, limit: int = 3
    ) -> Dict[str, Any]:
        """Get entity suggestions for autocompletion or context awareness"""
        try:
            # Extract entities from the text
            entities = await self.extract_entities(text)

            # Find contextually relevant entities
            relevant = await self.find_contextually_relevant_entities(
                text, user_id, limit=limit
            )

            # Find documents that might be related to the current context
            doc_suggestions = []
            for entity in relevant:
                docs = await self.query_related_documents(
                    entity["name"], user_id, limit=2
                )
                for doc in docs:
                    if not any(
                        d["document_id"] == doc["document_id"] for d in doc_suggestions
                    ):
                        doc_suggestions.append(doc)

            # Limit document suggestions
            doc_suggestions = doc_suggestions[:limit]

            # Build connections between mentioned entities
            connections = []
            extracted_entity_names = []
            for entity_type, items in entities.items():
                extracted_entity_names.extend(items)

            # Find connections between extracted entities (if multiple were found)
            if len(extracted_entity_names) >= 2:
                for i, entity1 in enumerate(
                    extracted_entity_names[:3]
                ):  # Limit to first 3 to avoid combinatorial explosion
                    for entity2 in extracted_entity_names[
                        i + 1 : min(i + 3, len(extracted_entity_names))
                    ]:
                        paths = await self.find_connections(entity1, entity2)
                        if paths:
                            connections.append(
                                {
                                    "entity1": entity1,
                                    "entity2": entity2,
                                    "paths": paths[
                                        :1
                                    ],  # Just the first path for brevity
                                }
                            )

            return {
                "extracted_entities": entities,
                "relevant_entities": relevant,
                "document_suggestions": doc_suggestions,
                "entity_connections": connections,
            }

        except Exception as e:
            self.logger.error(f"Error getting entity suggestions: {str(e)}")
            return {
                "extracted_entities": {},
                "relevant_entities": [],
                "document_suggestions": [],
                "entity_connections": [],
            }

    async def integrate_with_memory(self, memory_manager):
        """Integrate with memory manager for enhanced functionality"""
        self.memory_manager = memory_manager
        self.logger.info("Knowledge graph integrated with memory manager")

        # Traverse existing long-term memory to populate graph
        if hasattr(memory_manager, "long_term_memory"):
            for user_id, memory in memory_manager.long_term_memory.items():
                # Add facts to knowledge graph
                if "facts" in memory:
                    for fact in memory["facts"]:
                        if "content" in fact:
                            # Generate a unique ID for this fact
                            fact_id = f"memory_{user_id}_{uuid.uuid4().hex[:8]}"
                            await self.add_document_entities(
                                fact_id, fact["content"], user_id, "memory_fact"
                            )

                # Add preferences as entities
                if "preferences" in memory:
                    prefs_text = ""
                    for pref_key, pref_value in memory["preferences"].items():
                        prefs_text += f"User preference: {pref_key} = {pref_value}\n"

                    if prefs_text:
                        pref_id = f"preferences_{user_id}"
                        await self.add_document_entities(
                            pref_id, prefs_text, user_id, "user_preferences"
                        )
