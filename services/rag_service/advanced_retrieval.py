"""
Advanced RAG Retrieval Module - Ultra-Rare Query Handling

CRITICAL ENHANCEMENTS for handling ANY user input:
- Query expansion with synonyms and related terms
- Multi-query generation for complex questions
- Semantic chunking with overlap
- BM25 sparse retrieval
- Hybrid fusion with advanced weighting
- Fallback knowledge sources
- Query classification and routing
- Uncertainty quantification

This module addresses the user's requirement:
"I believe that code update needs to be done in some other core components for facing
the user's ultra rare questions and images, specifically in the multi modal files and RAG systems"
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"  # Single concept, direct question
    MODERATE = "moderate"  # Multiple concepts, needs context
    COMPLEX = "complex"  # Multi-step reasoning, rare materials
    ULTRA_RARE = "ultra_rare"  # Never-seen-before, requires creative reasoning


@dataclass
class ExpandedQuery:
    """Expanded query with multiple variations"""
    original: str
    expanded_terms: List[str]
    synonyms: Dict[str, List[str]]
    related_concepts: List[str]
    multi_queries: List[str]
    complexity: QueryComplexity
    confidence: float


class QueryExpander:
    """
    Query expansion for ultra-rare and ambiguous queries

    CRITICAL: Handles queries about materials/items never seen before
    """

    def __init__(self):
        # Material synonyms and related terms
        self.material_synonyms = {
            "plastic": ["polymer", "synthetic", "PET", "HDPE", "PVC", "polypropylene"],
            "glass": ["silica", "crystal", "transparent material"],
            "metal": ["aluminum", "steel", "iron", "copper", "alloy"],
            "paper": ["cardboard", "paperboard", "cellulose", "fiber"],
            "wood": ["timber", "lumber", "wooden"],
            "fabric": ["textile", "cloth", "fiber", "material"],
            "rubber": ["latex", "elastomer", "synthetic rubber"],
            "ceramic": ["pottery", "porcelain", "clay"],
            "electronic": ["circuit", "chip", "component", "device"],
            "battery": ["cell", "power source", "lithium-ion", "alkaline"]
        }

        # Action synonyms
        self.action_synonyms = {
            "recycle": ["reprocess", "reuse", "repurpose", "recover"],
            "dispose": ["discard", "throw away", "get rid of", "remove"],
            "upcycle": ["repurpose", "transform", "convert", "remake"],
            "compost": ["decompose", "break down", "organic recycling"],
            "donate": ["give away", "contribute", "gift"]
        }

        # Hazard-related terms
        self.hazard_terms = {
            "toxic": ["poisonous", "harmful", "dangerous", "hazardous"],
            "flammable": ["combustible", "ignitable", "fire hazard"],
            "corrosive": ["acidic", "caustic", "erosive"],
            "sharp": ["pointed", "cutting", "piercing"]
        }

    async def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand query with synonyms, related terms, and multiple variations

        CRITICAL: Handles ultra-rare queries by generating multiple search variations
        """
        query_lower = query.lower()

        # Classify query complexity
        complexity = self._classify_complexity(query)

        # Extract key terms
        key_terms = self._extract_key_terms(query)

        # Generate synonyms for each term
        synonyms = {}
        expanded_terms = []

        for term in key_terms:
            term_synonyms = self._get_synonyms(term)
            if term_synonyms:
                synonyms[term] = term_synonyms
                expanded_terms.extend(term_synonyms)

        # Generate related concepts
        related_concepts = self._generate_related_concepts(query, key_terms)

        # Generate multiple query variations
        multi_queries = self._generate_multi_queries(query, synonyms, related_concepts)

        # Calculate confidence
        confidence = self._calculate_expansion_confidence(query, expanded_terms, multi_queries)

        return ExpandedQuery(
            original=query,
            expanded_terms=expanded_terms,
            synonyms=synonyms,
            related_concepts=related_concepts,
            multi_queries=multi_queries,
            complexity=complexity,
            confidence=confidence
        )

    def _classify_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity"""
        query_lower = query.lower()
        word_count = len(query.split())

        # Ultra-rare indicators
        ultra_rare_patterns = [
            r'never seen',
            r'unknown',
            r'strange',
            r'weird',
            r'unusual',
            r'rare',
            r'not sure what',
            r'don\'t know',
            r'can\'t identify'
        ]

        if any(re.search(pattern, query_lower) for pattern in ultra_rare_patterns):
            return QueryComplexity.ULTRA_RARE

        # Complex indicators
        complex_indicators = [
            'multi' in query_lower,
            'composite' in query_lower,
            'combination' in query_lower,
            word_count > 20,
            query.count('and') > 2,
            query.count('?') > 1
        ]

        if sum(complex_indicators) >= 2:
            return QueryComplexity.COMPLEX

        # Moderate indicators
        if word_count > 10 or 'how' in query_lower or 'why' in query_lower:
            return QueryComplexity.MODERATE

        return QueryComplexity.SIMPLE


    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'can', 'could', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'this', 'that', 'these', 'those', 'to', 'from', 'in',
                      'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of'}

        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stop words and short words
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return key_terms

    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term"""
        synonyms = []

        # Check material synonyms
        for material, syns in self.material_synonyms.items():
            if term in material or material in term:
                synonyms.extend(syns)

        # Check action synonyms
        for action, syns in self.action_synonyms.items():
            if term in action or action in term:
                synonyms.extend(syns)

        # Check hazard terms
        for hazard, syns in self.hazard_terms.items():
            if term in hazard or hazard in term:
                synonyms.extend(syns)

        return list(set(synonyms))

    def _generate_related_concepts(self, query: str, key_terms: List[str]) -> List[str]:
        """Generate related concepts"""
        related = []
        query_lower = query.lower()

        # If asking about disposal, add recycling and upcycling
        if any(word in query_lower for word in ['dispose', 'throw', 'get rid']):
            related.extend(['recycling', 'upcycling', 'donation', 'composting'])

        # If asking about recycling, add material-specific info
        if 'recycle' in query_lower:
            related.extend(['sorting', 'cleaning', 'preparation', 'local facilities'])

        # If asking about upcycling, add creative ideas
        if 'upcycle' in query_lower:
            related.extend(['DIY', 'crafts', 'repurposing', 'creative reuse'])

        # If asking about hazards, add safety info
        if any(word in query_lower for word in ['safe', 'danger', 'toxic', 'hazard']):
            related.extend(['safety precautions', 'protective equipment', 'professional disposal'])

        # If asking about identification, add visual characteristics
        if any(word in query_lower for word in ['what is', 'identify', 'recognize']):
            related.extend(['material properties', 'visual features', 'common uses'])

        return list(set(related))

    def _generate_multi_queries(self, query: str, synonyms: Dict[str, List[str]],
                                related_concepts: List[str]) -> List[str]:
        """
        Generate multiple query variations for comprehensive retrieval

        CRITICAL: For ultra-rare queries, we need multiple search angles
        """
        multi_queries = [query]  # Original query

        # Generate synonym-based variations
        if synonyms:
            for original_term, term_synonyms in synonyms.items():
                for synonym in term_synonyms[:2]:  # Top 2 synonyms
                    variant = query.lower().replace(original_term, synonym)
                    if variant != query.lower():
                        multi_queries.append(variant)

        # Generate concept-based variations
        if related_concepts:
            for concept in related_concepts[:3]:  # Top 3 concepts
                multi_queries.append(f"{query} {concept}")

        # Generate decomposed queries for complex questions
        if '?' in query and len(query.split()) > 15:
            # Split into sub-questions
            sentences = re.split(r'[.?!]', query)
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    multi_queries.append(sentence.strip())

        # Limit to top 10 variations
        return multi_queries[:10]

    def _calculate_expansion_confidence(self, query: str, expanded_terms: List[str],
                                       multi_queries: List[str]) -> float:
        """Calculate confidence in query expansion"""
        confidence = 0.5  # Base confidence

        # More expanded terms = higher confidence
        if len(expanded_terms) > 5:
            confidence += 0.2
        elif len(expanded_terms) > 2:
            confidence += 0.1

        # More query variations = higher confidence
        if len(multi_queries) > 5:
            confidence += 0.2
        elif len(multi_queries) > 2:
            confidence += 0.1

        # Longer query = higher confidence (more context)
        if len(query.split()) > 10:
            confidence += 0.1

        return min(1.0, confidence)


class SemanticChunker:
    """
    Semantic chunking for better retrieval

    CRITICAL: Breaks documents into meaningful chunks with overlap
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document into overlapping semantic chunks

        CRITICAL: Preserves context across chunk boundaries
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_index": chunk_id,
                    "metadata": metadata
                })

                # Keep overlap sentences
                overlap_sentences = int(len(current_chunk) * (self.overlap / self.chunk_size))
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)
                chunk_id += 1

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_index": chunk_id,
                "metadata": metadata
            })

        logger.info(f"Chunked document {doc_id} into {len(chunks)} chunks")
        return chunks


class FallbackKnowledgeSource:
    """
    Fallback knowledge for ultra-rare queries with no retrieval results

    CRITICAL: Ensures we ALWAYS provide helpful responses, even for never-seen-before queries
    """

    def __init__(self):
        # General sustainability principles
        self.general_principles = [
            "When in doubt about recycling, check with your local waste management facility.",
            "Most materials can be recycled if properly cleaned and sorted.",
            "Upcycling is a creative way to give new life to old items.",
            "Hazardous materials should never be thrown in regular trash - contact local hazardous waste facilities.",
            "Donation is often a better option than disposal for items in good condition.",
            "Composting is ideal for organic materials like food scraps and yard waste.",
            "Electronic waste contains valuable materials and should be recycled at e-waste facilities."
        ]

        # Material-specific fallback knowledge
        self.material_fallbacks = {
            "plastic": "Plastics are typically recyclable if they have a recycling symbol (1-7). Clean and dry before recycling.",
            "glass": "Glass is infinitely recyclable. Remove lids and rinse before recycling.",
            "metal": "Most metals are highly recyclable. Aluminum and steel are particularly valuable.",
            "paper": "Paper and cardboard are recyclable if clean and dry. Remove any plastic components.",
            "electronic": "Electronics contain valuable and hazardous materials. Take to certified e-waste recyclers.",
            "battery": "Batteries contain hazardous materials. Never throw in regular trash. Take to battery recycling locations.",
            "organic": "Organic materials can be composted. Check if your area has composting programs."
        }

    async def get_fallback_response(self, query: str, query_complexity: QueryComplexity) -> Dict[str, Any]:
        """
        Generate fallback response when no retrieval results found

        CRITICAL: Provides helpful guidance even for ultra-rare queries
        """
        query_lower = query.lower()

        # Detect material type
        detected_material = None
        for material in self.material_fallbacks.keys():
            if material in query_lower:
                detected_material = material
                break

        # Build fallback response
        response = {
            "fallback": True,
            "confidence": 0.3,  # Low confidence for fallback
            "message": "I don't have specific information about this item in my knowledge base, but here's general guidance:",
            "guidance": []
        }

        # Add material-specific guidance
        if detected_material:
            response["guidance"].append(self.material_fallbacks[detected_material])
            response["confidence"] = 0.5  # Higher confidence if we detected material

        # Add general principles
        response["guidance"].extend(self.general_principles[:3])

        # Add complexity-specific advice
        if query_complexity == QueryComplexity.ULTRA_RARE:
            response["guidance"].append(
                "For unusual or rare items, I recommend: 1) Taking a photo and consulting with local waste management, "
                "2) Checking manufacturer websites for disposal instructions, 3) Contacting environmental organizations."
            )

        # Add safety warning if hazard-related
        if any(word in query_lower for word in ['toxic', 'hazard', 'danger', 'chemical', 'battery']):
            response["guidance"].insert(0,
                "⚠️ SAFETY FIRST: If this item may be hazardous, do NOT attempt disposal yourself. "
                "Contact your local hazardous waste facility or environmental protection agency."
            )
            response["safety_critical"] = True

        return response


