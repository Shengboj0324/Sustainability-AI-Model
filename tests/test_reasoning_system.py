"""
Comprehensive Integration Tests for the Reasoning System

Tests all components that were rewritten from keyword-matching to LLM-powered:
1. Prompt Templates (sustainability_prompts.py)
2. Intent Classifier (intent_classifier.py)
3. Entity Extractor (entity_extractor.py)
4. LLM Backend (llm_backend.py)
5. Prompt Builder (build_prompt)
6. End-to-End Pipeline Integration
"""

import sys
import os
import json
import asyncio
import pytest
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))


# ============================================================
# 1. PROMPT TEMPLATE TESTS
# ============================================================

class TestPromptTemplates:

    def test_imports(self):
        from llm_service.prompt_templates.sustainability_prompts import (
            MASTER_SYSTEM_PROMPT, TASK_PROMPTS, TaskType,
            build_prompt, classify_task_from_query,
            classify_intent_prompt, extract_entities_prompt,
        )
        assert len(MASTER_SYSTEM_PROMPT) > 500
        assert len(TASK_PROMPTS) >= 9

    def test_all_task_types_have_prompts(self):
        from llm_service.prompt_templates.sustainability_prompts import TASK_PROMPTS, TaskType
        for tt in TaskType:
            assert tt.value in TASK_PROMPTS, f"Missing prompt for {tt.value}"

    def test_build_prompt_basic(self):
        from llm_service.prompt_templates.sustainability_prompts import build_prompt
        msgs = build_prompt("bin_decision", "Can I recycle a pizza box?")
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert "pizza box" in msgs[-1]["content"]
        assert "MATERIAL IDENTIFICATION" in msgs[0]["content"]

    def test_build_prompt_with_context(self):
        from llm_service.prompt_templates.sustainability_prompts import build_prompt
        msgs = build_prompt(
            "bin_decision", "Can I recycle this?",
            vision_context={"class_name": "cardboard", "confidence": 0.9,
                          "material": "paper", "bin_type": "recycling"},
            rag_context=[{"source": "EPA", "content": "Cardboard is recyclable", "score": 0.8}],
            kg_context={"relationships": "cardboard -> recyclable"},
            detected_entities=[{"text": "cardboard", "type": "MATERIAL"}],
            conversation_history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ],
        )
        assert len(msgs) == 5  # system + context + 2 history + user
        # Check context injection
        context_msg = msgs[1]["content"]
        assert "VISION ANALYSIS" in context_msg
        assert "RETRIEVED KNOWLEDGE" in context_msg

    def test_build_prompt_unknown_task_falls_to_general(self):
        from llm_service.prompt_templates.sustainability_prompts import build_prompt
        msgs = build_prompt("nonexistent_task", "test query")
        assert "GENERAL SUSTAINABILITY" in msgs[0]["content"]

    def test_classification_prompts(self):
        from llm_service.prompt_templates.sustainability_prompts import (
            classify_task_from_query, classify_intent_prompt, extract_entities_prompt,
        )
        cls = classify_task_from_query("How do I dispose of motor oil?")
        assert "motor oil" in cls
        assert "bin_decision" in cls

        intent = classify_intent_prompt("Where can I donate clothes?")
        assert "donate" in intent.lower()
        assert "organization_search" in intent

        entities = extract_entities_prompt("Recycle HDPE plastic bottles with BPA")
        assert "HDPE" in entities
        assert "MATERIAL" in entities


# ============================================================
# 2. INTENT CLASSIFIER TESTS
# ============================================================

class TestIntentClassifier:

    def setup_method(self):
        from llm_service.intent_classifier import IntentClassifier
        self.classifier = IntentClassifier()

    def test_disposal_queries(self):
        from llm_service.intent_classifier import IntentCategory
        cases = [
            "Which bin does a pizza box go in?",
            "Can I recycle this aluminum can?",
            "How do I dispose of used motor oil?",
            "Is this recyclable?",
        ]
        for q in cases:
            intent, conf = self.classifier.classify(q)
            assert intent == IntentCategory.DISPOSAL_GUIDANCE, f"'{q}' -> {intent.value}"
            assert conf > 0.3

    def test_safety_queries(self):
        from llm_service.intent_classifier import IntentCategory
        cases = [
            "Is asbestos dangerous to handle?",
            "How toxic is lead paint?",
            "What PPE do I need for handling mercury?",
        ]
        for q in cases:
            intent, _ = self.classifier.classify(q)
            assert intent == IntentCategory.SAFETY_CHECK, f"'{q}' -> {intent.value}"

    def test_upcycling_queries(self):
        from llm_service.intent_classifier import IntentCategory
        cases = [
            "DIY projects with wine bottles",
            "Turn old t-shirts into something useful",
        ]
        for q in cases:
            intent, _ = self.classifier.classify(q)
            assert intent == IntentCategory.UPCYCLING_IDEAS, f"'{q}' -> {intent.value}"

    def test_chitchat(self):
        from llm_service.intent_classifier import IntentCategory
        for q in ["Hello!", "Thank you so much!", "Goodbye"]:
            intent, _ = self.classifier.classify(q)
            assert intent == IntentCategory.CHITCHAT, f"'{q}' -> {intent.value}"

    def test_edge_cases(self):
        from llm_service.intent_classifier import IntentCategory
        assert self.classifier.classify("")[0] == IntentCategory.GENERAL_QUESTION
        assert self.classifier.classify(None)[0] == IntentCategory.GENERAL_QUESTION
        assert self.classifier.classify("   ")[0] == IntentCategory.GENERAL_QUESTION



# ============================================================
# 3. ENTITY EXTRACTOR TESTS
# ============================================================

class TestEntityExtractor:

    def setup_method(self):
        from llm_service.entity_extractor import EntityExtractor
        self.extractor = EntityExtractor()

    def test_material_extraction(self):
        entities = self.extractor.extract("Can I recycle HDPE plastic bottles?")
        summary = self.extractor.get_entity_summary(entities)
        assert "MATERIAL" in summary
        found = [t.lower() for t in summary["MATERIAL"]]
        assert any("hdpe" in t for t in found)

    def test_chemical_extraction(self):
        entities = self.extractor.extract("Is BPA in plastic dangerous?")
        summary = self.extractor.get_entity_summary(entities)
        assert "CHEMICAL" in summary
        assert any("bpa" in t.lower() for t in summary["CHEMICAL"])

    def test_multi_entity(self):
        entities = self.extractor.extract(
            "How do I dispose of lithium batteries near me?"
        )
        summary = self.extractor.get_entity_summary(entities)
        assert "CHEMICAL" in summary or "ITEM" in summary
        assert "LOCATION" in summary

    def test_environmental_concept(self):
        entities = self.extractor.extract("What is the carbon footprint of aluminum?")
        summary = self.extractor.get_entity_summary(entities)
        assert "ENVIRONMENTAL_CONCEPT" in summary
        assert any("carbon footprint" in t.lower() for t in summary["ENVIRONMENTAL_CONCEPT"])

    def test_regulation_extraction(self):
        entities = self.extractor.extract("What does RCRA say about waste?")
        summary = self.extractor.get_entity_summary(entities)
        assert "REGULATION" in summary

    def test_empty_input(self):
        assert self.extractor.extract("") == []
        assert self.extractor.extract(None) == []
        assert self.extractor.extract("   ") == []

    def test_entity_to_dict(self):
        entities = self.extractor.extract("plastic bottle")
        assert len(entities) > 0
        d = entities[0].to_dict()
        assert "text" in d and "type" in d and "start" in d and "end" in d

    def test_llm_response_parsing(self):
        resp = '{"entities": [{"text": "HDPE", "type": "MATERIAL", "confidence": 0.95}]}'
        parsed = self.extractor.parse_llm_entity_response(resp, "Recycle HDPE")
        assert len(parsed) == 1
        assert parsed[0].type == "MATERIAL"

        bad = self.extractor.parse_llm_entity_response("not json", "test")
        assert bad == []


# ============================================================
# 4. LLM BACKEND TESTS
# ============================================================

class TestLLMBackend:

    def test_import(self):
        from llm_service.llm_backend import OpenAIBackend
        backend = OpenAIBackend(api_key="")
        assert not backend.available

    def test_available_with_key(self):
        from llm_service.llm_backend import OpenAIBackend, HTTPX_AVAILABLE
        if HTTPX_AVAILABLE:
            backend = OpenAIBackend(api_key="test-key-123")
            assert backend.available
        else:
            pytest.skip("httpx not installed")

    def test_unavailable_without_httpx(self):
        from llm_service.llm_backend import HTTPX_AVAILABLE
        if not HTTPX_AVAILABLE:
            from llm_service.llm_backend import OpenAIBackend
            backend = OpenAIBackend(api_key="test")
            assert not backend.available


# ============================================================
# 5. KNOWLEDGE CORPUS TESTS
# ============================================================

class TestKnowledgeCorpus:

    def test_corpus_valid(self):
        corpus_path = Path(__file__).parent.parent / "data" / "knowledge_corpus" / "sustainability_knowledge.jsonl"
        if not corpus_path.exists():
            pytest.skip("Knowledge corpus not found")

        docs = []
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    assert "id" in doc
                    assert "content" in doc
                    assert "title" in doc
                    assert "category" in doc
                    assert len(doc["content"]) > 100
                    docs.append(doc)

        assert len(docs) >= 20, f"Expected 20+ documents, got {len(docs)}"

        # Check unique IDs
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"

    def test_expert_sft_valid(self):
        sft_path = Path(__file__).parent.parent / "data" / "processed" / "llm_sft" / "expert_sft_train.jsonl"
        if not sft_path.exists():
            pytest.skip("Expert SFT data not found")

        docs = []
        with open(sft_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    msgs = doc["messages"]
                    assert len(msgs) in (2, 3)
                    if len(msgs) == 3:
                        assert msgs[0]["role"] == "system"
                        user_msg, assistant_msg = msgs[1], msgs[2]
                    else:
                        user_msg, assistant_msg = msgs[0], msgs[1]
                    assert user_msg["role"] == "user"
                    assert assistant_msg["role"] == "assistant"
                    assert len(assistant_msg["content"]) > 200
                    docs.append(doc)

        assert len(docs) >= 10, f"Expected 10+ examples, got {len(docs)}"


# ============================================================
# 6. INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """End-to-end pipeline tests (without actual LLM calls)."""

    def test_full_pipeline_classification_to_prompt(self):
        """Test the complete flow: classify → extract → build prompt."""
        from llm_service.intent_classifier import IntentClassifier
        from llm_service.entity_extractor import EntityExtractor
        from llm_service.prompt_templates.sustainability_prompts import build_prompt

        classifier = IntentClassifier()
        extractor = EntityExtractor()

        query = "Can I recycle HDPE plastic bottles with BPA-free labels?"

        # 1. Classify
        intent, confidence = classifier.classify(query)
        hints = classifier.get_context_hints(intent)

        # 2. Extract entities
        entities = extractor.extract(query)

        # 3. Build prompt
        task_type = hints["task_type"]
        msgs = build_prompt(
            task_type=task_type,
            user_query=query,
            detected_entities=[e.to_dict() for e in entities],
        )

        assert len(msgs) >= 2
        assert "system" == msgs[0]["role"]
        assert query in msgs[-1]["content"]
        assert len(msgs[0]["content"]) > 500  # Should be substantial

    def test_pipeline_with_vision_context(self):
        from llm_service.intent_classifier import IntentClassifier
        from llm_service.prompt_templates.sustainability_prompts import build_prompt

        classifier = IntentClassifier()
        intent, _ = classifier.classify("What is this item? Can I recycle it?")
        hints = classifier.get_context_hints(intent)

        msgs = build_prompt(
            task_type=hints["task_type"],
            user_query="What is this item?",
            vision_context={
                "class_name": "aluminum_can",
                "confidence": 0.95,
                "material": "aluminum",
                "bin_type": "recycling",
            },
        )

        # Should have vision context injected
        assert any("VISION ANALYSIS" in m["content"] for m in msgs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
