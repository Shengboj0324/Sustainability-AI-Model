"""Prompt templates for the LLM service."""
from .sustainability_prompts import (
    MASTER_SYSTEM_PROMPT,
    TASK_PROMPTS,
    TaskType,
    build_prompt,
    classify_task_from_query,
    classify_intent_prompt,
    extract_entities_prompt,
    VISION_CONTEXT_TEMPLATE,
    RAG_CONTEXT_TEMPLATE,
    KG_CONTEXT_TEMPLATE,
)
