"""
Local LLM Backend — LoRA Adapter Inference

Loads the fine-tuned Llama-3-8B + LoRA adapter locally.
Used when the adapter is available; falls back to OpenAI API otherwise.

Environment variables:
    LLM_ADAPTER_PATH: Path to LoRA adapter dir (default: models/llm/adapters/sustainability-expert-v1)
    LLM_BASE_MODEL: Base model name (default: meta-llama/Llama-3-8B-Instruct)
    LLM_USE_LOCAL: Force local inference (default: auto — uses local if adapter exists)
    LLM_MAX_NEW_TOKENS: Max tokens to generate (default: 1024)
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"transformers/peft not available: {e}")


class LocalLoRABackend:
    """
    Local inference using fine-tuned LoRA adapter on Llama-3-8B-Instruct.

    Supports MPS (Apple Silicon), CUDA, and CPU backends.
    Automatically merges LoRA weights on load for faster inference.
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        self.adapter_path = Path(adapter_path or os.getenv(
            "LLM_ADAPTER_PATH",
            "models/llm/adapters/sustainability-expert-v1"
        ))
        self.base_model_name = base_model or os.getenv(
            "LLM_BASE_MODEL",
            "meta-llama/Llama-3-8B-Instruct"
        )
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.device = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available and self.model is not None

    def adapter_exists(self) -> bool:
        """Check if LoRA adapter files exist on disk."""
        config_path = self.adapter_path / "adapter_config.json"
        model_path = self.adapter_path / "adapter_model.safetensors"
        model_path_bin = self.adapter_path / "adapter_model.bin"
        return config_path.exists() and (model_path.exists() or model_path_bin.exists())

    async def initialize(self) -> bool:
        """Load base model + LoRA adapter. Returns True if successful."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot initialize LocalLoRABackend — transformers not available")
            return False

        if not self.adapter_exists():
            logger.info(f"LoRA adapter not found at {self.adapter_path} — local backend disabled")
            return False

        try:
            logger.info(f"Loading base model: {self.base_model_name}")

            # Detect device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                dtype = torch.float16  # MPS doesn't support bfloat16
            else:
                self.device = torch.device("cpu")
                dtype = torch.float32

            logger.info(f"Using device: {self.device}, dtype: {dtype}")

            # Load base model
            self.model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                self.base_model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device.type != "cuda":
                self.model = self.model.to(self.device)

            # Load and merge LoRA adapter
            logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
            self.model = self.model.merge_and_unload()  # Merge for faster inference
            self.model.eval()
            logger.info("✅ LoRA adapter loaded and merged")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._available = True
            logger.info("✅ LocalLoRABackend ready for inference")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LocalLoRABackend: {e}")
            self._available = False
            return False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, int, int]:
        """
        Generate a chat completion using the local model.

        Returns: (response_text, prompt_tokens, completion_tokens)
        """
        if not self.available:
            raise RuntimeError("LocalLoRABackend not available")

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = await asyncio.to_thread(
                self.model.generate,
                **inputs,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0.01,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][prompt_tokens:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        return response_text, prompt_tokens, completion_tokens
