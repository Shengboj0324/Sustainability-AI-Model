"""
LLM Service - Domain-specialized language model for sustainability
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReleAF AI LLM Service",
    description="Domain-specialized language model service",
    version="0.1.0"
)


class LLMRequest(BaseModel):
    """LLM request"""
    messages: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class LLMResponse(BaseModel):
    """LLM response"""
    response: str
    usage: Dict[str, int]
    model: str


class LLMService:
    """LLM inference service"""
    
    def __init__(self, config_path: str = "configs/llm_sft.yaml"):
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.system_prompt = self.config.get("system_prompt", "")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}
    
    def load_model(self):
        """Load model and tokenizer"""
        logger.info("Loading LLM model...")
        
        base_model_name = self.config["model"]["base_model_name"]
        adapter_path = self.config["training"]["output_dir"]
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter if available
        try:
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
        except Exception as e:
            logger.warning(f"Could not load adapter: {e}. Using base model only.")
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def format_messages(self, messages: List[Dict[str, str]], context: Optional[Dict] = None) -> str:
        """Format messages for the model"""
        # Add system prompt if not present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                messages.insert(1, {"role": "system", "content": context_str})
        
        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted += f"<|{role}|>\n{content}\n"
            formatted += "<|assistant|>\n"
            return formatted
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information"""
        parts = []
        
        # Vision results
        if "vision_service_classify" in context:
            vision = context["vision_service_classify"]
            parts.append(f"Image analysis: {vision}")
        
        # RAG results
        if "rag_service_retrieve_knowledge" in context:
            rag = context["rag_service_retrieve_knowledge"]
            parts.append(f"Relevant information: {rag}")
        
        # KG results
        if "kg_service_query_relationships" in context:
            kg = context["kg_service_query_relationships"]
            parts.append(f"Related concepts: {kg}")
        
        return "\n\n".join(parts) if parts else ""
    
    @torch.inference_mode()
    def generate(self, request: LLMRequest) -> str:
        """Generate response"""
        # Format input
        prompt = self.format_messages(request.messages, request.context)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["data"]["max_length"]
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


# Initialize service
llm_service = LLMService()


@app.on_event("startup")
async def startup():
    """Load model on startup"""
    llm_service.load_model()


@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    """Generate text"""
    try:
        response = llm_service.generate(request)
        
        return LLMResponse(
            response=response,
            usage={
                "prompt_tokens": 0,  # TODO: calculate actual usage
                "completion_tokens": 0,
                "total_tokens": 0
            },
            model=llm_service.config["model"]["base_model_name"]
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_decision")
async def synthesize_decision(request: LLMRequest):
    """Synthesize bin decision"""
    return await generate(request)


@app.post("/generate_ideas")
async def generate_ideas(request: LLMRequest):
    """Generate upcycling ideas"""
    return await generate(request)


@app.post("/answer_question")
async def answer_question(request: LLMRequest):
    """Answer sustainability question"""
    return await generate(request)


@app.post("/rank_and_explain")
async def rank_and_explain(request: LLMRequest):
    """Rank and explain organizations"""
    return await generate(request)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "llm",
        "model_loaded": llm_service.model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)

