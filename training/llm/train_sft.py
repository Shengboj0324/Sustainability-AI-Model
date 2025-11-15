"""
LLM Supervised Fine-Tuning Script
Train domain-specialized model for sustainability
"""

import os
import sys
import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/llm_sft.yaml"):
    """Load training configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config):
    """Load base model and tokenizer"""
    model_name = config["model"]["base_model_name"]
    
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"]["trust_remote_code"]
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model: {model_name}")
    
    # Quantization config
    if config["model"]["quantization"]["enabled"]:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["model"]["quantization"]["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=config["model"]["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["model"]["quantization"]["bnb_4bit_use_double_quant"]
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=config["model"]["trust_remote_code"]
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
            device_map="auto",
            trust_remote_code=config["model"]["trust_remote_code"]
        )
    
    return model, tokenizer


def setup_lora(model, config):
    """Setup LoRA for efficient fine-tuning"""
    if not config["model"]["lora"]["enabled"]:
        return model
    
    logger.info("Setting up LoRA")
    
    # Prepare model for k-bit training if quantized
    if config["model"]["quantization"]["enabled"]:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config["model"]["lora"]["r"],
        lora_alpha=config["model"]["lora"]["alpha"],
        target_modules=config["model"]["lora"]["target_modules"],
        lora_dropout=config["model"]["lora"]["dropout"],
        bias=config["model"]["lora"]["bias"],
        task_type=config["model"]["lora"]["task_type"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def load_and_prepare_data(config, tokenizer):
    """Load and prepare training data"""
    logger.info("Loading datasets")
    
    # Load datasets
    train_files = config["data"]["train_files"]
    val_files = config["data"]["val_files"]
    
    train_dataset = load_dataset("json", data_files=train_files, split="train")
    val_dataset = load_dataset("json", data_files=val_files, split="train")
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Tokenization function
    def tokenize_function(examples):
        """Tokenize chat messages"""
        # Apply chat template
        texts = []
        for messages in examples["messages"]:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # Fallback formatting
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config["data"]["max_length"],
            padding="max_length" if not config["data"]["packing"] else False,
            return_tensors=None
        )
        
        # Set labels (for causal LM, labels = input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Tokenize datasets
    logger.info("Tokenizing datasets")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config["data"]["num_workers"],
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config["data"]["num_workers"],
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset


def get_training_arguments(config):
    """Get training arguments"""
    return TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        save_strategy=config["training"]["save_strategy"],
        seed=config["training"]["seed"],
        data_seed=config["training"]["data_seed"],
        optim=config["training"]["optim"],
        report_to=["wandb"],  # Enable W&B logging
        run_name="releaf-llm-sft"
    )


def main():
    """Main training function"""
    # Load config
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Load and prepare data
    train_dataset, val_dataset = load_and_prepare_data(config, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = get_training_arguments(config)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model()
    tokenizer.save_pretrained(config["training"]["output_dir"])
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

