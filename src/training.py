"""
Training setup for Verilog fine-tuning.
"""

from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from peft import PeftModel
import torch


def get_training_args(config: dict) -> TrainingArguments:
    """Create TrainingArguments from config dict."""
    
    training_config = config.get("training", {})
    
    return TrainingArguments(
        output_dir=training_config.get("output_dir", "/kaggle/working/checkpoints"),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 2e-4),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=training_config.get("max_grad_norm", 0.3),
        weight_decay=training_config.get("weight_decay", 0.001),
        optim=training_config.get("optim", "paged_adamw_8bit"),
        group_by_length=training_config.get("group_by_length", True),
        
        # Saving
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=training_config.get("save_total_limit", 3),
        
        # Evaluation
        evaluation_strategy=training_config.get("evaluation_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 200),
        load_best_model_at_end=training_config.get("load_best_model_at_end", False),
        
        # Logging
        logging_steps=training_config.get("logging_steps", 10),
        logging_first_step=True,
        
        # Hub
        push_to_hub=training_config.get("push_to_hub", True),
        hub_model_id=training_config.get("hub_model_id", ""),
        hub_strategy=training_config.get("hub_strategy", "checkpoint"),
        
        # Mixed precision
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        
        # Gradient checkpointing for memory
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Report to
        report_to=training_config.get("report_to", "none"),
        
        # Seed for reproducibility
        seed=42,
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    training_args=None,
    config=None,
):
    """Create SFTTrainer for Verilog fine-tuning."""
    
    if training_args is None:
        training_args = get_training_args(config or {})
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        max_seq_length=config.get("training", {}).get("max_seq_length", 2048) if config else 2048,
        dataset_text_field="text" if hasattr(train_dataset, "column_names") and "text" in train_dataset.column_names else None,
    )
    
    return trainer


def merge_and_save(model, tokenizer, output_path: str):
    """Merge LoRA weights with base model and save."""
    if isinstance(model, PeftModel):
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")
    return model
