"""
TrainingArguments fix for newer transformers versions.
Copy this into your Kaggle notebook cell.
"""

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    weight_decay=0.001,
    optim="paged_adamw_8bit",
    group_by_length=True,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="steps" if eval_dataset else "no",   # <-- eval_strategy (not evaluation_strategy)
    eval_steps=200,
    logging_steps=10,
    logging_first_step=True,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="checkpoint",
    hub_token=HF_TOKEN,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
)
