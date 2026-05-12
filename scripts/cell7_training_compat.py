# Cell 7: Training — Compatible con CUALQUIER versión de trl/transformers
# Fallback: usa Trainer estándar si SFTTrainer falla

import os
import inspect
from pathlib import Path
from transformers import (
    TrainingArguments, Trainer, TrainerCallback,
    DataCollatorForLanguageModeling
)


class HFHubCheckpointCallback(TrainerCallback):
    def __init__(self, ckpt_mgr, save_steps):
        self.ckpt_mgr = ckpt_mgr
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            print(f"\n📤 Uploading checkpoint-{state.global_step}...")
            self.ckpt_mgr.upload_checkpoint(str(checkpoint_dir), state.global_step)
        return control


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    weight_decay=0.001,
    optim="paged_adamw_8bit",
    group_by_length=True,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="steps" if eval_dataset else "no",
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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Intentar SFTTrainer primero ---
try:
    from trl import SFTTrainer
    sft_sig = inspect.signature(SFTTrainer.__init__)
    sft_args = set(sft_sig.parameters.keys())
    print("SFTTrainer args detectados:", sorted(sft_args))

    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "data_collator": data_collator,
        "callbacks": [HFHubCheckpointCallback(ckpt_mgr, SAVE_STEPS)],
    }

    # Cuidado: algunas versiones tienen **kwargs que pasan todo a Trainer
    # Solo pasar tokenizer/processing_class si es parámetro EXPLÍCITO (no **kwargs)
    explicit = {"processing_class", "tokenizer"} & sft_args
    if "processing_class" in explicit:
        kwargs["processing_class"] = tokenizer
        print("✓ Usando processing_class")
    elif "tokenizer" in explicit:
        kwargs["tokenizer"] = tokenizer
        print("✓ Usando tokenizer")
    else:
        print("ℹ️ SFTTrainer sin tokenizador explícito — omitiendo")

    if "max_seq_length" in sft_args and "max_seq_length" not in {"args", "kwargs"}:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    if "dataset_text_field" in sft_args and "dataset_text_field" not in {"args", "kwargs"}:
        kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**kwargs)
    print("✓ SFTTrainer creado")

except Exception as e:
    print(f"⚠️ SFTTrainer falló: {e}")
    print("→ Fallback a Trainer estándar")

    # Fallback: Trainer estándar con datos ya tokenizados
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[HFHubCheckpointCallback(ckpt_mgr, SAVE_STEPS)],
    )
    print("✓ Trainer estándar creado")

# --- Entrenamiento ---
resume_from = latest_ckpt if latest_ckpt else None
print(f"\nResume: {resume_from or 'scratch'}")
print(f"Auto-upload every {SAVE_STEPS} steps")
print("=" * 60)

try:
    trainer.train(resume_from_checkpoint=resume_from)
except KeyboardInterrupt:
    print("\n⚠️ Interrupted - saving...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted"))
    raise

# --- Final save ---
final_path = os.path.join(OUTPUT_DIR, "adapter_v1")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"\n✓ Saved locally: {final_path}")

try:
    api.upload_folder(
        folder_path=final_path,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        path_in_repo="adapter_v1",
        token=HF_TOKEN,
    )
    print(f"✓ Uploaded to {HUB_MODEL_ID}/adapter_v1")
except Exception as e:
    print(f"✗ Upload failed: {e}")
