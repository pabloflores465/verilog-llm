"""
SFTTrainer compatible con versiones nuevas de trl (sin max_seq_length).
Pega esto en tu Kaggle notebook.
"""

import inspect
from trl import SFTTrainer

# Detectar qué argumentos acepta esta versión
sig = inspect.signature(SFTTrainer.__init__)
valid_args = set(sig.parameters.keys())

# Argumentos que queremos pasar
kwargs = {
    "model": model,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "args": training_args,
    "data_collator": data_collator,
    "callbacks": [HFHubCheckpointCallback(ckpt_mgr, SAVE_STEPS)],
}

# Añadir solo si el argumento existe en esta versión
if "processing_class" in valid_args:
    kwargs["processing_class"] = tokenizer
elif "tokenizer" in valid_args:
    kwargs["tokenizer"] = tokenizer

if "max_seq_length" in valid_args:
    kwargs["max_seq_length"] = MAX_SEQ_LENGTH

if "dataset_text_field" in valid_args:
    kwargs["dataset_text_field"] = "text"

print("SFTTrainer args:", list(kwargs.keys()))
trainer = SFTTrainer(**kwargs)
