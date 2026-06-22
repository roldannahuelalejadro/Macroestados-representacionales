from __future__ import annotations

from transformers import AutoConfig


MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]


for model_id in MODELS:
    try:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        message = str(exc).splitlines()[0]
        print(f"{model_id}\tMISS\t{type(exc).__name__}\t{message[:220]}")
        continue

    print(
        f"{model_id}\tOK\tlayers={getattr(config, 'num_hidden_layers', None)}"
        f"\thidden={getattr(config, 'hidden_size', None)}"
    )
