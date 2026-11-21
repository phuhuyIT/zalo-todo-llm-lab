"""Model-family adapters. Tokenizers stay optional so CPU tests can round-trip prompts."""

from src.adapters.base import ModelSpec, load_model_registry
from src.adapters.gemma_adapter import GemmaAdapter
from src.adapters.qwen_adapter import QwenAdapter

__all__ = ["GemmaAdapter", "ModelSpec", "QwenAdapter", "adapter_for", "load_model_registry"]


def adapter_for(spec: ModelSpec, tokenizer: object | None = None) -> QwenAdapter | GemmaAdapter:
    if spec.adapter == "qwen":
        return QwenAdapter(
            tokenizer=tokenizer,
            enable_thinking=spec.enable_thinking,
            prefill_empty_think=spec.prefill_empty_think,
        )
    if spec.adapter == "gemma":
        return GemmaAdapter(tokenizer=tokenizer)
    raise ValueError(f"Unknown adapter family {spec.adapter}")
