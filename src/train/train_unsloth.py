"""T4-safe Unsloth QLoRA. Loss is masked to the assistant tool-call tokens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.adapters import adapter_for, load_model_registry
from src.adapters.base import ModelSpec


def load_train_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("train config must be a mapping")
    return raw


def response_markers(spec: ModelSpec) -> tuple[str, str]:
    """Instruction and response prefixes used by Unsloth ``train_on_responses_only``."""
    if spec.adapter == "gemma":
        return "<start_of_turn>user\n", "<start_of_turn>model\n"
    return "<|im_start|>user\n", "<|im_start|>assistant\n"


def rows_to_messages(rows: list[dict[str, Any]], spec: ModelSpec) -> list[dict[str, Any]]:
    adapter = adapter_for(spec)
    formatted: list[dict[str, Any]] = []
    for row in rows:
        assistant = adapter.render_assistant(str(row["tool_name"]), dict(row["arguments"]))
        formatted.append(
            {
                "messages": [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user"]},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
    return formatted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune a registry model on the gated tool-call set.")
    parser.add_argument("--config", default="configs/train_qlora.yaml")
    parser.add_argument("--model", default=None, help="Registry key. Overrides the config model.")
    parser.add_argument("--registry", default="configs/models.yaml")
    args = parser.parse_args(argv)
    config = load_train_config(args.config)
    model_key = str(args.model or config["model"])
    _defaults, registry = load_model_registry(args.registry)
    if model_key not in registry:
        raise SystemExit(f"Unknown model {model_key}")
    spec = registry[model_key]
    _train(config, spec)
    return 0


def _train(config: dict[str, Any], spec: ModelSpec) -> None:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    max_seq_length = int(config["max_seq_length"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.repo_id,
        max_seq_length=max_seq_length,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(config.get("seed", 42)),
    )
    dataset = load_dataset("json", data_files=str(config["train_path"]), split="train")
    rows = rows_to_messages([dict(row) for row in dataset], spec)

    def _render(batch: dict[str, Any]) -> dict[str, str]:
        messages = batch["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    from datasets import Dataset

    text_dataset = Dataset.from_list(rows).map(_render)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=text_dataset,
        args=SFTConfig(
            output_dir=str(config.get("output_dir", "outputs/qlora")),
            per_device_train_batch_size=int(config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            num_train_epochs=int(config["num_train_epochs"]),
            learning_rate=float(config["learning_rate"]),
            lr_scheduler_type=str(config["lr_scheduler_type"]),
            warmup_ratio=float(config["warmup_ratio"]),
            optim=str(config["optim"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
            logging_steps=int(config.get("logging_steps", 10)),
            save_steps=int(config.get("save_steps", 100)),
            seed=int(config.get("seed", 42)),
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            report_to="none",
        ),
    )
    if config.get("train_on_responses_only", True):
        instruction_part, response_part = response_markers(spec)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    trainer.train()
    output = Path(str(config.get("output_dir", "outputs/qlora")))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "lora")
    tokenizer.save_pretrained(output / "lora")


if __name__ == "__main__":
    raise SystemExit(main())
