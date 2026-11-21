"""Merge a LoRA adapter and write Q4_K_M, IQ4_XS, an Ollama Modelfile, and a deploy note."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters import load_model_registry

QUANT_METHODS: tuple[tuple[str, str], ...] = (
    ("q4_k_m", "Q4_K_M"),
    ("iq4_xs", "IQ4_XS"),
)


def render_modelfile(gguf_filename: str = "model-Q4_K_M.gguf") -> str:
    return (
        f"FROM ./{gguf_filename}\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER temperature 0\n"
        "PARAMETER top_p 1\n"
    )


def render_local_deploy_readme(run_name: str, base_repo: str = "") -> str:
    trained = f"The adapter was trained from `{base_repo}`.\n\n" if base_repo else ""
    return f"""# Local deploy: {run_name}

{trained}The Node.js Phase 2 router calls this model with `LLM_PROVIDER=local`. The server still builds the system prompt, the nonce fences, and the scoped tool list. This Modelfile only sets the runtime window.

## Ollama

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama create zalo-todo-{run_name} -f Modelfile
```

`OLLAMA_KV_CACHE_TYPE=q8_0` stores the KV cache in 8-bit and cuts that part of VRAM roughly in half. `num_ctx` in the Modelfile is 4096, which covers the static rule prefix plus a forwarded letter.

`Q4_K_M` is the default the Modelfile loads. `IQ4_XS` is the smaller file for an 8 GB GPU: point `FROM` at `model-IQ4_XS.gguf` when you need that tier.

## llama.cpp

```bash
llama-server -m model-Q4_K_M.gguf -c 4096 --temp 0 --top-p 1
```

Point the bot's local provider at that server. Do not put an API key in the Modelfile or in this folder.
"""


def export_quantized(
    *,
    base_repo: str,
    lora_path: str,
    output_dir: str | Path,
    max_seq_length: int,
) -> list[Path]:
    """Load the saved adapter and emit both GGUF tiers plus the Modelfile.

    ``save_pretrained_gguf`` merges the adapter. Calling it on the Unsloth
    wrapper keeps the quantizer that a plain ``merge_and_unload`` would drop.
    ``base_repo`` is recorded in the deploy note. The adapter folder already
    points FastLanguageModel at that checkpoint.
    """
    from unsloth import FastLanguageModel

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    written: list[Path] = []
    for method, label in QUANT_METHODS:
        model.save_pretrained_gguf(str(destination), tokenizer, quantization_method=method)
        match = _find_gguf(destination, method)
        target = destination / f"model-{label}.gguf"
        if match is not None and match.resolve() != target.resolve():
            match.replace(target)
        written.append(target)
    (destination / "Modelfile").write_text(render_modelfile(), encoding="utf-8")
    (destination / "README_LOCAL_DEPLOY.md").write_text(
        render_local_deploy_readme(destination.name, base_repo),
        encoding="utf-8",
    )
    return written


def _find_gguf(directory: Path, method: str) -> Path | None:
    token = method.replace("_", "").lower()
    matches = [
        path
        for path in directory.rglob("*.gguf")
        if token in path.name.replace("_", "").replace("-", "").lower()
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Q4_K_M and IQ4_XS GGUF files plus an Ollama Modelfile.")
    parser.add_argument("--model", required=True, help="Registry key of the base checkpoint.")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--run-name", default="qlora")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to /content/drive/MyDrive/zalo-todo-models/<run-name> on Colab.",
    )
    parser.add_argument("--registry", default="configs/models.yaml")
    args = parser.parse_args(argv)
    _defaults, registry = load_model_registry(args.registry)
    if args.model not in registry:
        raise SystemExit(f"Unknown model {args.model}")
    spec = registry[args.model]
    output = Path(args.output_dir or f"/content/drive/MyDrive/zalo-todo-models/{args.run_name}")
    paths = export_quantized(
        base_repo=spec.repo_id,
        lora_path=args.lora_path,
        output_dir=output,
        max_seq_length=spec.max_seq_length,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
