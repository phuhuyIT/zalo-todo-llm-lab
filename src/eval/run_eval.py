"""Zero-shot and LoRA benchmark runner.

Loads one 4-bit model at a time, scores ``data/golden_eval.jsonl``, then frees
VRAM before the next model. CPU machines can still import the scoring helpers;
generation itself requires a CUDA GPU and Unsloth.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from src.adapters import adapter_for, load_model_registry
from src.adapters.base import ModelSpec
from src.eval.report import percentile, render_report, score_case
from src.spec.prompt_builder import prompt_from_case
from src.spec.tools_schema import get_scoped_tools
from src.spec.validator import validate_tool_call


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            cases.append(json.loads(stripped))
    return cases


def _trace(
    case: dict[str, Any],
    raw_output: str,
    tool_name: str | None,
    arguments: dict[str, Any] | None,
    scored: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "user_text": case.get("user_text"),
        "expected_tool": case.get("expected_tool"),
        "expected_args": case.get("expected_args") or {},
        "raw_output": raw_output,
        "parsed_tool": tool_name,
        "parsed_args": arguments or {},
        **scored,
    }


def evaluate_prediction(case: dict[str, Any], tool_name: str | None, arguments: dict[str, Any] | None) -> dict[str, Any]:
    if tool_name is None or arguments is None:
        scored = score_case(case, None, None, valid=False)
        return {**scored, "reason": "unparsed"}
    result = validate_tool_call(
        tool_name,
        arguments,
        combined_text=str(case.get("user_text") or ""),
        snapshot=case.get("snapshot") or {},
    )
    scored = score_case(case, tool_name, result.arguments if result.ok else arguments, valid=result.ok)
    reason = "ok" if scored["tool_ok"] and scored["args_ok"] else _reason(scored, result.issues)
    return {**scored, "reason": reason}


def _reason(scored: dict[str, Any], issues: list[Any]) -> str:
    if not scored["valid"]:
        detail = "; ".join(issue.message for issue in issues) or "schema"
        return f"invalid: {detail}"
    if not scored["tool_ok"]:
        return "wrong tool"
    if not scored["args_ok"]:
        return "argument mismatch"
    return "ok"


def summarize(model: str, details: list[dict[str, Any]], latencies: list[float], peak_vram_gb: float | None) -> dict[str, Any]:
    count = max(1, len(details))
    injection = [item["injection_defended"] for item in details if item["injection_defended"] is not None]
    failures = [
        {"id": item["id"], "reason": item["reason"]}
        for item in details
        if item["reason"] != "ok"
    ]
    return {
        "model": model,
        "json_schema_validity": 100.0 * sum(1 for item in details if item["valid"]) / count,
        "tool_selection_accuracy": 100.0 * sum(1 for item in details if item["tool_ok"]) / count,
        "critical_argument_exact_match": 100.0 * sum(1 for item in details if item["args_ok"]) / count,
        "prompt_injection_defense": None if not injection else 100.0 * sum(1 for item in injection if item) / len(injection),
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "peak_vram_gb": peak_vram_gb,
        "failures": failures,
        "traces": list(details),
    }


# Training stays at 3072 so a 14B QLoRA run fits a T4. Eval needs room for the
# fenced calendar prompt plus the completion, which already exceeds 3072 tokens.
EVAL_CONTEXT = 4096


def context_for_eval(configured: int) -> int:
    return max(configured, EVAL_CONTEXT)


def max_prompt_tokens(context: int, max_new_tokens: int) -> int:
    return max(1, context - max_new_tokens)


def flatten_token_ids(value: Any) -> list[int]:
    """Turn a tokenizer id field into one 1D list.

    Gemma 3 returns an already-batched ``[[...]]`` tensor. Wrapping that again
    makes a 3D attention mask, which ``generate`` rejects.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("tokenizer input_ids are not a list")
    return [int(item) for item in value]


def generation_batch(input_ids: list[int]) -> dict[str, list[int]]:
    """Prompt tokens plus a 1D mask. The caller adds the batch dimension."""
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def clip_token_rows(rows: dict[str, list[int]], max_input: int) -> tuple[dict[str, list[int]], bool]:
    """Keep the tail of an over-long prompt so the user turn stays inside the context."""
    length = len(rows.get("input_ids") or [])
    if length <= max_input:
        return rows, False
    clipped = {key: values[-max_input:] for key, values in rows.items()}
    if "attention_mask" in clipped:
        clipped["attention_mask"] = [1] * len(clipped["input_ids"])
    return clipped, True


def build_eval_prompt(case: dict[str, Any], spec: ModelSpec) -> str:
    built = prompt_from_case(case)
    tools = get_scoped_tools(str(case.get("user_text") or ""))
    adapter = adapter_for(spec)
    messages = [
        {"role": "system", "content": built.system},
        {"role": "user", "content": built.user},
    ]
    return adapter.format_messages(messages, tools)


def run_models(
    model_keys: list[str],
    cases: list[dict[str, Any]],
    *,
    registry_path: str | Path = "configs/models.yaml",
    lora_path: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    defaults, registry = load_model_registry(registry_path)
    rows: list[dict[str, Any]] = []
    for key in model_keys:
        if key not in registry:
            raise SystemExit(f"Unknown model {key}. Known: {', '.join(registry)}")
        spec = registry[key]
        label = key if not lora_path else f"{key}+lora"
        try:
            details, latencies, peak = _generate_one(spec, cases, defaults.max_new_tokens, defaults.temperature, lora_path)
            rows.append(summarize(label, details, latencies, peak))
        except Exception as exc:
            print(f"{label}: model run failed: {exc}")
            rows.append(summarize(label, [_failed_case(str(exc))], [], None))
        if output_path is not None:
            _write_report(rows, output_path)
    return rows


def _failed_case(reason: str) -> dict[str, Any]:
    return {
        "id": "model",
        "valid": False,
        "tool_ok": False,
        "args_ok": False,
        "injection_defended": None,
        "reason": reason,
    }


def trace_path_for(output_path: str | Path) -> Path:
    destination = Path(output_path)
    return destination.with_name(destination.stem + "_traces.jsonl")


def _write_report(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(rows), encoding="utf-8")
    traces: list[dict[str, Any]] = []
    for row in rows:
        for trace in row.get("traces") or []:
            traces.append({"model": row["model"], **trace})
    payload = "\n".join(json.dumps(trace, ensure_ascii=False) for trace in traces)
    trace_path_for(destination).write_text(payload + ("\n" if traces else ""), encoding="utf-8")


def _generate_one(
    spec: ModelSpec,
    cases: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    lora_path: str | None,
) -> tuple[list[dict[str, Any]], list[float], float | None]:
    import torch
    from unsloth import FastLanguageModel

    context = context_for_eval(spec.max_seq_length)
    prompt_budget = max_prompt_tokens(context, max_new_tokens)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.repo_id,
        max_seq_length=context,
        load_in_4bit=spec.load_in_4bit,
        dtype=None,
    )
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)
    if getattr(model, "generation_config", None) is not None:
        # Leave max_length unset so transformers does not warn on every sample.
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = max_new_tokens
    adapter = adapter_for(spec, tokenizer)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    try:
        for case in cases:
            tools = get_scoped_tools(str(case.get("user_text") or ""))
            built = prompt_from_case(case)
            messages = [
                {"role": "system", "content": built.system},
                {"role": "user", "content": built.user},
            ]
            try:
                prompt = adapter.format_messages(messages, tools)
                encoded = tokenizer(prompt, add_special_tokens=False)
                token_ids = flatten_token_ids(encoded["input_ids"])
                rows, clipped = clip_token_rows({"input_ids": token_ids}, prompt_budget)
                if clipped:
                    print(f"{case.get('id')}: prompt exceeded {prompt_budget} tokens and was clipped from the left")
                batch = generation_batch(rows["input_ids"])
                inputs = {
                    "input_ids": torch.tensor([batch["input_ids"]], dtype=torch.long),
                    "attention_mask": torch.tensor([batch["attention_mask"]], dtype=torch.long),
                }
                if inputs["attention_mask"].ndim != 2:
                    raise RuntimeError("attention_mask must be 2D")
                if torch.cuda.is_available():
                    inputs = {key: value.to(model.device) for key, value in inputs.items()}
                started = time.perf_counter()
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "use_cache": True,
                }
                if temperature > 0:
                    generate_kwargs["temperature"] = temperature
                with torch.inference_mode():
                    output = model.generate(**inputs, **generate_kwargs)
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)
                parsed = adapter.parse_tool_call(generated)
                scored = evaluate_prediction(
                    case,
                    None if parsed is None else parsed.name,
                    None if parsed is None else parsed.arguments,
                )
                details.append(_trace(case, generated, None if parsed is None else parsed.name, None if parsed is None else parsed.arguments, scored))
            except Exception as exc:
                latencies.append(0.0)
                defended = False if case.get("category") == "prompt_injection" else None
                scored = {
                    "valid": False,
                    "tool_ok": False,
                    "args_ok": False,
                    "injection_defended": defended,
                    "reason": f"generate failed: {exc}",
                }
                details.append(_trace(case, "", None, None, scored))
                print(f"{case.get('id')}: generate failed: {exc}")
    finally:
        peak = None
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return details, latencies, peak


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen and Gemma on the golden Vietnamese set.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--golden", default="data/golden_eval.jsonl")
    parser.add_argument("--registry", default="configs/models.yaml")
    parser.add_argument("--output", default="results/benchmark_report.md")
    args = parser.parse_args(argv)
    cases = load_cases(args.golden)
    rows = run_models(
        args.models,
        cases,
        registry_path=args.registry,
        lora_path=args.lora_path,
        output_path=args.output,
    )
    print(Path(args.output).read_text(encoding="utf-8"))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
