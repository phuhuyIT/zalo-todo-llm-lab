"""Shared adapter interface and the model registry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    adapter: str
    load_in_4bit: bool
    approx_vram_gb: float
    max_seq_length: int
    enable_thinking: bool
    prefill_empty_think: bool
    notes: str


@dataclass(frozen=True)
class InferenceDefaults:
    max_seq_length: int = 3072
    temperature: float = 0.0
    max_new_tokens: int = 384


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


class ModelAdapter(Protocol):
    def format_messages(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> str: ...

    def parse_tool_call(self, text: str) -> ParsedToolCall | None: ...

    def render_assistant(self, name: str, arguments: dict[str, Any]) -> str: ...


def parse_tool_payload(text: str, *, prefer_tool_tag: bool) -> ParsedToolCall | None:
    """Pull one ``{"name", "arguments"}`` object out of a model completion."""
    cleaned = _THINK.sub("", text).strip()
    blobs: list[str] = []
    if prefer_tool_tag:
        blobs.extend(match.group(1) for match in _TOOL_CALL.finditer(cleaned))
        blobs.extend(match.group(1) for match in _FENCE.finditer(cleaned))
    else:
        blobs.extend(match.group(1) for match in _FENCE.finditer(cleaned))
        blobs.extend(match.group(1) for match in _TOOL_CALL.finditer(cleaned))
    raw_object = _first_json_object(cleaned)
    if raw_object is not None:
        blobs.append(raw_object)
    for blob in blobs:
        parsed = _coerce_call(blob)
        if parsed is not None:
            return parsed
    return None


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_call(blob: str) -> ParsedToolCall | None:
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool")
    arguments = payload.get("arguments", payload.get("parameters", payload.get("args")))
    if not isinstance(name, str) or not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return ParsedToolCall(name=name, arguments=arguments)


def load_model_registry(path: str | Path = "configs/models.yaml") -> tuple[InferenceDefaults, dict[str, ModelSpec]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    inference_raw = raw.get("inference") or {}
    defaults = InferenceDefaults(
        max_seq_length=int(inference_raw.get("max_seq_length", 3072)),
        temperature=float(inference_raw.get("temperature", 0.0)),
        max_new_tokens=int(inference_raw.get("max_new_tokens", 384)),
    )
    specs: dict[str, ModelSpec] = {}
    for key, item in (raw.get("models") or {}).items():
        specs[key] = ModelSpec(
            key=key,
            repo_id=str(item["repo_id"]),
            adapter=str(item["adapter"]),
            load_in_4bit=bool(item.get("load_in_4bit", True)),
            approx_vram_gb=float(item.get("approx_vram_gb", 0)),
            max_seq_length=int(item.get("max_seq_length", defaults.max_seq_length)),
            enable_thinking=bool(item.get("enable_thinking", False)),
            prefill_empty_think=bool(item.get("prefill_empty_think", False)),
            notes=str(item.get("notes") or ""),
        )
    return defaults, specs
