"""Gemma 2 / Gemma 3 JSON tool wrapper. These templates have no Hermes tool tags."""

from __future__ import annotations

import json
from typing import Any

from src.adapters.base import ParsedToolCall, parse_tool_payload
from src.spec.tools_schema import tool_names_from_schemas


_INSTRUCTION = (
    "Chỉ trả về một đối tượng JSON, không markdown, không lời dẫn. "
    'Đúng dạng {"name": "<tool_name>", "arguments": {...}}. '
    "name phải là một tool trong danh sách được phép."
)


class GemmaAdapter:
    def __init__(self, tokenizer: object | None = None) -> None:
        self.tokenizer = tokenizer

    def format_messages(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        spec = self._tool_block(tools)
        prepared: list[dict[str, str]] = []
        injected = False
        for message in messages:
            if message["role"] == "system" and not injected:
                prepared.append({"role": "system", "content": f"{message['content']}\n\n{spec}"})
                injected = True
            else:
                prepared.append(dict(message))
        if not injected:
            prepared.insert(0, {"role": "system", "content": spec})
        if self.tokenizer is not None:
            apply = getattr(self.tokenizer, "apply_chat_template", None)
            if apply is None:
                raise TypeError("tokenizer has no apply_chat_template")
            rendered = apply(prepared, tokenize=False, add_generation_prompt=True)
            return str(rendered)
        return self._fallback(prepared)

    def parse_tool_call(self, text: str) -> ParsedToolCall | None:
        return parse_tool_payload(text, prefer_tool_tag=False)

    def render_assistant(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))

    def _tool_block(self, tools: list[dict[str, Any]]) -> str:
        names = ", ".join(tool_names_from_schemas(tools))
        listing = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        return f"{_INSTRUCTION}\nTool được phép: {names}\n{listing}"

    def _fallback(self, messages: list[dict[str, str]]) -> str:
        chunks: list[str] = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            chunks.append(f"<start_of_turn>{role}\n{message['content']}<end_of_turn>")
        chunks.append("<start_of_turn>model\n")
        return "\n".join(chunks)
