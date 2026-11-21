"""Qwen chat template. Hermes ``<tool_call>`` tags, with ``enable_thinking=False`` for Qwen3."""

from __future__ import annotations

import json
from typing import Any

from src.adapters.base import ParsedToolCall, parse_tool_payload
from src.spec.tools_schema import tool_names_from_schemas


class QwenAdapter:
    """Format and parse Qwen2.5 / Qwen3 tool calls.

    When a tokenizer is attached, ``format_messages`` calls
    ``apply_chat_template(..., enable_thinking=False)``. Without one, a ChatML
    fallback is used so CPU tests can round-trip a tool call.
    """

    def __init__(
        self,
        tokenizer: object | None = None,
        *,
        enable_thinking: bool = False,
        prefill_empty_think: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.prefill_empty_think = prefill_empty_think
        self.chat_template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }

    def format_messages(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        if self.tokenizer is not None:
            apply = getattr(self.tokenizer, "apply_chat_template", None)
            if apply is None:
                raise TypeError("tokenizer has no apply_chat_template")
            try:
                rendered = apply(messages, tools=tools, **self.chat_template_kwargs)
            except TypeError:
                rendered = apply(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return str(rendered)
        return self._fallback(messages, tools)

    def parse_tool_call(self, text: str) -> ParsedToolCall | None:
        return parse_tool_payload(text, prefer_tool_tag=True)

    def render_assistant(self, name: str, arguments: dict[str, Any]) -> str:
        body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))
        return f"<tool_call>{body}</tool_call>"

    def _fallback(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        user = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
        names = ", ".join(tool_names_from_schemas(tools))
        tool_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        thinking = ""
        if self.prefill_empty_think and not self.enable_thinking:
            thinking = "<think>\n\n</think>\n\n"
        elif self.enable_thinking:
            thinking = "<think>\n"
        return (
            "<|im_start|>system\n"
            f"{system}\n\n"
            "Bạn được gọi đúng một tool trong danh sách. "
            "Trả về một khối <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>.\n"
            f"Tools: {names}\n"
            f"{tool_json}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{thinking}"
        )
