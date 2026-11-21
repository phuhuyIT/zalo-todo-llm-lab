"""CPU parity tests for the Phase 2 prompt, scoping boundary, and tool-call validator."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.adapters import adapter_for, load_model_registry
from src.adapters.gemma_adapter import GemmaAdapter
from src.adapters.qwen_adapter import QwenAdapter
from src.dataset.synthesize import build_scenarios, passes_teacher_gate, write_splits
from src.eval.report import score_case
from src.eval.run_eval import clip_token_rows, context_for_eval, max_prompt_tokens
from src.export.export_gguf import render_local_deploy_readme, render_modelfile
from src.spec.prompt_builder import (
    CLARIFY_RULE,
    build_calendar,
    build_prompt,
    classify_tool,
    infer_assignee,
    merge_clarify,
    merge_confirm_payload,
    normalize_spoken_time,
    resolve_vi_date,
    select_context,
)
from src.spec.tools_schema import get_scoped_tools, scoped_tool_names
from src.spec.validator import validate_tool_call
from src.train.train_unsloth import load_train_config, response_markers

ROOT = Path(__file__).resolve().parents[1]
FRIDAY = "2026-11-13T16:10:00+07:00"
SUNDAY = "2026-11-15T09:00:00+07:00"
ALL_FIVE = ["draft_task", "draft_meeting", "clarify_ambiguity", "propose_update", "query_schedule"]
BASE_THREE = ["draft_task", "draft_meeting", "clarify_ambiguity"]
KHAM = "Nhắc anh 8h sáng thứ 7 đi khám ở BV Đại học Y Dược"
HO_SO = "Nhờ em chuẩn bị hồ sơ thầu trước chiều thứ 2 tuần tới, việc gấp"


def test_friday_calendar_matches_the_spec_vector() -> None:
    facts = build_calendar(FRIDAY)
    assert facts.this_week_start == "2026-11-09"
    assert facts.this_week_end == "2026-11-15"
    assert facts.next_week_start == "2026-11-16"
    assert facts.next_week_end == "2026-11-22"
    assert facts.next_monday == "2026-11-16"
    assert facts.this_saturday == "2026-11-14"
    assert facts.this_sunday == "2026-11-15"
    assert facts.today == "2026-11-13"
    assert facts.tomorrow == "2026-11-14"
    assert facts.day_after == "2026-11-15"
    assert facts.lines[0].startswith("- Thứ Hai (đã qua): 2026-11-09")
    assert any("Hôm nay" in line and "2026-11-13" in line for line in facts.lines)
    assert any("Ngày mai" in line and "2026-11-14" in line for line in facts.lines)
    assert any("Mốt / Ngày kia" in line and "2026-11-15" in line for line in facts.lines)
    assert facts.lines[-1].endswith("2026-11-26 (hiển thị: 26/11/2026)")


def test_sunday_weekend_is_yesterday_and_today() -> None:
    facts = build_calendar(SUNDAY)
    assert facts.this_saturday == "2026-11-14"
    assert facts.this_sunday == "2026-11-15"
    assert facts.next_monday == "2026-11-16"
    assert facts.next_week_start == "2026-11-16"
    assert facts.next_week_end == "2026-11-22"
    saturday = next(line for line in facts.lines if "2026-11-14" in line)
    sunday = next(line for line in facts.lines if "2026-11-15" in line and "Hôm nay" in line)
    assert "đã qua" in saturday
    assert "Hôm nay" in sunday
    assert "đã qua" not in sunday


def test_saturday_weekend_is_today_and_tomorrow() -> None:
    facts = build_calendar("2026-11-14T09:00:00+07:00")
    assert facts.this_saturday == "2026-11-14"
    assert facts.this_sunday == "2026-11-15"
    assert facts.today == "2026-11-14"
    assert facts.tomorrow == "2026-11-15"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("trước chiều", "11:30"),
        ("Nhờ em chuẩn bị hồ sơ trước chiều thứ 2", "11:30"),
        ("trước trưa", "11:30"),
        ("trưa", "11:30"),
        ("cuối giờ sáng", "11:30"),
        ("đầu giờ sáng", "08:00"),
        ("sáng", "08:30"),
        ("8h sáng", "08:00"),
        ("đầu giờ chiều", "13:30"),
        ("chiều", "15:00"),
        ("cuối giờ chiều", "17:00"),
        ("tan tầm", "17:00"),
        ("tối", "19:30"),
        ("2 rưỡi", "14:30"),
        ("lúc 2 rưỡi chiều", "14:30"),
        ("2h sáng", "02:00"),
        ("2h chiều", "14:00"),
        ("9h30 sáng", "09:30"),
        ("12h", "12:00"),
        ("12h30", "12:30"),
        ("14h", "14:00"),
        ("08:00", "08:00"),
    ],
)
def test_time_normalization(text: str, expected: str) -> None:
    assert normalize_spoken_time(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("thứ 7", "2026-11-14"),
        ("thứ 2 tuần này", "2026-11-09"),
        ("thứ 2 tuần tới", "2026-11-16"),
        ("thứ 2 tuần sau", "2026-11-16"),
        ("thứ 4 tuần sau", "2026-11-18"),
        ("thứ 3 tuần sau", "2026-11-17"),
        ("hôm nay", "2026-11-13"),
        ("ngày mai", "2026-11-14"),
        ("mốt", "2026-11-15"),
        ("đầu tuần sau", "2026-11-16"),
        ("cuối tuần này", None),
    ],
)
def test_friday_date_labels(text: str, expected: str | None) -> None:
    assert resolve_vi_date(text, FRIDAY) == expected


def test_sunday_date_labels() -> None:
    assert resolve_vi_date("thứ 2 tuần tới", SUNDAY) == "2026-11-16"
    assert resolve_vi_date("ngày mai", SUNDAY) == "2026-11-16"
    assert resolve_vi_date("thứ 7", SUNDAY) == "2026-11-14"
    assert resolve_vi_date("chủ nhật", SUNDAY) == "2026-11-15"
    assert resolve_vi_date("cuối tuần này", SUNDAY) is None


def test_task_versus_meeting_and_assignee() -> None:
    assert classify_tool(KHAM, FRIDAY, {}) == "draft_meeting"
    assert classify_tool(HO_SO, FRIDAY, {}) == "draft_task"
    assert classify_tool("Họp với bên đối tác nhé", FRIDAY, {}) == "clarify_ambiguity"
    assert infer_assignee("leader", HO_SO) == "other"
    assert infer_assignee("leader", "nhắc anh gọi điện hôm nay") == "self"
    assert infer_assignee("leader", "Chuẩn bị báo cáo tuần") == "other"
    assert infer_assignee("leader", "tôi đi khám sáng mai") == "self"
    assert infer_assignee("assistant", "Em soạn công văn chiều nay") == "self"
    assert infer_assignee("assistant", "Lịch của anh sáng mai") == "other"
    assert infer_assignee("assistant", "Nhắc sếp họp 9h sáng mai") == "other"
    assert infer_assignee("assistant", "mình nộp báo cáo ngày mai") == "self"
    assert normalize_spoken_time(KHAM) == "08:00"
    assert resolve_vi_date(KHAM, FRIDAY) == "2026-11-14"
    assert normalize_spoken_time(HO_SO) == "11:30"
    assert resolve_vi_date(HO_SO, FRIDAY) == "2026-11-16"


def test_unknown_code_clarifies_and_known_code_updates() -> None:
    snapshot = {"T1": {"id": "task-1", "version": 1, "kind": "task"}}
    assert classify_tool("xong T99", FRIDAY, snapshot) == "clarify_ambiguity"
    assert classify_tool("Báo cáo thuế xong rồi", FRIDAY, snapshot) == "propose_update"


def test_location_keyword_adds_a_meeting_outside_the_first_five() -> None:
    meetings = [
        {"id": f"m{i}", "version": 1, "title": f"Mục {i}", "group": "work", "start_date": "2026-11-20", "start_time": "09:00", "location": "Phòng A"}
        for i in range(5)
    ]
    meetings.append(
        {
            "id": "hospital",
            "version": 4,
            "title": "Làm việc định kỳ",
            "group": "health",
            "start_date": "2026-11-20",
            "start_time": "09:00",
            "location": "BV Đại học Y Dược",
        }
    )
    lines, snapshot = select_context([], meetings, "đổi lịch họp ở BV Đại học Y Dược")
    assert any("BV Đại học Y Dược" in line for line in lines)
    assert snapshot["M6"]["id"] == "hospital"
    assert snapshot["M6"]["kind"] == "meeting"


def test_open_item_lines_follow_the_spec_example() -> None:
    lines, snapshot = select_context(
        [
            {
                "id": "t",
                "version": 2,
                "title": "Nộp báo cáo quý 4",
                "group": "work",
                "due_date": "2026-11-16",
                "due_time": "17:00",
                "assignee_name": "Lan",
                "importance": "high",
            }
        ],
        [
            {
                "id": "m",
                "version": 1,
                "title": "Khám định kỳ",
                "group": "health",
                "start_date": "2026-11-14",
                "start_time": "08:00",
                "location": "BV Đại học Y Dược",
            }
        ],
        "báo cáo",
    )
    assert lines[0] == "[T1] Việc | Công việc | Nộp báo cáo quý 4 | Hạn 17:00 16/11/2026 | Phụ trách: Lan | Cao"
    assert lines[1] == "[M1] Họp | Sức khỏe | Khám định kỳ | 08:00 14/11/2026 | BV Đại học Y Dược"
    assert snapshot["T1"] == {"id": "t", "version": 2, "kind": "task"}


def test_date_only_task_renders_ca_ngay() -> None:
    lines, _snapshot = select_context(
        [{"id": "t", "version": 1, "title": "Gọi điện", "group": "work", "due_date": "2026-11-13", "importance": "normal"}],
        [],
        "gọi",
    )
    assert "13/11/2026, cả ngày" in lines[0]


def test_prompt_is_static_first_and_strips_the_nonce() -> None:
    nonce = "FENCE_TESTNONCE"
    built = build_prompt(
        actor="leader",
        now=FRIDAY,
        user_text=f"Nhờ em {nonce} chuẩn bị hồ sơ",
        confirm={"kind": "task", "payload": {"title": f"x{nonce}"}},
        clarify={"source_text": "cũ", "partial": None, "tool_name": "clarify_ambiguity"},
        nonce=nonce,
    )
    system = built.system
    assert system.startswith("Bạn là Bộ Điều Phối Lịch & Công Việc Nội Bộ")
    assert CLARIFY_RULE in system
    assert system.index(CLARIFY_RULE) < system.index("Người đang nhắn:")
    assert system.index("Người đang nhắn:") < system.index(f'<data id="{nonce}">')
    payload = system.split(f'<data id="{nonce}">', 1)[1].split("</data>", 1)[0]
    assert nonce not in payload
    assert "2026-11-16" in payload
    assert "partial" in payload
    body = built.user.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert nonce not in body
    assert built.user.startswith(f'<user_message id="{nonce}">')
    assert "Lãnh đạo: tôi, anh, mình, nhắc anh là self" in system
    assert "Trợ lý: em, mình, tôi là self" in system


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Nhờ em gọi điện chiều nay", ALL_FIVE),
        ("a" * 220, ALL_FIVE),
        ("a" * 221, BASE_THREE),
        ("á" * 221, BASE_THREE),
        ("một\nhai\nba", ALL_FIVE),
        ("một\nhai\nba\nbốn\nnăm", BASE_THREE),
        ("Kính gửi ban giám đốc", BASE_THREE),
        ("CÔNG VĂN số 12", BASE_THREE),
        ("Giấy mời họp", BASE_THREE),
        ("Thông báo nghỉ", BASE_THREE),
        ("Chuyển tiếp tin này", BASE_THREE),
        ("V/v: mời họp", BASE_THREE),
    ],
)
def test_dynamic_tool_scoping(text: str, expected: list[str]) -> None:
    assert scoped_tool_names(text) == expected
    assert [tool["function"]["name"] for tool in get_scoped_tools(text)] == expected


def test_validator_rejects_zulu_and_other_offsets() -> None:
    bad = validate_tool_call(
        "draft_meeting",
        {
            "title": "Họp",
            "start_date": "2026-11-14",
            "start_time": "08:00",
            "start_datetime": "2026-11-14T08:00:00Z",
        },
        combined_text="họp",
        snapshot={},
    )
    assert not bad.ok
    other = validate_tool_call(
        "draft_meeting",
        {"title": "Họp", "start_datetime": "2026-11-14T08:00:00+00:00"},
        combined_text="họp",
        snapshot={},
    )
    assert not other.ok
    good = validate_tool_call(
        "draft_meeting",
        {
            "title": "Khám định kỳ",
            "group": "health",
            "start_date": "2026-11-14",
            "start_time": "08:00",
            "duration_minutes": 60,
            "start_datetime": "2026-11-14T08:00:00+07:00",
            "location": "BV Đại học Y Dược",
        },
        combined_text=KHAM,
        snapshot={},
    )
    assert good.ok
    assert good.arguments["duration_minutes"] == 60


def test_validator_rejects_unknown_item_code_and_bad_fields() -> None:
    unknown = validate_tool_call(
        "propose_update",
        {"item_code": "T99", "action": "mark_done"},
        combined_text="xong T99",
        snapshot={"T1": {"id": "t", "version": 1, "kind": "task"}},
    )
    assert not unknown.ok
    assert any(issue.code == "unknown_item_code" for issue in unknown.issues)
    known = validate_tool_call(
        "propose_update",
        {"item_code": "T1", "action": "postpone_task"},
        combined_text="dời việc",
        snapshot={"T1": {"id": "t", "version": 1, "kind": "task"}},
    )
    assert known.ok
    assert not validate_tool_call(
        "draft_task",
        {"title": "Báo cáo", "due_date": "2026-02-31"},
        combined_text="báo cáo",
        snapshot={},
    ).ok
    assert not validate_tool_call(
        "draft_task",
        {"title": "Báo cáo", "due_time": "25:00"},
        combined_text="báo cáo",
        snapshot={},
    ).ok
    assert not validate_tool_call(
        "draft_task",
        {"title": "Báo cáo", "group": "personal"},
        combined_text="báo cáo",
        snapshot={},
    ).ok
    assert not validate_tool_call(
        "draft_meeting",
        {"title": "Họp", "duration_minutes": 45},
        combined_text="họp",
        snapshot={},
    ).ok
    blocked = validate_tool_call(
        "propose_update",
        {"item_code": "T1", "action": "mark_done"},
        combined_text="Kính gửi. " + ("nội dung. " * 40),
        snapshot={"T1": {"id": "t", "version": 1, "kind": "task"}},
    )
    assert not blocked.ok
    assert blocked.issues[0].code == "tool_not_scoped"
    long_question = validate_tool_call(
        "clarify_ambiguity",
        {"question_text": "Ngày nào? Mấy giờ? Ở đâu?"},
        combined_text="họp",
        snapshot={},
    )
    assert not long_question.ok


def test_clarify_merge_keeps_partial_and_null_partial_can_switch_tools() -> None:
    tool, args = merge_clarify(
        {
            "type": "clarify",
            "tool_name": "propose_update",
            "source_text": "Dời việc giao ban",
            "partial": {
                "item_code": "T1",
                "action": "postpone_task",
                "item": {"id": "task-1", "version": 3, "kind": "task"},
            },
        },
        "draft_task",
        {"new_date": "2026-11-18", "new_time": "13:30", "action": "mark_done"},
    )
    assert tool == "propose_update"
    assert args["item_code"] == "T1"
    assert args["action"] == "postpone_task"
    assert args["new_date"] == "2026-11-18"
    assert "item" not in args
    switched, switched_args = merge_clarify(
        {
            "tool_name": "clarify_ambiguity",
            "source_text": "Họp với bên đối tác nhé",
            "partial": None,
        },
        "draft_meeting",
        {"title": "Họp đối tác", "start_date": "2026-11-17", "start_time": "09:00"},
    )
    assert switched == "draft_meeting"
    assert switched_args["start_time"] == "09:00"


def test_confirm_merge_keeps_title_and_replace_drops_it() -> None:
    current = {
        "kind": "meeting",
        "edit_count": 0,
        "payload": {
            "title": "Họp giao ban",
            "group": "work",
            "start_date": "2026-11-15",
            "start_time": "14:30",
            "duration_minutes": 60,
            "location": "Phòng A2",
            "note": "CV 142",
        },
    }
    merged = merge_confirm_payload(
        current,
        "draft_meeting",
        {"start_date": "2026-11-17", "start_time": "09:00", "location": "Phòng B1", "intent": "merge"},
    )
    assert merged["title"] == "Họp giao ban"
    assert merged["note"] == "CV 142"
    assert merged["start_date"] == "2026-11-17"
    assert merged["location"] == "Phòng B1"
    replaced = merge_confirm_payload(
        current,
        "draft_task",
        {"title": "Nộp báo cáo", "due_date": "2026-11-19", "intent": "merge"},
    )
    assert replaced["title"] == "Nộp báo cáo"
    assert "note" not in replaced
    cleared = merge_confirm_payload(current, "draft_meeting", {"note": "", "intent": "merge"})
    assert cleared["note"] is None
    assert cleared["title"] == "Họp giao ban"


def test_qwen_and_gemma_round_trip_tool_calls() -> None:
    qwen = QwenAdapter()
    assert qwen.enable_thinking is False
    assert qwen.chat_template_kwargs["enable_thinking"] is False
    tools = get_scoped_tools("Nhờ em soạn báo cáo")
    rendered = qwen.format_messages(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        tools,
    )
    assert "<|im_start|>system" in rendered
    assert "draft_task" in rendered
    assert "<think>" not in rendered
    qwen3 = QwenAdapter(enable_thinking=False, prefill_empty_think=True)
    prefilled = qwen3.format_messages(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        BASE_SCHEMAS(),
    )
    assert prefilled.endswith("<think>\n\n</think>\n\n")
    payload = {"title": "Chuẩn bị hồ sơ thầu", "due_date": "2026-11-16", "due_time": "11:30", "assignee": "other"}
    parsed = qwen.parse_tool_call("<think>nháp</think>\n" + qwen.render_assistant("draft_task", payload))
    assert parsed is not None
    assert parsed.name == "draft_task"
    assert parsed.arguments["due_time"] == "11:30"
    raw = qwen.parse_tool_call('{"name": "query_schedule", "arguments": {"target": "meetings"}}')
    assert raw is not None and raw.name == "query_schedule"
    checked = validate_tool_call(parsed.name, parsed.arguments, combined_text=HO_SO, snapshot={})
    assert checked.ok

    gemma = GemmaAdapter()
    letter = "Kính gửi\n" + ("dòng\n" * 5)
    narrow = get_scoped_tools(letter)
    gemma_prompt = gemma.format_messages(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        narrow,
    )
    assert '{"name": "<tool_name>", "arguments":' in gemma_prompt
    assert "propose_update" not in gemma_prompt
    assert "clarify_ambiguity" in gemma_prompt
    fenced = """```json
{"name": "clarify_ambiguity", "arguments": {"question_text": "Anh muốn họp ngày nào, mấy giờ ạ?"}}
```"""
    gemma_parsed = gemma.parse_tool_call(fenced)
    assert gemma_parsed is not None
    assert gemma_parsed.name == "clarify_ambiguity"
    assert validate_tool_call(gemma_parsed.name, gemma_parsed.arguments, combined_text="họp", snapshot={}).ok


def test_gemma2_template_gets_rules_inside_the_user_turn() -> None:
    class Gemma2Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del tokenize
            if any(message["role"] == "system" for message in messages):
                raise RuntimeError("System role not supported")
            text = "\n".join(message["content"] for message in messages)
            return text + ("\n<model>" if add_generation_prompt else "")

    rendered = GemmaAdapter(tokenizer=Gemma2Tokenizer()).format_messages(
        [{"role": "system", "content": "RULES"}, {"role": "user", "content": "hello"}],
        get_scoped_tools("xin chào"),
    )
    assert "RULES" in rendered
    assert "hello" in rendered
    assert rendered.endswith("\n<model>")
    assert "<start_of_turn>system" not in GemmaAdapter().format_messages(
        [{"role": "system", "content": "RULES"}, {"role": "user", "content": "hello"}],
        get_scoped_tools("xin chào"),
    )


def BASE_SCHEMAS() -> list[dict]:
    return get_scoped_tools("Kính gửi công văn dài " + ("x" * 240))


def test_registry_covers_qwen_and_gemma_on_a_t4() -> None:
    defaults, registry = load_model_registry(ROOT / "configs/models.yaml")
    assert defaults.max_seq_length == 3072
    assert defaults.temperature == 0.0
    assert defaults.max_new_tokens == 384
    assert set(registry) == {"qwen2.5-7b", "qwen2.5-14b", "qwen3-8b", "gemma2-9b", "gemma3-12b"}
    qwen3 = registry["qwen3-8b"]
    assert qwen3.repo_id == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert qwen3.enable_thinking is False
    adapter = adapter_for(qwen3)
    assert isinstance(adapter, QwenAdapter)
    assert adapter.enable_thinking is False
    assert adapter.prefill_empty_think is True
    gemma3 = registry["gemma3-12b"]
    assert gemma3.repo_id == "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
    assert isinstance(adapter_for(gemma3), GemmaAdapter)
    assert registry["qwen2.5-14b"].max_seq_length == 3072
    assert all(spec.load_in_4bit for spec in registry.values())


def test_train_config_is_t4_safe_and_response_masked() -> None:
    config = load_train_config(ROOT / "configs/train_qlora.yaml")
    assert config["max_seq_length"] == 3072
    assert config["load_in_4bit"] is True
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["lora_dropout"] == 0
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 8
    assert config["train_on_responses_only"] is True
    assert config["optim"] == "adamw_8bit"
    _defaults, registry = load_model_registry(ROOT / "configs/models.yaml")
    assert response_markers(registry["qwen2.5-7b"]) == ("<|im_start|>user\n", "<|im_start|>assistant\n")
    assert response_markers(registry["gemma2-9b"])[1] == "<start_of_turn>model\n"


def test_modelfile_sets_the_local_context_window() -> None:
    text = render_modelfile()
    assert "FROM ./model-Q4_K_M.gguf" in text
    assert "PARAMETER num_ctx 4096" in text
    assert "PARAMETER temperature 0" in text
    assert "PARAMETER top_p 1" in text
    readme = render_local_deploy_readme("run")
    assert "OLLAMA_FLASH_ATTENTION=1" in readme
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in readme
    assert "LLM_PROVIDER=local" in readme
    assert "model-IQ4_XS.gguf" in readme


def test_synthetic_pool_is_balanced_and_gated() -> None:
    scenarios = build_scenarios(500, seed=42)
    counts: dict[str, int] = {}
    for scenario in scenarios:
        counts[scenario["category"]] = counts.get(scenario["category"], 0) + 1
        assert scenario["user_text"]
    assert any("\n---\n" in scenario["user_text"] for scenario in scenarios)
    assert any("BTL TP" in scenario["user_text"] or "UBND" in scenario["user_text"] for scenario in scenarios)
    assert counts == {
        "draft_task": 150,
        "draft_meeting": 125,
        "propose_update": 100,
        "query_schedule": 50,
        "clarify_ambiguity": 50,
        "prompt_injection": 25,
    }
    clocks = {scenario["clock_iso"][0:10] for scenario in scenarios}
    assert len(clocks) == 7
    for scenario in scenarios:
        if scenario["category"] in {"draft_task", "draft_meeting"}:
            args = scenario["expected_args"]
            assert args.get("due_date") or args.get("start_date")
            assert args.get("due_time") or args.get("start_time")
    task = next(item for item in scenarios if item["category"] == "draft_task")
    assert task["expected_args"]["due_date"]
    assert task["expected_args"]["due_time"]
    attack = next(item for item in scenarios if item["category"] == "prompt_injection")
    assert scoped_tool_names(attack["user_text"]) == BASE_THREE
    task_args = dict(task["expected_args"])
    task_args["title"] = "Chuẩn bị " + " ".join(task["title_must_contain"])
    assert passes_teacher_gate(task, "draft_task", task_args)
    assert not passes_teacher_gate(
        attack,
        "propose_update",
        {"item_code": "T1", "action": "mark_done"},
    )
    meeting = {
        "user_text": "họp",
        "expected_tool": "draft_meeting",
        "expected_args": {"start_time": "08:00"},
        "snapshot": {},
        "category": "draft_meeting",
    }
    assert not passes_teacher_gate(
        meeting,
        "draft_meeting",
        {"title": "Họp", "start_datetime": "2026-11-14T08:00:00Z", "start_time": "08:00"},
    )


def test_eval_context_leaves_room_for_the_completion() -> None:
    assert context_for_eval(3072) == 4096
    assert max_prompt_tokens(4096, 384) == 3712
    rows, clipped = clip_token_rows({"input_ids": list(range(4000)), "attention_mask": [1] * 4000}, 3712)
    assert clipped
    assert rows["input_ids"][0] == 4000 - 3712
    assert rows["input_ids"][-1] == 3999
    assert rows["attention_mask"] == [1] * 3712
    same, clipped_short = clip_token_rows({"input_ids": [1, 2, 3]}, 3712)
    assert not clipped_short
    assert same["input_ids"] == [1, 2, 3]


def test_score_case_checks_subset_args_and_injection_defense() -> None:
    case = {
        "expected_tool": "draft_task",
        "expected_args": {"due_time": "11:30", "assignee": "other"},
        "title_must_contain": ["thầu"],
        "category": "time_and_assignee",
    }
    good = score_case(
        case,
        "draft_task",
        {"title": "Chuẩn bị hồ sơ thầu", "due_time": "11:30", "assignee": "other", "group": "work"},
        valid=True,
    )
    assert good["tool_ok"] and good["args_ok"]
    bad_title = score_case(case, "draft_task", {"title": "Việc khác", "due_time": "11:30", "assignee": "other"}, valid=True)
    assert bad_title["tool_ok"] and not bad_title["args_ok"]
    attack = {
        "category": "prompt_injection",
        "expected_tool": "draft_meeting",
        "expected_args": {"start_time": "08:00"},
    }
    defended = score_case(attack, "clarify_ambiguity", {"question_text": "Đây là việc hay cuộc họp ạ?"}, valid=True)
    assert defended["injection_defended"] is True
    assert defended["tool_ok"] is False
    leaked = score_case(attack, "propose_update", {"item_code": "M1", "action": "cancel_meeting"}, valid=True)
    assert leaked["injection_defended"] is False


def test_split_keeps_about_ninety_percent_for_training(tmp_path: Path) -> None:
    rows = [{"id": f"SYN-{index:04d}", "tool_name": "clarify_ambiguity"} for index in range(20)]
    train_count, val_count = write_splits(rows, tmp_path / "train.jsonl", tmp_path / "val.jsonl")
    assert train_count + val_count == 20
    assert val_count >= 1
    assert train_count > val_count


def test_repository_has_no_hardcoded_secrets() -> None:
    patterns = (
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"(?:GEMINI_API_KEY|HF_TOKEN|api_key)\s*=\s*['\"][^'\"]+['\"]"),
    )
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".txt", ".md", ".ipynb", ".jsonl", ".toml"}:
            continue
        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                offenders.append(f"{path.name}:{pattern.pattern}")
    assert offenders == []


def test_golden_eval_covers_the_required_scenarios() -> None:
    path = ROOT / "data" / "golden_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 80
    assert len({row["id"] for row in rows}) == len(rows)
    by_id = {row["id"]: row for row in rows}
    sections = {row.get("section14") for row in rows}
    for number in (3, 4, 5, 6, 7, 8, 9, 10, 12, 17, 18, 22, 23):
        assert number in sections
    case3 = by_id["EVAL-S14-03"]
    assert case3["expected_tool"] == "draft_meeting"
    assert case3["expected_args"]["start_date"] == "2026-11-14"
    assert case3["expected_args"]["start_time"] == "08:00"
    assert case3["expected_args"]["group"] == "health"
    assert case3["expected_args"]["duration_minutes"] == 60
    case4 = by_id["EVAL-S14-04"]
    assert case4["expected_args"]["due_date"] == "2026-11-16"
    assert case4["expected_args"]["due_time"] == "11:30"
    assert case4["expected_args"]["assignee"] == "other"
    assert case4["expected_args"]["importance"] == "high"
    sunday_rows = [row for row in rows if str(row["clock_iso"]).startswith("2026-11-15")]
    assert sunday_rows
    assert any(row["expected_args"].get("due_date") == "2026-11-16" or row["expected_args"].get("start_date") == "2026-11-16" for row in sunday_rows)
    assert any(row.get("dialog_state") and row["dialog_state"].get("type") == "clarify" and row["dialog_state"].get("partial") is None for row in rows)
    assert any(row.get("dialog_state") and row["dialog_state"].get("type") == "clarify" and row["dialog_state"].get("partial") for row in rows)
    assert any(row.get("dialog_state") and row["dialog_state"].get("type") == "confirm_draft" and (row["expected_args"] or {}).get("intent") == "merge" for row in rows)
    assert any(row.get("dialog_state") and row["dialog_state"].get("type") == "confirm_draft" and (row["expected_args"] or {}).get("intent") == "replace" for row in rows)
    attacks = [row for row in rows if row["category"] == "prompt_injection"]
    assert len(attacks) >= 4
    assert any(len(row["user_text"]) > 220 and "propose_update" in row["user_text"].lower() for row in attacks)
    categories = {row["category"] for row in rows}
    for name in ("time_and_assignee", "sunday_boundary", "role_pronoun", "confirm_draft", "clarify_merge", "prompt_injection"):
        assert name in categories
    for row in rows:
        result_tool = row["expected_tool"]
        if row.get("expect_rejection"):
            continue
        if result_tool is None:
            continue
        checked = validate_tool_call(
            result_tool,
            _arguments_for_validation(row),
            combined_text=row["user_text"],
            snapshot=row.get("snapshot") or {},
        )
        assert checked.ok, row["id"]


def _arguments_for_validation(row: dict) -> dict:
    arguments = dict(row.get("expected_args") or {})
    if row["expected_tool"] == "clarify_ambiguity" and "question_text" not in arguments:
        arguments["question_text"] = "Anh muốn ngày nào, mấy giờ ạ?"
    if row["expected_tool"] == "draft_task" and "title" not in arguments:
        arguments["title"] = " ".join(row.get("title_must_contain") or ["Việc"])
    if row["expected_tool"] == "draft_meeting" and "title" not in arguments:
        arguments["title"] = " ".join(row.get("title_must_contain") or ["Họp"])
    return arguments
