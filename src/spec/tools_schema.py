"""Exact Phase 2 tool declarations and the forwarded-letter scoping boundary (§10)."""

from __future__ import annotations

from typing import Any

TOOL_ORDER: tuple[str, ...] = (
    "draft_task",
    "draft_meeting",
    "clarify_ambiguity",
    "propose_update",
    "query_schedule",
)

FORWARDED_BASE: tuple[str, ...] = (
    "draft_task",
    "draft_meeting",
    "clarify_ambiguity",
)

# Lowercased markers. A hit keeps only the three create/clarify tools, even if a draft is open.
FORWARD_MARKERS: tuple[str, ...] = (
    "kính gửi",
    "thông báo",
    "công văn",
    "giấy mời",
    "chuyển tiếp",
    "v/v:",
)

FORWARD_CODE_POINT_LIMIT = 220
FORWARD_NEWLINE_LIMIT = 4

_GROUP = ["work", "health", "family"]
_ASSIGNEE = ["self", "other", "unassigned"]
_IMPORTANCE = ["high", "normal", "low"]
_INTENT = ["merge", "replace"]
_ACTIONS = [
    "mark_done",
    "postpone_task",
    "reassign_task",
    "complete_meeting",
    "cancel_meeting",
    "reschedule_meeting",
]
_TARGETS = ["all", "tasks", "meetings", "overdue"]
_ASSIGNEE_FILTER = ["all", "self", "other"]


def _string(description: str, *, enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def _declarations() -> dict[str, dict[str, Any]]:
    return {
        "draft_task": {
            "description": "Tạo hoặc sửa nháp công việc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": _string("Nội dung việc, 1–120 ký tự."),
                    "group": _string("Nhóm. Mặc định work.", enum=_GROUP),
                    "due_date": _string("Ngày hạn YYYY-MM-DD. Bỏ trống nếu chưa có ngày."),
                    "due_time": _string("Giờ hạn HH:mm. Bỏ trống nếu cả ngày."),
                    "assignee": _string("Người phụ trách.", enum=_ASSIGNEE),
                    "importance": _string("Mức ưu tiên. Mặc định normal.", enum=_IMPORTANCE),
                    "note": _string("Ghi chú, tối đa 300 ký tự."),
                    "intent": _string("merge khi sửa nháp đang mở, replace khi đổi sang việc khác.", enum=_INTENT),
                },
                "required": [],
            },
        },
        "draft_meeting": {
            "description": "Tạo hoặc sửa nháp cuộc họp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": _string("Nội dung cuộc họp, 1–120 ký tự."),
                    "group": _string("Nhóm. Mặc định work.", enum=_GROUP),
                    "start_date": _string("Ngày bắt đầu YYYY-MM-DD."),
                    "start_time": _string("Giờ bắt đầu HH:mm."),
                    "duration_minutes": {
                        "type": "integer",
                        "enum": [30, 60, 120],
                        "description": "30, 60, hoặc 120. Mặc định 60.",
                    },
                    "location": _string("Địa điểm, tối đa 80 ký tự."),
                    "note": _string("Ghi chú, tối đa 300 ký tự."),
                    "intent": _string("merge khi sửa nháp đang mở, replace khi đổi sang cuộc khác.", enum=_INTENT),
                    "start_datetime": _string(
                        "YYYY-MM-DDTHH:mm:00+07:00. Cấm hậu tố Z và mọi offset khác +07:00."
                    ),
                },
                "required": [],
            },
        },
        "propose_update": {
            "description": "Đề xuất cập nhật một mục trong danh sách.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_code": _string("Mã trong snapshot hiện tại, ví dụ T1 hoặc M1."),
                    "action": _string("Thao tác cập nhật.", enum=_ACTIONS),
                    "new_date": _string("Ngày mới YYYY-MM-DD khi dời hạn hoặc dời họp."),
                    "new_time": _string("Giờ mới HH:mm."),
                    "new_assignee": _string("Người nhận khi giao lại việc.", enum=_ASSIGNEE),
                },
                "required": ["item_code", "action"],
            },
        },
        "query_schedule": {
            "description": "Xem lịch theo khoảng ngày.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": _string("Tập cần xem.", enum=_TARGETS),
                    "start_date": _string("Ngày đầu YYYY-MM-DD, inclusive."),
                    "end_date": _string("Ngày cuối YYYY-MM-DD, inclusive."),
                    "assignee_filter": _string("Lọc việc theo người. Họp không có người phụ trách.", enum=_ASSIGNEE_FILTER),
                },
                "required": ["target"],
            },
        },
        "clarify_ambiguity": {
            "description": "Hỏi lại một câu ngắn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_text": _string("Tối đa hai câu tiếng Việt ngắn."),
                },
                "required": ["question_text"],
            },
        },
    }


_DECLARATIONS = _declarations()


def tool_schema(name: str) -> dict[str, Any]:
    """Hermes / OpenAI function schema for one tool."""
    spec = _DECLARATIONS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        },
    }


def gemini_declaration(name: str) -> dict[str, Any]:
    """Gemini function declaration. Integer enums stay JSON numbers."""
    spec = _DECLARATIONS[name]
    parameters = spec["parameters"]
    properties: dict[str, Any] = {}
    for key, raw in parameters["properties"].items():
        prop = dict(raw)
        json_type = prop.pop("type")
        prop["type"] = "INTEGER" if json_type == "integer" else "STRING"
        properties[key] = prop
    return {
        "name": name,
        "description": spec["description"],
        "parameters": {
            "type": "OBJECT",
            "properties": properties,
            "required": list(parameters.get("required") or []),
        },
    }


def is_forwarded_text(combined_text: str) -> bool:
    """True when the prose is a long or official forward and must not update or list items.

    The length check counts Unicode code points. Four or more newline characters trip the
    boundary. Marker matching is case-insensitive. An open draft does not widen the list.
    """
    if len(combined_text) > FORWARD_CODE_POINT_LIMIT:
        return True
    if combined_text.count("\n") >= FORWARD_NEWLINE_LIMIT:
        return True
    lowered = combined_text.lower()
    return any(marker in lowered for marker in FORWARD_MARKERS)


def scoped_tool_names(combined_text: str) -> list[str]:
    if is_forwarded_text(combined_text):
        return list(FORWARDED_BASE)
    return list(TOOL_ORDER)


def get_scoped_tools(combined_text: str) -> list[dict[str, Any]]:
    """Tool schemas the model is allowed to call for this combined prose."""
    return [tool_schema(name) for name in scoped_tool_names(combined_text)]


def tool_names_from_schemas(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if "function" in tool and isinstance(tool["function"], dict):
            names.append(str(tool["function"]["name"]))
        elif "name" in tool:
            names.append(str(tool["name"]))
    return names
