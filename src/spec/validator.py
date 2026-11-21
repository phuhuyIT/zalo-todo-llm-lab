"""Strict tool-call validator: enums, civil dates, HH:mm, snapshot codes, and +07:00 (§10)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.spec.tools_schema import scoped_tool_names

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^(\d{2}):(\d{2})$")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?P<offset>Z|[+-]\d{2}:\d{2})?",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"[.!?…]+")

GroupName = Literal["work", "health", "family"]
AssigneeName = Literal["self", "other", "unassigned"]
ImportanceName = Literal["high", "normal", "low"]
IntentName = Literal["merge", "replace"]
ActionName = Literal[
    "mark_done",
    "postpone_task",
    "reassign_task",
    "complete_meeting",
    "cancel_meeting",
    "reschedule_meeting",
]
TargetName = Literal["all", "tasks", "meetings", "overdue"]
AssigneeFilter = Literal["all", "self", "other"]


def has_forbidden_timestamp(value: str) -> bool:
    """Reject a ``Z`` suffix and any offset other than ``+07:00``. A bare civil time is allowed."""
    if re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z\b", value, flags=re.IGNORECASE):
        return True
    for match in _TIMESTAMP.finditer(value):
        offset = match.group("offset")
        if offset is None:
            continue
        if offset.upper() == "Z" or offset != "+07:00":
            return True
    return False


def is_real_date(value: str) -> bool:
    if not _DATE.fullmatch(value):
        return False
    year, month, day = (int(part) for part in value.split("-"))
    try:
        datetime(year, month, day)
    except ValueError:
        return False
    return True


def is_hhmm(value: str) -> bool:
    match = _TIME.fullmatch(value)
    if match is None:
        return False
    hour = int(match.group(1))
    minute = int(match.group(2))
    return 0 <= hour <= 23 and 0 <= minute <= 59


def _check_date(value: str | None, label: str) -> str | None:
    if value is None:
        return value
    if not is_real_date(value):
        raise ValueError(f"{label} must be a real YYYY-MM-DD date")
    return value


def _check_time(value: str | None, label: str) -> str | None:
    if value is None:
        return value
    if not is_hhmm(value):
        raise ValueError(f"{label} must be HH:mm from 00:00 to 23:59")
    return value


def _reject_offsets(value: str | None) -> str | None:
    if value is not None and has_forbidden_timestamp(value):
        raise ValueError("timestamp must use +07:00 and must not use Z")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _no_bad_offsets(self) -> "_Strict":
        for key, value in self.model_dump().items():
            if isinstance(value, str) and has_forbidden_timestamp(value):
                raise ValueError(f"{key} contains a Z timestamp or an offset other than +07:00")
        return self


class DraftTaskArgs(_Strict):
    title: str | None = None
    group: GroupName | None = None
    due_date: str | None = None
    due_time: str | None = None
    assignee: AssigneeName | None = None
    importance: ImportanceName | None = None
    note: str | None = None
    intent: IntentName | None = None

    @field_validator("title")
    @classmethod
    def _title(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not 1 <= len(value) <= 120:
            raise ValueError("title must be 1–120 characters")
        return value

    @field_validator("due_date")
    @classmethod
    def _due_date(cls, value: str | None) -> str | None:
        return _check_date(value, "due_date")

    @field_validator("due_time")
    @classmethod
    def _due_time(cls, value: str | None) -> str | None:
        return _check_time(value, "due_time")

    @field_validator("note")
    @classmethod
    def _note(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 300:
            raise ValueError("note must be at most 300 characters")
        return _reject_offsets(value)


class DraftMeetingArgs(_Strict):
    title: str | None = None
    group: GroupName | None = None
    start_date: str | None = None
    start_time: str | None = None
    duration_minutes: Literal[30, 60, 120] | None = None
    location: str | None = None
    note: str | None = None
    intent: IntentName | None = None
    start_datetime: str | None = None

    @field_validator("title")
    @classmethod
    def _title(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not 1 <= len(value) <= 120:
            raise ValueError("title must be 1–120 characters")
        return value

    @field_validator("start_date")
    @classmethod
    def _start_date(cls, value: str | None) -> str | None:
        return _check_date(value, "start_date")

    @field_validator("start_time")
    @classmethod
    def _start_time(cls, value: str | None) -> str | None:
        return _check_time(value, "start_time")

    @field_validator("location")
    @classmethod
    def _location(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 80:
            raise ValueError("location must be at most 80 characters")
        return value

    @field_validator("note")
    @classmethod
    def _note(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 300:
            raise ValueError("note must be at most 300 characters")
        return value

    @field_validator("start_datetime")
    @classmethod
    def _start_datetime(cls, value: str | None) -> str | None:
        return _reject_offsets(value)


class ProposeUpdateArgs(_Strict):
    item_code: str = Field(min_length=1)
    action: ActionName
    new_date: str | None = None
    new_time: str | None = None
    new_assignee: AssigneeName | None = None

    @field_validator("new_date")
    @classmethod
    def _new_date(cls, value: str | None) -> str | None:
        return _check_date(value, "new_date")

    @field_validator("new_time")
    @classmethod
    def _new_time(cls, value: str | None) -> str | None:
        return _check_time(value, "new_time")


class QueryScheduleArgs(_Strict):
    target: TargetName
    start_date: str | None = None
    end_date: str | None = None
    assignee_filter: AssigneeFilter | None = None

    @field_validator("start_date")
    @classmethod
    def _start(cls, value: str | None) -> str | None:
        return _check_date(value, "start_date")

    @field_validator("end_date")
    @classmethod
    def _end(cls, value: str | None) -> str | None:
        return _check_date(value, "end_date")


class ClarifyArgs(_Strict):
    question_text: str = Field(min_length=1, max_length=400)

    @field_validator("question_text")
    @classmethod
    def _two_sentences(cls, value: str) -> str:
        parts = [part.strip() for part in _SENTENCE_SPLIT.split(value.strip()) if part.strip()]
        if len(parts) > 2:
            raise ValueError("question_text must be at most two short sentences")
        return value


_MODELS: dict[str, type[BaseModel]] = {
    "draft_task": DraftTaskArgs,
    "draft_meeting": DraftMeetingArgs,
    "propose_update": ProposeUpdateArgs,
    "query_schedule": QueryScheduleArgs,
    "clarify_ambiguity": ClarifyArgs,
}


class ValidationIssue(BaseModel):
    code: str
    message: str


class ValidationResult(BaseModel):
    ok: bool
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)


def validate_tool_call(
    name: str,
    arguments: Mapping[str, Any] | None,
    *,
    combined_text: str,
    snapshot: Mapping[str, Any] | None = None,
) -> ValidationResult:
    """Validate one parsed tool call against the scoped tool list and the §10 field rules."""
    allowed = scoped_tool_names(combined_text)
    if name not in allowed:
        return ValidationResult(
            ok=False,
            tool=name,
            issues=[ValidationIssue(code="tool_not_scoped", message=f"{name} is not in the scoped tool list")],
        )
    model = _MODELS.get(name)
    if model is None:
        return ValidationResult(
            ok=False,
            tool=name,
            issues=[ValidationIssue(code="unknown_tool", message=f"Unknown tool {name}")],
        )
    payload = dict(arguments or {})
    try:
        parsed = model.model_validate(payload)
    except ValidationError as error:
        issues = [
            ValidationIssue(code="schema", message=issue["msg"])
            for issue in error.errors()
        ]
        return ValidationResult(ok=False, tool=name, issues=issues)
    dumped = parsed.model_dump(exclude_none=True)
    if name == "propose_update":
        code = str(dumped.get("item_code") or "")
        if snapshot is None or code not in snapshot:
            return ValidationResult(
                ok=False,
                tool=name,
                arguments=dumped,
                issues=[ValidationIssue(code="unknown_item_code", message=f"{code} is not in the current snapshot")],
            )
    return ValidationResult(ok=True, tool=name, arguments=dumped)
