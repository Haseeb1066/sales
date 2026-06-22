"""Shared sales dashboard logic for Flask and MCP."""

from __future__ import annotations

import json
import os
import re
from threading import Lock
from typing import Any

import tiktoken
from openai import OpenAI

SYSTEM_PROMPT = """You are a Tableau sales dashboard analyst.

Rules:
- Use ONLY facts present in the dashboard data (via tools or DASHBOARD_DATA). Do not invent metrics, regions, products, or dates.
- If something is not in the data, say clearly that the current extract does not show it.
- Prefer exact figures from the data. Replace $ with PKR when reporting currency if the data uses those symbols.
- For follow-ups, stay consistent with your earlier answers in this conversation when they refer to the same metrics.
- Be concise unless the user asks for detail.
- When tools are available, call them to look up specific worksheets or rows instead of guessing."""

MAX_DASHBOARD_CHARS = 80_000
MAX_DASHBOARD_TOKENS = 55_000
MAX_HISTORY_MESSAGES = 4
MAX_ASSISTANT_HISTORY_CHARS = 6_000
MAX_TOOL_ROUNDS = 6

_TIKTOKEN_ENC = None
_chat_lock = Lock()

dashboard_data_store: dict[str, Any] = {}
chat_history_store: dict[str, list[dict[str, str]]] = {}


def _get_encoder():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        _TIKTOKEN_ENC = tiktoken.encoding_for_model("gpt-4o-mini")
    return _TIKTOKEN_ENC


def session_key(dashboard: str, session_id: str) -> str:
    return f"{dashboard}\t{session_id}"


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    enc = _get_encoder()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens]) + (
        "\n\n[... truncated: dashboard text exceeded token budget; narrow filters or ask about one sheet ...]"
    )


def compact_dashboard_text(data, max_chars: int = MAX_DASHBOARD_CHARS) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return (
            data
            if len(data) <= max_chars
            else data[: max_chars - 200] + "\n\n[... truncated: extract exceeded size limit ...]"
        )
    if not isinstance(data, list):
        raw = str(data)
        return raw if len(raw) <= max_chars else raw[: max_chars - 200] + "\n\n[... truncated ...]"

    def build(max_rows_per_sheet: int) -> str:
        parts = []
        for ws in data:
            if not isinstance(ws, dict):
                parts.append(str(ws))
                continue
            name = ws.get("worksheet") or "Worksheet"
            cols = ws.get("columns") or []
            rows = ws.get("data") or []
            if max_rows_per_sheet > 0:
                rows = rows[:max_rows_per_sheet]
            header = " | ".join(str(c) for c in cols)
            lines = [f"=== {name} ===", header]
            for row in rows:
                lines.append(" | ".join(str(c) for c in row))
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    max_rows = 2_000
    text = build(max_rows)
    while len(text) > max_chars and max_rows > 10:
        max_rows = max(max_rows // 2, 10)
        text = build(max_rows)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n\n[... truncated: narrow filters or ask about a specific sheet ...]"


def _worksheets_for(dashboard: str) -> list[dict]:
    data = dashboard_data_store.get(dashboard)
    if not data or not isinstance(data, list):
        return []
    return [ws for ws in data if isinstance(ws, dict)]


def list_dashboards() -> list[str]:
    return sorted(dashboard_data_store.keys())


def list_worksheets(dashboard: str) -> list[str]:
    return [ws.get("worksheet") or "Worksheet" for ws in _worksheets_for(dashboard)]


def get_worksheet_preview(dashboard: str, worksheet: str, max_rows: int = 50) -> str:
    for ws in _worksheets_for(dashboard):
        name = ws.get("worksheet") or "Worksheet"
        if name.lower() != worksheet.lower():
            continue
        cols = ws.get("columns") or []
        rows = (ws.get("data") or [])[:max_rows]
        header = " | ".join(str(c) for c in cols)
        lines = [f"=== {name} ({len(ws.get('data') or [])} total rows) ===", header]
        for row in rows:
            lines.append(" | ".join(str(c) for c in row))
        if len(ws.get("data") or []) > max_rows:
            lines.append(f"[... {len(ws.get('data') or []) - max_rows} more rows not shown ...]")
        return "\n".join(lines)
    return f"Worksheet '{worksheet}' not found. Available: {', '.join(list_worksheets(dashboard))}"


def search_worksheet(
    dashboard: str,
    worksheet: str,
    column: str,
    value_contains: str,
    max_rows: int = 30,
) -> str:
    for ws in _worksheets_for(dashboard):
        name = ws.get("worksheet") or "Worksheet"
        if name.lower() != worksheet.lower():
            continue
        cols = [str(c) for c in (ws.get("columns") or [])]
        col_idx = next((i for i, c in enumerate(cols) if c.lower() == column.lower()), None)
        if col_idx is None:
            return f"Column '{column}' not found. Columns: {', '.join(cols)}"
        needle = value_contains.lower()
        matches = []
        for row in ws.get("data") or []:
            if col_idx < len(row) and needle in str(row[col_idx]).lower():
                matches.append(row)
            if len(matches) >= max_rows:
                break
        if not matches:
            return f"No rows in '{worksheet}' where '{column}' contains '{value_contains}'."
        header = " | ".join(cols)
        lines = [f"=== matches in {name} ===", header]
        for row in matches:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)
    return f"Worksheet '{worksheet}' not found."


DASHBOARD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_worksheets",
            "description": "List worksheet names available in the current dashboard extract.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_worksheet_preview",
            "description": "Get column headers and sample rows from a worksheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worksheet": {"type": "string", "description": "Worksheet name"},
                    "max_rows": {
                        "type": "integer",
                        "description": "Max rows to return (default 50)",
                    },
                },
                "required": ["worksheet"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_worksheet",
            "description": "Filter rows where a column contains a substring (case-insensitive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "worksheet": {"type": "string"},
                    "column": {"type": "string"},
                    "value_contains": {"type": "string"},
                    "max_rows": {"type": "integer"},
                },
                "required": ["worksheet", "column", "value_contains"],
                "additionalProperties": False,
            },
        },
    },
]


def _run_tool(dashboard: str, name: str, args: dict) -> str:
    if name == "list_worksheets":
        sheets = list_worksheets(dashboard)
        return json.dumps(sheets) if sheets else "No worksheets in store."
    if name == "get_worksheet_preview":
        return get_worksheet_preview(
            dashboard,
            args.get("worksheet", ""),
            int(args.get("max_rows") or 50),
        )
    if name == "search_worksheet":
        return search_worksheet(
            dashboard,
            args.get("worksheet", ""),
            args.get("column", ""),
            args.get("value_contains", ""),
            int(args.get("max_rows") or 30),
        )
    return f"Unknown tool: {name}"


def infer_reference_period(dashboard_text: str) -> str | None:
    match = re.search(r"Reference (Month|Period):\s*(\w+\s+\d{4})", dashboard_text)
    return match.group(2) if match else None


def build_user_message(
    question: str,
    current_period,
    reference_period,
    is_narrative: bool,
    *,
    tools_mode: bool,
    dashboard: str,
    dashboard_text: str = "",
) -> str:
    ctx_lines = []
    if current_period:
        ctx_lines.append(f"Current period: {current_period}")
    if reference_period:
        ctx_lines.append(f"Reference period: {reference_period}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "(not specified)"

    if is_narrative:
        task = (
            "Generate a concise executive summary using ONLY the dashboard data. "
            "Cover revenue trends, top and weak areas, and notable changes. "
            "Use clear language for business stakeholders. Include time periods where applicable. "
            "For comparisons use + or − with percentages (e.g. +12.5%, −7.3%)."
        )
    elif tools_mode:
        task = (
            "Answer using dashboard tools to fetch exact worksheet rows. "
            "For month-specific questions, use monthly rows; not YTD unless asked. "
            "Give the direct answer without explaining calculation steps."
        )
    else:
        task = (
            "Answer using ONLY the data below. "
            "For month-specific questions, use monthly rows; not YTD unless asked. "
            "Give the direct answer without explaining calculation steps."
        )

    if tools_mode:
        sheets = list_worksheets(dashboard)
        data_block = f"--- DASHBOARD ---\nName: {dashboard}\nWorksheets: {', '.join(sheets) or '(none)'}\n"
    else:
        data_block = f"--- DASHBOARD_DATA ---\n{dashboard_text}\n"

    return (
        f"{data_block}"
        f"--- CONTEXT ---\n{context_block}\n"
        f"--- TASK ---\n{task}\n"
        f"--- QUESTION ---\n{question.strip()}"
    )


def is_narrative_question(question: str) -> bool:
    q_lower = question.lower()
    return q_lower in (
        "generate summary",
        "give me a summary",
        "provide a summary narrative of the current dashboard data.",
        "dashboard summary",
        "summarize",
    ) or "summary" in q_lower


def ask_dashboard(
    client: OpenAI,
    *,
    dashboard: str,
    question: str,
    session_id: str = "default",
    current_period=None,
    reference_period=None,
    use_tools: bool | None = None,
) -> dict[str, Any]:
    if dashboard not in dashboard_data_store:
        return {"error": "Dashboard data not found"}
    data = dashboard_data_store[dashboard]
    if not data:
        return {"error": "No data available for this dashboard."}
    if not question.strip():
        return {"error": "question is required"}

    if use_tools is None:
        use_tools = os.getenv("SALES_USE_MCP_TOOLS", "1").strip().lower() in ("1", "true", "yes")

    dashboard_text = compact_dashboard_text(data)
    dashboard_text = truncate_to_token_budget(dashboard_text, MAX_DASHBOARD_TOKENS)

    if not reference_period:
        reference_period = infer_reference_period(dashboard_text)

    narrative = is_narrative_question(question)
    # Narratives need broad context; Q&A uses tools when enabled.
    tools_mode = use_tools and not narrative

    user_message = build_user_message(
        question,
        current_period,
        reference_period,
        narrative,
        tools_mode=tools_mode,
        dashboard=dashboard,
        dashboard_text=dashboard_text,
    )

    key = session_key(dashboard, session_id)
    with _chat_lock:
        prior = list(chat_history_store.get(key, []))

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in prior:
        content = m["content"]
        if m["role"] == "assistant" and len(content) > MAX_ASSISTANT_HISTORY_CHARS:
            content = (
                content[: MAX_ASSISTANT_HISTORY_CHARS - 120]
                + "\n[... earlier reply truncated for context size ...]"
            )
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": user_message})

    tools = DASHBOARD_TOOLS if tools_mode else None

    for _ in range(MAX_TOOL_ROUNDS):
        kwargs: dict[str, Any] = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = (msg.content or "").replace("$", "PKR ")
            with _chat_lock:
                hist = chat_history_store.setdefault(key, [])
                hist.append({"role": "user", "content": question})
                hist.append({"role": "assistant", "content": answer})
                if len(hist) > MAX_HISTORY_MESSAGES:
                    chat_history_store[key] = hist[-MAX_HISTORY_MESSAGES:]
            return {"answer": answer, "session_id": session_id}

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(dashboard, tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return {"error": "Tool loop exceeded max rounds"}


def reset_session(dashboard: str, session_id: str) -> None:
    key = session_key(dashboard, session_id)
    with _chat_lock:
        chat_history_store.pop(key, None)
