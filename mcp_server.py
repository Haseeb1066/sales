#!/usr/bin/env python3
"""Minimal MCP stdio server for the Tableau sales dashboard backend.

Works on Python 3.9+ without the official `mcp` SDK (which requires 3.10+).
Set SALES_API_BASE to your Flask URL (default http://127.0.0.1:5000).
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import httpx

API_BASE = os.getenv("SALES_API_BASE", "http://127.0.0.1:5000").rstrip("/")
SERVER_NAME = "sales-dashboard"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "list_dashboards",
        "description": "List dashboard names currently stored on the sales backend.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_worksheets",
        "description": "List worksheet names for a stored dashboard.",
        "inputSchema": {
            "type": "object",
            "properties": {"dashboard": {"type": "string"}},
            "required": ["dashboard"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ask_sales_question",
        "description": "Ask a natural-language question about a dashboard (same as the Tableau chat).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dashboard": {"type": "string"},
                "question": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["dashboard", "question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reset_chat_session",
        "description": "Clear conversation history for a dashboard session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dashboard": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["dashboard"],
            "additionalProperties": False,
        },
    },
]


def _post(path: str, body: dict) -> dict:
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{API_BASE}{path}", json=body)
        r.raise_for_status()
        return r.json()


def _get(path: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{API_BASE}{path}")
        r.raise_for_status()
        return r.json()


def _tool_list_dashboards(_args: dict) -> str:
    data = _get("/api/dashboards")
    dashboards = data.get("dashboards") or []
    if not dashboards:
        return "No dashboards in store. Open the Tableau extension so it can sync data to the backend."
    return json.dumps(dashboards, indent=2)


def _tool_list_worksheets(args: dict) -> str:
    dashboard = args["dashboard"]
    data = _get(f"/api/dashboard/{dashboard}/worksheets")
    return json.dumps(data.get("worksheets") or [], indent=2)


def _tool_ask(args: dict) -> str:
    data = _post(
        "/api/ask",
        {
            "dashboard": args["dashboard"],
            "question": args["question"],
            "session_id": args.get("session_id") or "mcp",
        },
    )
    if data.get("error"):
        return f"Error: {data['error']}"
    return data.get("answer") or "(no answer)"


def _tool_reset(args: dict) -> str:
    data = _post(
        "/api/reset_session",
        {
            "dashboard": args["dashboard"],
            "session_id": args.get("session_id") or "mcp",
        },
    )
    return json.dumps(data)


TOOL_HANDLERS = {
    "list_dashboards": _tool_list_dashboards,
    "list_worksheets": _tool_list_worksheets,
    "ask_sales_question": _tool_ask,
    "reset_chat_session": _tool_reset,
}


def _send(msg: dict) -> None:
    body = json.dumps(msg, ensure_ascii=False)
    sys.stdout.write(f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n{body}")
    sys.stdout.flush()


def _read_message() -> dict | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    raw = sys.stdin.buffer.read(length)
    return json.loads(raw.decode("utf-8"))


def _ok(req_id, result: dict) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _handle(req: dict) -> None:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        _ok(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
        return

    if method == "notifications/initialized":
        return

    if method == "ping":
        _ok(req_id, {})
        return

    if method == "tools/list":
        _ok(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True,
                },
            )
            return
        try:
            text = handler(args)
            _ok(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as exc:
            _ok(
                req_id,
                {
                    "content": [{"type": "text", "text": f"{exc}\n{traceback.format_exc()}"}],
                    "isError": True,
                },
            )
        return

    if req_id is not None:
        _err(req_id, -32601, f"Method not found: {method}")


def main() -> None:
    while True:
        msg = _read_message()
        if msg is None:
            break
        try:
            _handle(msg)
        except Exception:
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
