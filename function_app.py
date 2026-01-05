"""Azure Functions entrypoint for the MCP server."""

import json
from typing import Dict

import azure.functions as func
from loguru import logger

from main import get_cors_allowed_origins, get_mcp_server


def _build_cors_headers(origin: str | None, allowed: list[str]) -> Dict[str, str]:
	"""Create CORS headers mirroring logic in the HTTP server."""
	headers: Dict[str, str] = {
		"Access-Control-Allow-Headers": "Authorization, Origin, Content-Type, Content-Length, X-API-Key, X-Session-ID",
		"Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
	}

	if "*" in allowed:
		headers["Access-Control-Allow-Origin"] = "*"
	elif origin and origin in allowed:
		headers["Access-Control-Allow-Origin"] = origin
		headers["Access-Control-Allow-Credentials"] = "true"
	elif allowed:
		headers["Access-Control-Allow-Origin"] = allowed[0]

	return headers


def _ensure_session(session_id: str) -> None:
	server = get_mcp_server()
	if session_id not in server.sessions:
		server.create_session(session_id)


def _handle_health(headers: Dict[str, str]) -> func.HttpResponse:
	payload = {
		"status": "ok",
		"service": "dojo360-mcp-server",
		"transport": "azure-functions-http",
		"endpoint": "/api/mcp",
	}
	return func.HttpResponse(
		body=json.dumps(payload),
		status_code=200,
		mimetype="application/json",
		headers=headers,
	)


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="mcp", methods=["GET", "POST", "OPTIONS"], auth_level=func.AuthLevel.FUNCTION)
def mcp(req: func.HttpRequest) -> func.HttpResponse:
	origin = req.headers.get("Origin")
	allowed = get_cors_allowed_origins()
	headers = _build_cors_headers(origin, allowed)

	if req.method == "OPTIONS":
		return func.HttpResponse(status_code=200, headers=headers)

	if req.method == "GET":
		return _handle_health(headers)

	session_id = req.headers.get("X-Session-ID", "http-session")
	_ensure_session(session_id)

	try:
		request_body = req.get_json()
	except ValueError:
		logger.error("Invalid JSON payload in request body")
		return func.HttpResponse(
			body=json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Invalid JSON"}}),
			status_code=400,
			mimetype="application/json",
			headers=headers,
		)
	except Exception as exc:  # pragma: no cover - defensive
		logger.error(f"Unexpected error reading request body: {exc}")
		return func.HttpResponse(
			body=json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": "Request read failure"}}),
			status_code=500,
			mimetype="application/json",
			headers=headers,
		)

	server = get_mcp_server()

	try:
		response_payload = server.handle_request(request_body, session_id)
		return func.HttpResponse(
			body=json.dumps(response_payload),
			status_code=200,
			mimetype="application/json",
			headers=headers,
		)
	except Exception as exc:  # pragma: no cover - defensive
		logger.error(f"MCP handler error: {exc}")
		return func.HttpResponse(
			body=json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}}),
			status_code=500,
			mimetype="application/json",
			headers=headers,
		)