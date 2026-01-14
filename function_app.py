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
		"service": "finance-mcp-server",
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
		# Handle batch requests (JSON-RPC 2.0 array) or single requests
		if isinstance(request_body, list):
			# Batch request - execute concurrently
			response_payload = server.handle_batch_request(request_body, session_id)
		else:
			# Single request
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


@app.route(route="mcp/stream", methods=["POST", "OPTIONS"], auth_level=func.AuthLevel.FUNCTION)
def mcp_stream(req: func.HttpRequest) -> func.HttpResponse:
	"""
	Streaming endpoint for MCP tool calls.
	
	Since Azure Functions doesn't support true SSE streaming,
	this endpoint collects all streaming events and returns them as a JSON response.
	The response includes all events in an array, allowing the client to process them.
	"""
	origin = req.headers.get("Origin")
	allowed = get_cors_allowed_origins()
	headers = _build_cors_headers(origin, allowed)

	if req.method == "OPTIONS":
		return func.HttpResponse(status_code=200, headers=headers)

	session_id = req.headers.get("X-Session-ID", "http-session")
	logger.info(f"Streaming endpoint - session_id from header: {session_id}")
	_ensure_session(session_id)

	try:
		request_body = req.get_json()
	except ValueError:
		logger.error("Invalid JSON payload in streaming request")
		return func.HttpResponse(
			body=json.dumps({"error": "Invalid JSON"}),
			status_code=400,
			mimetype="application/json",
			headers=headers,
		)

	server = get_mcp_server()
	logger.info(f"Streaming endpoint - available sessions: {list(server.sessions.keys())}")
	
	# WORKAROUND: If streaming session has no variables, copy from main client session
	# The mcp_agent_bridge uses different session IDs for streaming (stream_*) vs regular (client-*) calls
	streaming_session = server.sessions.get(session_id, {})
	streaming_memory = streaming_session.get("memory", {})
	
	if not streaming_memory:
		# Find a client-* session that has variables and copy them
		for other_session_id, other_session in server.sessions.items():
			if other_session_id.startswith("client-"):
				other_memory = other_session.get("memory", {})
				if other_memory:
					logger.info(f"Streaming session {session_id} empty - copying memory from {other_session_id}")
					# Copy the memory reference
					if session_id not in server.sessions:
						server.sessions[session_id] = {}
					server.sessions[session_id]["memory"] = other_memory
					break

	try:
		# Collect all streaming events
		events = []
		final_analysis = ""
		final_usage = {}
		success = True
		error_msg = None
		
		for event in server.handle_tools_call_streaming(request_body, session_id):
			events.append(event)
			event_type = event.get("type")
			if event_type == "complete":
				final_analysis = event.get("analysis", "")
				final_usage = event.get("usage", {})
				success = event.get("success", True)
			elif event_type == "error":
				error_msg = event.get("error")
				success = False
		
		# Return JSON-RPC response with streaming metadata
		if error_msg:
			response_payload = {
				"jsonrpc": "2.0",
				"id": request_body.get("id"),
				"error": {"code": -32000, "message": error_msg},
				"streaming_events": events
			}
		else:
			response_payload = {
				"jsonrpc": "2.0",
				"id": request_body.get("id"),
				"result": {
					"content": [{"type": "text", "text": final_analysis}],
					"streaming": True,
					"usage": final_usage,
					"success": success,
					"events": events  # Include all streaming events for debugging/processing
				}
			}
		
		return func.HttpResponse(
			body=json.dumps(response_payload),
			status_code=200,
			mimetype="application/json",
			headers=headers,
		)
	except Exception as exc:
		logger.error(f"MCP streaming handler error: {exc}")
		return func.HttpResponse(
			body=json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}}),
			status_code=500,
			mimetype="application/json",
			headers=headers,
		)