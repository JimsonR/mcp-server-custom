# main.py
# Dojo360 MCP Server – Python Implementation

from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

import os
import sys
import json
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import click

from typing import Dict, Any, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from loguru import logger

# ------------------------------------------------------------------
# Lazy registries (speed up startup)
# ------------------------------------------------------------------
_tool_registry = None
_resource_registry = None


def get_tool_registry():
    global _tool_registry
    if _tool_registry is None:
        from src.tools import tool_registry
        _tool_registry = tool_registry
    return _tool_registry


def get_resource_registry():
    global _resource_registry

    return _resource_registry


# ------------------------------------------------------------------
# Version metadata
# ------------------------------------------------------------------
VERSION = "1.0.0"
VERSION_PRERELEASE = "dev"
VERSION_METADATA = ""
GIT_COMMIT = ""
BUILD_DATE = "1970-01-01T00:00:01Z"


def get_human_version():
    version = VERSION
    if VERSION_PRERELEASE:
        version += f"-{VERSION_PRERELEASE}"
    if VERSION_METADATA:
        version += f"+{VERSION_METADATA}"
    return version.replace("'", "")


# ------------------------------------------------------------------
# CORS handling
# ------------------------------------------------------------------
def get_cors_allowed_origins() -> List[str]:
    allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if not allowed_origins_str:
        return ["*"]

    allowed_origins = [o.strip() for o in allowed_origins_str.split(",")]
    valid_origins = []

    for origin in allowed_origins:
        if origin == "*":
            return ["*"]
        elif origin.startswith("http://") or origin.startswith("https://"):
            valid_origins.append(origin)
        elif origin:
            logging.warning(
                f"Origin '{origin}' does not start with http:// or https://, adding prefix"
            )
            valid_origins.append(f"https://{origin}")

    if not valid_origins:
        logging.info("No valid origins found, defaulting to '*'")
        return ["*"]

    return valid_origins


# ------------------------------------------------------------------
# MCP Core Server
# ------------------------------------------------------------------
class MCPServer:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()

    def create_session(self, session_id: str):
        with self.session_lock:
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": "now",
            }

        from src.registry_client import new_http_client
        terraform_skip_tls_verify = (
            os.getenv("TERRAFORM_SKIP_TLS_VERIFY", "false").lower() == "true"
        )
        new_http_client(session_id, terraform_skip_tls_verify)
        logger.info(f"Created session: {session_id}")

    def delete_session(self, session_id: str):
        with self.session_lock:
            self.sessions.pop(session_id, None)

        from src.registry_client import delete_http_client
        delete_http_client(session_id)
        logger.info(f"Deleted session: {session_id}")

    # --------------------------------------------------------------
    # MCP Handlers
    # --------------------------------------------------------------
    def handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": True,
                        "concurrency": {"maxConcurrency": 10}
                    },
                    "resources": {"subscribe": True, "listChanged": True},
                },
                "serverInfo": {
                    "name": "dojo360-mcp-server",
                    "version": get_human_version(),
                },
            },
        }

    def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tools = []
        registry = get_tool_registry()

        for tool_name in registry.list_tools():
            tool = registry.get_tool(tool_name)

            if tool_name == "search_modules":
                input_schema = {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query that the user is interested in.",
                            "example": "Can you tell me about Dojo360?",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "The number of results to return.",
                            "example": 10,
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                }
            elif hasattr(tool, "api_base"):
                input_schema = getattr(tool, "input_schema", {
                    "type": "object",
                    "additionalProperties": True,
                    "description": f"Dynamic tool: {getattr(tool, 'description', '')}",
                })
            else:
                input_schema = {
                    "type": "object",
                    "properties": {
                        "provider_name": {"type": "string"},
                        "provider_namespace": {"type": "string"},
                        "service_slug": {"type": "string"},
                        "provider_data_type": {
                            "type": "string",
                            "enum": [
                                "resources",
                                "data-sources",
                                "functions",
                                "guides",
                                "overview",
                            ],
                            "default": "resources",
                        },
                        "provider_version": {"type": "string"},
                    },
                    "required": [
                        "provider_name",
                        "provider_namespace",
                        "service_slug",
                    ],
                }

            tools.append({
                "name": tool_name,
                "description": getattr(tool, "description", ""),
                "inputSchema": input_schema,
            })

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"tools": tools},
        }

    def handle_tools_call(self, request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        result = get_tool_registry().handle_tool_call(
            tool_name, session_id, arguments
        )

        if "error" in result:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": result["error"]},
            }

        content = result.get("content", "")

        # Elicitation handling
        if isinstance(content, dict) and content.get("elicitation"):
            questions = content.get("questions", [])
            question_text = "I need some additional information to proceed:\n\n"
            for i, q in enumerate(questions, 1):
                question_text += f"{i}. {q.get('question')}\n"
                if q.get("example"):
                    question_text += f"   Example: {q.get('example')}\n"

            required_fields = [
                q.get("field") for q in questions if q.get("field")
            ]
            if required_fields:
                question_text += (
                    "\nRequired fields: " + ", ".join(required_fields)
                )

            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": question_text}]
                },
            }

        if isinstance(content, dict):
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(content, indent=2),
                    }]
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "content": [{"type": "text", "text": str(content)}]
            },
        }

    def handle_resources_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        resources = []
        registry = get_resource_registry()

        for uri in registry.list_resources():
            resource = registry.get_resource(uri)
            resources.append({
                "uri": uri,
                "name": getattr(resource, "description", ""),
                "description": getattr(resource, "description", ""),
                "mimeType": getattr(resource, "mime_type", "text/plain"),
            })

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"resources": resources},
        }

    def handle_resources_read(
        self, request: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        params = request.get("params", {})
        uri = params.get("uri")

        result = get_resource_registry().handle_resource_request(
            uri, session_id, params
        )

        if "error" in result:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": result["error"]},
            }

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": result,
        }

    def handle_request(self, request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        method = request.get("method")

        if method == "initialize":
            return self.handle_initialize(request)
        elif method == "tools/list":
            return self.handle_tools_list(request)
        elif method == "tools/call":
            return self.handle_tools_call(request, session_id)
        elif method == "tools/call/stream":
            # For non-streaming HTTP contexts, collect all events and return final result
            return self._collect_streaming_response(request, session_id)
        elif method == "resources/list":
            return self.handle_resources_list(request)
        elif method == "resources/read":
            return self.handle_resources_read(request, session_id)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    def handle_tools_call_streaming(self, request: Dict[str, Any], session_id: str):
        """
        Generator that yields streaming events for a tool call.
        Use this method for SSE endpoints.
        """
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        for event in get_tool_registry().handle_tool_call_streaming(
            tool_name, session_id, arguments
        ):
            yield event

    def _collect_streaming_response(self, request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Collect all streaming events and return final result.
        Used for non-SSE contexts that still want to use streaming tools.
        """
        params = request.get("params", {})
        tool_name = params.get("name")
        
        final_analysis = ""
        final_usage = {}
        success = True
        error_msg = None
        
        for event in self.handle_tools_call_streaming(request, session_id):
            event_type = event.get("type")
            if event_type == "complete":
                final_analysis = event.get("analysis", "")
                final_usage = event.get("usage", {})
                success = event.get("success", True)
            elif event_type == "error":
                error_msg = event.get("error")
                success = False
        
        if error_msg:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": error_msg},
            }
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "content": [{
                    "type": "text",
                    "text": final_analysis
                }],
                "streaming": True,
                "usage": final_usage,
                "success": success
            },
        }


    def handle_batch_request(
        self,
        requests: List[Dict[str, Any]],
        session_id: str,
        max_workers: int = 10
    ) -> List[Dict[str, Any]]:
        """Handle batch of JSON-RPC requests concurrently.
        
        Executes multiple requests in parallel using ThreadPoolExecutor.
        Returns responses in completion order (not request order).
        
        Args:
            requests: List of JSON-RPC request objects
            session_id: Session identifier for the requests
            max_workers: Maximum number of concurrent workers (default: 10)
            
        Returns:
            List of JSON-RPC response objects
        """
        if not requests:
            return []
        
        # Limit max_workers to number of requests
        effective_workers = min(max_workers, len(requests))
        responses = []
        
        logger.info(f"Executing batch of {len(requests)} requests with {effective_workers} workers")
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all requests for concurrent execution
            future_to_request = {
                executor.submit(self.handle_request, req, session_id): req
                for req in requests
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_request):
                req = future_to_request[future]
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing batch request {req.get('id')}: {e}")
                    responses.append({
                        "jsonrpc": "2.0",
                        "id": req.get("id"),
                        "error": {"code": -32000, "message": str(e)}
                    })
        
        logger.info(f"Batch execution completed: {len(responses)} responses")
        return responses


# ------------------------------------------------------------------
# HTTP Transport
# ------------------------------------------------------------------
class MCPHTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        
        # Handle SSE streaming endpoint
        if path in ["/api/mcp/stream", "/mcp/stream"]:
            self._handle_stream_request()
            return
            
        if path not in ["/api/mcp", "/mcp"]:
            self.send_error(404)
            return

        try:
            content_length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))

            session_id = self.headers.get("X-Session-ID", "http-session")
            server = get_mcp_server()
            if session_id not in server.sessions:
                server.create_session(session_id)

            # Handle batch requests (JSON-RPC 2.0 array) or single requests
            if isinstance(data, list):
                # Batch request - execute concurrently
                response = server.handle_batch_request(data, session_id)
            else:
                # Single request
                response = server.handle_request(data, session_id)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        except Exception as e:
            logger.error(f"HTTP error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)}
            }).encode("utf-8"))

    def _handle_stream_request(self):
        """Handle SSE streaming request for tool calls."""
        try:
            content_length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))

            session_id = self.headers.get("X-Session-ID", "http-session")
            server = get_mcp_server()
            if session_id not in server.sessions:
                server.create_session(session_id)

            # Set up SSE response headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._set_cors_headers()
            self.end_headers()

            # Stream events
            for event in server.handle_tools_call_streaming(data, session_id):
                event_type = event.get("type", "message")
                event_data = json.dumps(event)
                
                # Format as SSE
                sse_message = f"event: {event_type}\ndata: {event_data}\n\n"
                self.wfile.write(sse_message.encode("utf-8"))
                self.wfile.flush()
            
            # Send end event
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()

        except Exception as e:
            logger.error(f"SSE streaming error: {e}")
            error_event = json.dumps({"type": "error", "error": str(e)})
            try:
                self.wfile.write(f"event: error\ndata: {error_event}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ["/api/health_check", "/health"]:
            self._handle_health()
        elif path in ["/api/mcp", "/mcp"]:
            self.send_response(405)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Method not allowed"}).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _set_cors_headers(self):
        allowed = get_cors_allowed_origins()
        origin = self.headers.get("Origin")

        if "*" in allowed:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin and origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
        elif allowed:
            self.send_header("Access-Control-Allow-Origin", allowed[0])

        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Origin, Content-Type, Content-Length, X-API-Key, X-Session-ID",
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, HEAD, OPTIONS",
        )

    def _handle_health(self):
        response = {
            "status": "ok",
            "service": "dojo360-mcp-server",
            "transport": "streamable-http",
            "endpoint": "/api/health_check",
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))


# ------------------------------------------------------------------
# Server bootstrap
# ------------------------------------------------------------------
_mcp_server = None


def get_mcp_server():
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server


def run_http_server(host="127.0.0.1", port=8080):
    logger.info(f"Starting MCP HTTP Server on {host}:{port}")
    server = HTTPServer((host, port), MCPHTTPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server")
        server.shutdown()


def run_stdio_server():
    logger.info("Starting MCP Server in stdio mode")
    session_id = "stdio-session"
    server = get_mcp_server()
    server.create_session(session_id)

    try:
        while True:
            line = input()
            if not line.strip():
                continue
            request = json.loads(line)
            response = server.handle_request(request, session_id)
            print(json.dumps(response))
            sys.stdout.flush()
    finally:
        server.delete_session(session_id)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
@click.group()
@click.version_option(get_human_version())
@click.option("--log-file", help="Path to log file")
def cli(log_file):
    if log_file:
        logger.add(log_file, rotation="10 MB")
    else:
        logger.add(sys.stderr, level="INFO")


@cli.command()
def stdio():
    run_stdio_server()


@cli.command("streamable-http")
@click.option("--transport-host", default="0.0.0.0")
@click.option("--transport-port", default=8080)
def streamable_http(transport_host, transport_port):
    run_http_server(transport_host, transport_port)


def main():
    functions_worker_runtime = os.getenv("FUNCTIONS_WORKER_RUNTIME")
    custom_handler_port = os.getenv("FUNCTIONS_CUSTOMHANDLER_PORT")

    if functions_worker_runtime and custom_handler_port:
        run_http_server("0.0.0.0", int(custom_handler_port))
    else:
        if len(sys.argv) > 1:
            cli()
        else:
            run_stdio_server()


if __name__ == "__main__":
    main()
