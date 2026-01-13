"""
Yahoo Finance MCP Server
A simple MCP server providing access to Yahoo Finance data.

Usage:
    python server.py                    # Start HTTP server on default port 8080
    python server.py --port 9000        # Start on custom port
    python server.py --transport stdio  # Use stdio transport (for MCP clients)
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

from loguru import logger

# Import the main server components
from main import run_http_server, run_stdio_server, get_mcp_server


def main():
    parser = argparse.ArgumentParser(
        description="Yahoo Finance MCP Server with HTTP/stdio transport support"
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["http", "stdio"],
        default="http",
        help="Transport type to use (http or stdio). Default: http",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP server to. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the HTTP server to. Default: PORT env var or 8080",
    )
    
    args = parser.parse_args()
    
    # Use PORT from environment if not specified via CLI
    if args.port is None:
        args.port = int(os.environ.get("PORT", 8080))
    
    # Configure logging
    logger.add(sys.stderr, level="INFO")
    
    if args.transport == "stdio":
        print("Starting Yahoo Finance MCP server with stdio transport...")
        run_stdio_server()
    else:  # http
        print(f"Starting Yahoo Finance MCP server on http://{args.host}:{args.port}")
        print(f"\nAvailable endpoints:")
        print(f"  POST http://{args.host}:{args.port}/mcp       - JSON-RPC endpoint")
        print(f"  POST http://{args.host}:{args.port}/mcp/stream - SSE streaming endpoint")
        print(f"  GET  http://{args.host}:{args.port}/health     - Health check")
        print(f"\nAvailable tools:")
        
        # Initialize server to load tools
        server = get_mcp_server()
        from src.tools import tool_registry
        
        for tool_name in tool_registry.list_tools():
            print(f"  - {tool_name}")
        
        print(f"\nExample curl request:")
        print(f'''  curl -X POST http://{args.host}:{args.port}/mcp \\
    -H "Content-Type: application/json" \\
    -H "X-Session-ID: test-session" \\
    -d '{{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {{"name": "get_stock_info", "arguments": {{"ticker": "AAPL"}}}}}}'
''')
        run_http_server(args.host, args.port)


if __name__ == "__main__":
    main()
