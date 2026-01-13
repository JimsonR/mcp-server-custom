import os

from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

from typing import Dict, Any, List
from loguru import logger
import json

# from src.search_modules import SearchModulesHandler
from src.dynamic_tool import DynamicToolHandler

from src.yfinance_tools import (
    GetHistoricalStockPricesTool,
    GetStockInfoTool,
    GetYahooFinanceNewsTool,
    GetStockActionsTool,
    GetFinancialStatementTool,
    GetHolderInfoTool,
    GetOptionExpirationDatesTool,
    GetOptionChainTool,
    GetRecommendationsTool,
)


class ToolRegistry:
    """
    Registry for all available tools.
    """

    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self._register_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def _register_tools(self):
        """Register all available tools."""

        # Register search module tool
        # search_modules = SearchModulesHandler()
        # self.tools[search_modules.name] = search_modules

        # Load dynamic tools
        try:
            self._load_dynamic_tools()
        except Exception as e:
            logger.error(f"Critical error loading dynamic tools: {e}")

        logger.info(f"Registered {len(self.tools)} tools")

    # ------------------------------------------------------------------
    # Load dynamic tools from config
    # ------------------------------------------------------------------
    def _load_dynamic_tools(self):
        """Load tools from JSON configuration."""

        config_path = os.getenv("DYNAMIC_TOOLS_CONFIG", "tools_config.json")
        logger.info(f"Looking for dynamic tools config at: {config_path}")

        if not os.path.exists(config_path):
            logger.info(f"Dynamic tools config file not found: {config_path}")
            return

        try:
            with open(config_path, "r") as f:
                tools_config = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tools config: {e}")
            return
        except PermissionError:
            logger.error(f"Permission denied reading tools config: {config_path}")
            return
        except Exception as e:
            logger.error(f"Failed to read tools config file: {e}")
            return

        if not isinstance(tools_config, dict):
            logger.error("Tools config must be a JSON object")
            return

        tools_list = tools_config.get("tools", [])
        if not isinstance(tools_list, list):
            logger.error("Tools config 'tools' must be an array")
            return

        logger.info(f"Loaded config with {len(tools_list)} tools")

        for i, tool_config in enumerate(tools_list):
            if not isinstance(tool_config, dict):
                logger.warning(
                    f"Tool config at index {i} is not an object, skipping"
                )
                continue

            tool_name = tool_config.get("name", f"tool_{i}")

            # Validate required fields
            if not tool_config.get("name"):
                logger.warning(
                    f"Tool at index {i} missing 'name' field, skipping"
                )
                continue

            if not tool_config.get("api_base"):
                logger.warning(
                    f"Tool '{tool_name}' missing 'api_base' field, skipping"
                )
                continue

            try:
                if tool_config.get("api_base") == "local":
                    local_tool_type = tool_config.get("local_tool") or tool_config.get("handler")
                    if local_tool_type == "get_historical_stock_prices":
                        tool = GetHistoricalStockPricesTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_stock_info":
                        tool = GetStockInfoTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_yahoo_finance_news":
                        tool = GetYahooFinanceNewsTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_stock_actions":
                        tool = GetStockActionsTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_financial_statement":
                        tool = GetFinancialStatementTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_holder_info":
                        tool = GetHolderInfoTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_option_expiration_dates":
                        tool = GetOptionExpirationDatesTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_option_chain":
                        tool = GetOptionChainTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    elif local_tool_type == "get_recommendations":
                        tool = GetRecommendationsTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered yfinance tool: {tool.name}")
                    else:
                        logger.warning(
                            f"Unknown local tool '{tool_name}' (local_tool={local_tool_type}), skipping"
                        )
                        continue
                else:
                    tool = DynamicToolHandler(tool_config)
                    self.tools[tool.name] = tool
                    logger.info(f"Registered dynamic tool: {tool.name}")

            except KeyError as e:
                logger.warning(
                    f"Tool '{tool_name}' missing required field {e}, skipping"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to register tool '{tool_name}': {e}"
                )

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------
    def get_tool(self, name: str) -> Any:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def handle_tool_call(
        self,
        tool_name: str,
        session_id: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a tool call."""

        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            return tool.handle(session_id, arguments)
        except Exception as e:
            logger.error(
                f"Error handling tool call for '{tool_name}': {e}"
            )
            return {"error": str(e)}

    def handle_tool_call_streaming(
        self,
        tool_name: str,
        session_id: str,
        arguments: Dict[str, Any],
    ):
        """
        Handle a streaming tool call.
        
        Returns a generator that yields events for tools that support streaming.
        For tools without streaming support, yields a single complete event.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            yield {"type": "error", "error": f"Tool '{tool_name}' not found"}
            return

        try:
            # Check if the tool supports streaming
            if hasattr(tool, 'handle_streaming') and callable(getattr(tool, 'handle_streaming')):
                logger.info(f"Starting streaming tool call for '{tool_name}'")
                for event in tool.handle_streaming(session_id, arguments):
                    yield event
            else:
                # Fallback to regular handle and yield as complete event
                logger.info(f"Tool '{tool_name}' does not support streaming, using regular handle")
                result = tool.handle(session_id, arguments)
                if "error" in result:
                    yield {"type": "error", "error": result["error"], "agent": tool_name}
                else:
                    content = result.get("content", {})
                    analysis = content.get("analysis", "") if isinstance(content, dict) else str(content)
                    yield {
                        "type": "complete",
                        "analysis": analysis,
                        "agent": tool_name,
                        "success": content.get("success", True) if isinstance(content, dict) else True,
                        "usage": content.get("usage", {}) if isinstance(content, dict) else {}
                    }
        except Exception as e:
            logger.error(f"Error in streaming tool call for '{tool_name}': {e}")
            yield {"type": "error", "error": str(e), "agent": tool_name}


tool_registry = ToolRegistry()

