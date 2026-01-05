import os

from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

from typing import Dict, Any, List
from loguru import logger
import json

# from src.search_modules import SearchModulesHandler
from src.dynamic_tool import DynamicToolHandler
from src.local_tools import MemoryStoreTool, MemoryMutateTool, CreateVariableTool, DebtAnalystTool


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
                    if local_tool_type == "memory_store":
                        tool = MemoryStoreTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered local tool: {tool.name}")
                    elif local_tool_type == "memory_mutate":
                        tool = MemoryMutateTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered local tool: {tool.name}")
                    elif local_tool_type == "create_variable":
                        tool = CreateVariableTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered local tool: {tool.name}")
                    elif local_tool_type == "debt_analyst_agent":
                        tool = DebtAnalystTool(tool_config)
                        self.tools[tool.name] = tool
                        logger.info(f"Registered local tool: {tool.name}")
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


tool_registry = ToolRegistry()
