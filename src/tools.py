import os

from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

from typing import Dict, Any, List
from loguru import logger
import json

# from src.search_modules import SearchModulesHandler
from src.dynamic_tool import DynamicToolHandler
from src.local_tools import (
    MemoryStoreTool, 
    MemoryMutateTool, 
    CreateVariableTool, 
    DebtAnalystTool,
    LiquidityAnalystTool,
    QOEAnalystTool,
    AssetQualityAnalystTool,
    CriticAgentTool,
    ConsolidatedCriticTool
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
                    elif local_tool_type in ["debt_analyst_agent", "liquidity_analyst_agent", "qoe_analyst_agent", "asset_quality_analyst_agent", "critic_agent", "consolidated_analyst_critique"]:
                        # Create analyst tool
                        if local_tool_type == "debt_analyst_agent":
                            tool = DebtAnalystTool(tool_config)
                        elif local_tool_type == "liquidity_analyst_agent":
                            tool = LiquidityAnalystTool(tool_config)
                        elif local_tool_type == "qoe_analyst_agent":
                            tool = QOEAnalystTool(tool_config)
                        elif local_tool_type == "asset_quality_analyst_agent":
                            tool = AssetQualityAnalystTool(tool_config)
                        elif local_tool_type == "critic_agent":
                            tool = CriticAgentTool(tool_config)
                        elif local_tool_type == "consolidated_analyst_critique":
                            tool = ConsolidatedCriticTool(tool_config)
                        
                        # Inject memory tools - will be resolved after all tools are loaded
                        self.tools[tool.name] = tool
                        logger.info(f"Registered local tool: {tool.name} (analyst agent)")
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

        # Second pass: inject memory tools into analyst agents
        self._inject_memory_tools_to_analysts()

    def _inject_memory_tools_to_analysts(self) -> None:
        """Inject memory_tool and mutate_tool into analyst agents after all tools are loaded."""
        # Find memory tools and critic tool
        memory_tool = None
        mutate_tool = None
        critic_tool = None
        
        for tool in self.tools.values():
            if isinstance(tool, MemoryStoreTool):
                memory_tool = tool
            elif isinstance(tool, MemoryMutateTool):
                mutate_tool = tool
            elif isinstance(tool, CriticAgentTool):
                critic_tool = tool
        
        if not memory_tool:
            logger.warning("No MemoryStoreTool found - analyst agents will not have memory access")
            return
        
        # Inject into analyst agents and critic agent
        analyst_count = 0
        for tool in self.tools.values():
            if isinstance(tool, (DebtAnalystTool, LiquidityAnalystTool, QOEAnalystTool, AssetQualityAnalystTool, CriticAgentTool, ConsolidatedCriticTool)):
                tool.memory_tool = memory_tool
                if isinstance(tool, (DebtAnalystTool, LiquidityAnalystTool, QOEAnalystTool, AssetQualityAnalystTool)):
                    tool.mutate_tool = mutate_tool
                    # Give analysts access to critic tool
                    if critic_tool:
                        tool.critic_tool = critic_tool
                # Give critic agent access to all tools for context
                if isinstance(tool, CriticAgentTool):
                    tool.all_tools = self.tools
                analyst_count += 1
        
        logger.info(f"Injected memory tools into {analyst_count} analyst agent(s)")

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
