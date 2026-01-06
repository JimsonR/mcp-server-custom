from __future__ import annotations

import json
import os
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

# Session-scoped in-memory store.
# Persists for the lifetime of the Python process.
_SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def _get_session_store(session_id: str) -> Dict[str, Any]:
    if session_id not in _SESSION_STATE:
        _SESSION_STATE[session_id] = {}
    return _SESSION_STATE[session_id]


def _ensure_json_serializable(value: Any) -> None:
    # Value comes from MCP arguments (JSON), but validate to avoid surprises.
    json.dumps(value)


@dataclass
class MemoryStoreTool:
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "Session-scoped in-memory key/value store (supports lists)."
        )

        # Keep parity with DynamicToolHandler so main.py's tools/list uses input_schema.
        self.api_base: str = "local"

        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": [
                        "set",
                        "get",
                        "delete",
                        "keys",
                        "clear",
                        "append",
                        "extend",
                        "pop",
                        "dump",
                    ],
                },
                "key": {"type": "string", "description": "Variable name"},
                "value": {
                    "description": "JSON-serializable value",
                    "type": ["string", "number", "object", "array", "boolean", "null"],
                },
                "values": {
                    "description": "List of values for extend",
                    "type": "array",
                    "items": {
                        "type": ["string", "number", "object", "array", "boolean", "null"]
                    },
                },
                "default": {
                    "description": "Default returned when key is missing (get)",
                    "type": ["string", "number", "object", "array", "boolean", "null"],
                },
                "index": {
                    "description": "Index for pop (defaults to -1)",
                    "type": "integer",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        action = str(args.get("action") or "").strip().lower()

        store = _get_session_store(session_id)

        if action == "keys":
            return {"content": {"keys": sorted(store.keys())}}

        if action == "clear":
            key = args.get("key")
            if key is None:
                store.clear()
                return {"content": {"cleared": True, "scope": "session"}}
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}
            existed = key in store
            store.pop(key, None)
            return {"content": {"cleared": existed, "key": key}}

        if action == "delete":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}
            existed = key in store
            store.pop(key, None)
            return {"content": {"deleted": existed, "key": key}}

        if action == "set":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}
            value = args.get("value")
            try:
                _ensure_json_serializable(value)
            except Exception as e:
                return {"error": f"'value' must be JSON-serializable: {e}"}
            store[key] = value
            return {"content": {"key": key, "value": value}}

        if action == "get":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}
            default = args.get("default")
            value = store.get(key, default)
            return {"content": {"key": key, "value": value, "found": key in store}}

        if action == "append":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}
            value = args.get("value")
            try:
                _ensure_json_serializable(value)
            except Exception as e:
                return {"error": f"'value' must be JSON-serializable: {e}"}

            existing = store.get(key)
            if existing is None:
                existing = []
                store[key] = existing
            if not isinstance(existing, list):
                return {"error": f"Key '{key}' is not a list (found {type(existing).__name__})"}

            existing.append(value)
            return {"content": {"key": key, "value": existing, "length": len(existing)}}

        if action == "extend":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}
            values = args.get("values")
            if not isinstance(values, list):
                return {"error": "'values' must be an array"}
            try:
                _ensure_json_serializable(values)
            except Exception as e:
                return {"error": f"'values' must be JSON-serializable: {e}"}

            existing = store.get(key)
            if existing is None:
                existing = []
                store[key] = existing
            if not isinstance(existing, list):
                return {"error": f"Key '{key}' is not a list (found {type(existing).__name__})"}

            existing.extend(values)
            return {"content": {"key": key, "value": existing, "length": len(existing)}}

        if action == "pop":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                return {"error": "'key' must be a non-empty string"}

            existing = store.get(key)
            if not isinstance(existing, list):
                return {"error": f"Key '{key}' is not a list (found {type(existing).__name__})"}

            index = args.get("index", -1)
            if not isinstance(index, int):
                return {"error": "'index' must be an integer"}

            if not existing:
                return {"error": f"List at key '{key}' is empty"}

            try:
                item = existing.pop(index)
            except Exception as e:
                return {"error": str(e)}

            return {"content": {"key": key, "popped": item, "value": existing, "length": len(existing)}}

        if action == "dump":
            # Returns the entire session store (useful for debugging).
            # Keep it simple; caller can store only what they need.
            try:
                _ensure_json_serializable(store)
            except Exception:
                # Should not happen if we enforce on set/append/extend.
                return {"error": "Session store contains non-JSON-serializable values"}
            return {"content": {"state": store}}

        return {"error": "Invalid 'action'. Use one of: set/get/delete/keys/clear/append/extend/pop/dump"}


@dataclass
class MemoryMutateTool:
    """Tool for mutating existing variables (lists, dicts, numbers, etc.)."""
    
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "Mutate existing in-memory variables (insert, remove, update, increment, etc.)."
        )

        self.api_base: str = "local"

        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Mutation operation",
                    "enum": [
                        "list_insert",
                        "list_remove",
                        "list_reverse",
                        "list_sort",
                        "dict_update",
                        "dict_merge",
                        "dict_delete_key",
                        "dict_get_nested",
                        "dict_set_nested",
                        "increment",
                        "decrement",
                        "multiply",
                        "toggle_bool",
                        "string_concat",
                        "string_replace",
                    ],
                },
                "key": {"type": "string", "description": "Variable name"},
                "index": {"type": "integer", "description": "Index for list operations"},
                "value": {
                    "description": "Value to insert/use",
                    "type": ["string", "number", "object", "array", "boolean", "null"],
                },
                "old": {"type": "string", "description": "Old substring (string_replace)"},
                "new": {"type": "string", "description": "New substring (string_replace)"},
                "amount": {"type": "number", "description": "Amount for increment/decrement/multiply"},
                "reverse": {"type": "boolean", "description": "Reverse sort order"},
                "path": {
                    "type": "array",
                    "description": "Nested path for dict operations (e.g., ['a', 'b', 'c'])",
                    "items": {"type": "string"},
                },
                "updates": {
                    "type": "object",
                    "description": "Key-value pairs to update (dict_update/dict_merge)",
                    "additionalProperties": True,
                },
                "dict_key": {"type": "string", "description": "Key to delete from dict"},
            },
            "required": ["action", "key"],
            "additionalProperties": False,
        }

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        action = str(args.get("action") or "").strip().lower()
        key = args.get("key")

        if not isinstance(key, str) or not key:
            return {"error": "'key' must be a non-empty string"}

        store = _get_session_store(session_id)

        if key not in store:
            return {"error": f"Variable '{key}' does not exist"}

        existing = store[key]

        # List operations
        if action == "list_insert":
            if not isinstance(existing, list):
                return {"error": f"Variable '{key}' is not a list"}
            index = args.get("index")
            if not isinstance(index, int):
                return {"error": "'index' must be an integer"}
            value = args.get("value")
            try:
                _ensure_json_serializable(value)
                existing.insert(index, value)
            except Exception as e:
                return {"error": str(e)}
            return {"content": {"key": key, "value": existing, "length": len(existing)}}

        if action == "list_remove":
            if not isinstance(existing, list):
                return {"error": f"Variable '{key}' is not a list"}
            value = args.get("value")
            try:
                existing.remove(value)
            except ValueError:
                return {"error": f"Value not found in list: {value}"}
            except Exception as e:
                return {"error": str(e)}
            return {"content": {"key": key, "value": existing, "length": len(existing)}}

        if action == "list_reverse":
            if not isinstance(existing, list):
                return {"error": f"Variable '{key}' is not a list"}
            existing.reverse()
            return {"content": {"key": key, "value": existing, "length": len(existing)}}

        if action == "list_sort":
            if not isinstance(existing, list):
                return {"error": f"Variable '{key}' is not a list"}
            reverse = args.get("reverse", False)
            try:
                existing.sort(reverse=reverse)
            except Exception as e:
                return {"error": f"Cannot sort list: {e}"}
            return {"content": {"key": key, "value": existing, "length": len(existing)}}

        # Dict operations
        if action == "dict_update":
            if not isinstance(existing, dict):
                return {"error": f"Variable '{key}' is not a dict"}
            updates = args.get("updates")
            if not isinstance(updates, dict):
                return {"error": "'updates' must be an object"}
            try:
                _ensure_json_serializable(updates)
                existing.update(updates)
            except Exception as e:
                return {"error": str(e)}
            return {"content": {"key": key, "value": existing}}

        if action == "dict_merge":
            if not isinstance(existing, dict):
                return {"error": f"Variable '{key}' is not a dict"}
            updates = args.get("updates")
            if not isinstance(updates, dict):
                return {"error": "'updates' must be an object"}
            try:
                _ensure_json_serializable(updates)
                # Deep merge
                def deep_merge(base: dict, update: dict) -> dict:
                    for k, v in update.items():
                        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                            deep_merge(base[k], v)
                        else:
                            base[k] = v
                    return base
                deep_merge(existing, updates)
            except Exception as e:
                return {"error": str(e)}
            return {"content": {"key": key, "value": existing}}

        if action == "dict_delete_key":
            if not isinstance(existing, dict):
                return {"error": f"Variable '{key}' is not a dict"}
            dict_key = args.get("dict_key")
            if not isinstance(dict_key, str):
                return {"error": "'dict_key' must be a string"}
            existed = dict_key in existing
            existing.pop(dict_key, None)
            return {"content": {"key": key, "value": existing, "deleted": existed}}

        if action == "dict_get_nested":
            if not isinstance(existing, dict):
                return {"error": f"Variable '{key}' is not a dict"}
            path = args.get("path")
            if not isinstance(path, list):
                return {"error": "'path' must be an array"}
            try:
                result = existing
                for p in path:
                    result = result[p]
                return {"content": {"key": key, "path": path, "value": result}}
            except (KeyError, TypeError) as e:
                return {"error": f"Path not found: {e}"}

        if action == "dict_set_nested":
            if not isinstance(existing, dict):
                return {"error": f"Variable '{key}' is not a dict"}
            path = args.get("path")
            if not isinstance(path, list) or not path:
                return {"error": "'path' must be a non-empty array"}
            value = args.get("value")
            try:
                _ensure_json_serializable(value)
                # Navigate to parent
                current = existing
                for p in path[:-1]:
                    if p not in current:
                        current[p] = {}
                    current = current[p]
                current[path[-1]] = value
            except Exception as e:
                return {"error": str(e)}
            return {"content": {"key": key, "value": existing}}

        # Number operations
        if action == "increment":
            if not isinstance(existing, (int, float)):
                return {"error": f"Variable '{key}' is not a number"}
            amount = args.get("amount", 1)
            if not isinstance(amount, (int, float)):
                return {"error": "'amount' must be a number"}
            store[key] = existing + amount
            return {"content": {"key": key, "value": store[key]}}

        if action == "decrement":
            if not isinstance(existing, (int, float)):
                return {"error": f"Variable '{key}' is not a number"}
            amount = args.get("amount", 1)
            if not isinstance(amount, (int, float)):
                return {"error": "'amount' must be a number"}
            store[key] = existing - amount
            return {"content": {"key": key, "value": store[key]}}

        if action == "multiply":
            if not isinstance(existing, (int, float)):
                return {"error": f"Variable '{key}' is not a number"}
            amount = args.get("amount")
            if not isinstance(amount, (int, float)):
                return {"error": "'amount' must be a number"}
            store[key] = existing * amount
            return {"content": {"key": key, "value": store[key]}}

        # Boolean operations
        if action == "toggle_bool":
            if not isinstance(existing, bool):
                return {"error": f"Variable '{key}' is not a boolean"}
            store[key] = not existing
            return {"content": {"key": key, "value": store[key]}}

        # String operations
        if action == "string_concat":
            if not isinstance(existing, str):
                return {"error": f"Variable '{key}' is not a string"}
            value = args.get("value")
            if not isinstance(value, str):
                return {"error": "'value' must be a string"}
            store[key] = existing + value
            return {"content": {"key": key, "value": store[key]}}

        if action == "string_replace":
            if not isinstance(existing, str):
                return {"error": f"Variable '{key}' is not a string"}
            old = args.get("old")
            new = args.get("new")
            if not isinstance(old, str):
                return {"error": "'old' must be a string"}
            if not isinstance(new, str):
                return {"error": "'new' must be a string"}
            store[key] = existing.replace(old, new)
            return {"content": {"key": key, "value": store[key]}}

        return {"error": f"Invalid action: {action}"}


@dataclass
class CreateVariableTool:
    """Tool for creating new variables/objects in memory."""
    
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "Create a new object or variable in memory with a given name and value."
        )

        self.api_base: str = "local"

        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the variable to create"
                },
                "value": {
                    "description": "The value to assign (can be any JSON type)",
                    "type": ["string", "number", "object", "array", "boolean", "null"]
                },
                "type": {
                    "type": "string",
                    "description": "Optional type hint for the variable",
                    "enum": ["string", "number", "object", "array", "boolean", "null"]
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether to overwrite if variable already exists (default: false)",
                    "default": False
                }
            },
            "required": ["name", "value"],
            "additionalProperties": False,
        }

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        name = args.get("name")
        value = args.get("value")
        overwrite = args.get("overwrite", False)
        type_hint = args.get("type")

        # Validate name
        if not isinstance(name, str) or not name:
            return {"error": "'name' must be a non-empty string"}

        # Validate value
        try:
            _ensure_json_serializable(value)
        except Exception as e:
            return {"error": f"Value must be JSON-serializable: {e}"}

        store = _get_session_store(session_id)

        # Check if variable already exists
        if name in store and not overwrite:
            return {
                "error": f"Variable '{name}' already exists. Set 'overwrite' to true to replace it.",
                "existing_value": store[name]
            }

        # Optional type validation
        if type_hint:
            actual_type = type(value).__name__
            if type_hint == "null" and value is not None:
                return {"error": f"Type hint 'null' specified but value is {actual_type}"}
            elif type_hint == "array" and not isinstance(value, list):
                return {"error": f"Type hint 'array' specified but value is {actual_type}"}
            elif type_hint == "object" and not isinstance(value, dict):
                return {"error": f"Type hint 'object' specified but value is {actual_type}"}
            elif type_hint == "string" and not isinstance(value, str):
                return {"error": f"Type hint 'string' specified but value is {actual_type}"}
            elif type_hint == "number" and not isinstance(value, (int, float)):
                return {"error": f"Type hint 'number' specified but value is {actual_type}"}
            elif type_hint == "boolean" and not isinstance(value, bool):
                return {"error": f"Type hint 'boolean' specified but value is {actual_type}"}

        # Create the variable
        store[name] = value

        # Determine the actual type
        if value is None:
            actual_type = "null"
        elif isinstance(value, bool):
            actual_type = "boolean"
        elif isinstance(value, (int, float)):
            actual_type = "number"
        elif isinstance(value, str):
            actual_type = "string"
        elif isinstance(value, list):
            actual_type = "array"
        elif isinstance(value, dict):
            actual_type = "object"
        else:
            actual_type = type(value).__name__

        return {
            "content": {
                "name": name,
                "value": value,
                "type": actual_type,
                "created": True,
                "overwritten": name in store and overwrite
            }
        }


@dataclass
class DebtAnalystTool:
    """AI agent that analyzes debt-related facts from blackboard and generates hypotheses."""
    
    config: Dict[str, Any]
    memory_tool: Optional['MemoryStoreTool'] = None
    mutate_tool: Optional['MemoryMutateTool'] = None

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "AI agent that reads facts from blackboard and generates debt analysis hypotheses."
        )

        self.api_base: str = "local"

        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {
            "type": "object",
            "properties": {
                "blackboard_key": {
                    "type": "string",
                    "description": "The key in memory store containing facts/data to analyze",
                    "default": "blackboard"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or specific question for the debt analyst"
                },
                "api_key": {
                    "type": "string",
                    "description": "OpenRouter API key"
                },
                "save_to": {
                    "type": "string",
                    "description": "Optional key to save results to"
                },
                "model": {
                    "type": "string",
                    "description": "Model to use",
                    "default": "deepseek/deepseek-r1-0528:free"
                },
                "timeout": {
                    "type": "number",
                    "description": "API request timeout in seconds (default: 25). Note: MCP client has a 30-second timeout.",
                    "default": 25
                }
            },
            "required": [],
            "additionalProperties": False,
        }

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        blackboard_key = args.get("blackboard_key")
        context = args.get("context", "")
        # Check both OPENROUTER_API_KEY and OPEN_ROUTER_API_KEY (with underscore)
        api_key = args.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        save_to = args.get("save_to")
        model = args.get("model", "deepseek/deepseek-r1-0528:free")
        timeout = args.get("timeout", 25)  # Default 25 seconds to fit within MCP client's 30-second timeout

        # Validate API key
        if not api_key:
            return {
                "error": "OpenRouter API key required. Provide via 'api_key' parameter or set OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY) environment variable."
            }

        # Validate tools are available
        if not self.memory_tool:
            return {"error": "MemoryStoreTool not configured for this analyst"}

        # Read all variables from memory using tool
        keys_result = self.memory_tool.handle(session_id, {"action": "keys"})
        if "error" in keys_result:
            return keys_result
        all_variables = keys_result.get("content", {}).get("keys", [])
        
        if not all_variables:
            return {
                "error": "No variables found in memory store. Please create variables first using 'create_variable' or 'memory_store' tools."
            }
        
        # Auto-discover blackboard if not specified
        if not blackboard_key:
            # Search for variables with 'facts' property
            candidates = []
            for var_name in all_variables:
                get_result = self.memory_tool.handle(session_id, {"action": "get", "key": var_name})
                if "error" not in get_result:
                    var_value = get_result.get("content", {}).get("value")
                    if isinstance(var_value, dict) and "facts" in var_value:
                        candidates.append(var_name)
            
            if candidates:
                blackboard_key = candidates[0]
                print(f"Auto-discovered blackboard variable: {blackboard_key}")
            else:
                # Look for any variable named 'blackboard' or containing 'blackboard'
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        blackboard_key = var_name
                        print(f"Found blackboard variable: {blackboard_key}")
                        break
                
                if not blackboard_key:
                    return {
                        "error": (
                            f"No blackboard variable found. Available variables: {all_variables}. "
                            "Please specify 'blackboard_key' parameter or create a variable with 'facts' property."
                        ),
                        "available_variables": all_variables,
                        "hint": "Create a blackboard object with facts property, or specify which variable to analyze"
                    }
        
        # Get blackboard object
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {
                "error": f"Blackboard key '{blackboard_key}' not found in memory store. Available keys: {all_variables}",
                "available_variables": all_variables
            }
        
        blackboard_obj = get_result.get("content", {}).get("value")
        if not blackboard_obj:
            return {
                "error": f"Blackboard key '{blackboard_key}' is empty or not found.",
                "available_variables": all_variables
            }
        
        # Check if blackboard is an object with 'facts' property
        if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj:
            facts = blackboard_obj["facts"]
            metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"}
        else:
            # If no 'facts' property, use the entire blackboard object as facts
            facts = blackboard_obj
            metadata = {}

        # Prepare the prompt for the AI
        prompt = self._build_prompt(facts, context, metadata, all_variables)

        # Call OpenRouter API
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/xendex-mcp-server",
                    "X-Title": "Xendex MCP Debt Analyst",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a debt analyst AI agent. Your role is to analyze financial facts and generate insightful hypotheses about debt situations, risks, opportunities, and recommendations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract the hypothesis from the response
            if "choices" in result and len(result["choices"]) > 0:
                hypothesis = result["choices"][0]["message"]["content"]
                
                # Create hypothesis object
                hypothesis_obj = {
                    "analyst": "debt_analyst",
                    "hypothesis": hypothesis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append hypothesis to blackboard.hypotheses if it exists
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj:
                    # Get current blackboard
                    current_bb = self.memory_tool.handle(session_id, {
                        "action": "get",
                        "key": blackboard_key
                    })
                    if "error" not in current_bb:
                        bb_value = current_bb.get("content", {}).get("value", {})
                        if isinstance(bb_value, dict) and "hypotheses" in bb_value:
                            if isinstance(bb_value["hypotheses"], list):
                                bb_value["hypotheses"].append(hypothesis_obj)
                                # Save updated blackboard
                                self.memory_tool.handle(session_id, {
                                    "action": "set",
                                    "key": blackboard_key,
                                    "value": bb_value
                                })
                                added_to_blackboard = True
                
                # Optionally save to separate variable
                if save_to:
                    self.memory_tool.handle(session_id, {
                        "action": "set",
                        "key": save_to,
                        "value": hypothesis_obj
                    })

                return {
                    "content": {
                        "hypothesis": hypothesis,
                        "facts_analyzed": facts,
                        "blackboard_key": blackboard_key,
                        "blackboard_metadata": metadata,
                        "all_variables": all_variables,
                        "model_used": model,
                        "saved_to": save_to if save_to else None,
                        "added_to_blackboard": added_to_blackboard,
                        "usage": result.get("usage", {})
                    }
                }
            else:
                return {"error": "No response generated from the model"}

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse API response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def _build_prompt(self, facts: Any, context: str, metadata: Dict[str, Any] = None, all_variables: List[str] = None) -> str:
        """Build the analysis prompt from facts and context."""
        prompt_parts = [
            "# Debt Analysis Task",
            "",
            "## Memory Store Context:"
        ]
        
        if all_variables:
            prompt_parts.append(f"Available variables in memory: {', '.join(all_variables)}")
            prompt_parts.append("")
        
        if metadata:
            prompt_parts.append("## Blackboard Metadata:")
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(metadata, indent=2))
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Facts from Blackboard:"
        ])

        # Format facts based on type
        if isinstance(facts, dict):
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(facts, indent=2))
            prompt_parts.append("```")
        elif isinstance(facts, list):
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(facts, indent=2))
            prompt_parts.append("```")
        else:
            prompt_parts.append(str(facts))

        if context:
            prompt_parts.extend([
                "",
                "## Additional Context:",
                context
            ])

        prompt_parts.extend([
            "",
            "## Task:",
            "Based on the facts above, generate comprehensive debt analysis hypotheses including:",
            "1. Key insights and patterns identified",
            "2. Risk assessment and potential concerns",
            "3. Opportunities for debt optimization or management",
            "4. Specific actionable recommendations",
            "5. Any assumptions or additional information needed",
            "",
            "Provide a structured, detailed analysis."
        ])

        return "\n".join(prompt_parts)


@dataclass
class LiquidityAnalystTool:
    """AI agent that analyzes liquidity positions and cash flow."""
    
    config: Dict[str, Any]
    memory_tool: Optional['MemoryStoreTool'] = None
    mutate_tool: Optional['MemoryMutateTool'] = None

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "AI agent that analyzes liquidity positions and cash flow."
        )

        self.api_base: str = "local"
        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {}

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        blackboard_key = args.get("blackboard_key")
        context = args.get("context", "")
        api_key = args.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        save_to = args.get("save_to")
        model = args.get("model", "deepseek/deepseek-r1-0528:free")
        timeout = args.get("timeout", 25)

        if not api_key:
            return {"error": "OpenRouter API key required. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY environment variable."}

        if not self.memory_tool:
            return {"error": "MemoryStoreTool not configured for this analyst"}

        keys_result = self.memory_tool.handle(session_id, {"action": "keys"})
        if "error" in keys_result:
            return keys_result
        all_variables = keys_result.get("content", {}).get("keys", [])
        
        if not all_variables:
            return {"error": "No variables found in memory store. Please create variables first."}
        
        # Auto-discover blackboard
        if not blackboard_key:
            candidates = []
            for k in all_variables:
                get_result = self.memory_tool.handle(session_id, {"action": "get", "key": k})
                if "error" not in get_result:
                    val = get_result.get("content", {}).get("value")
                    if isinstance(val, dict) and "facts" in val:
                        candidates.append(k)
            if candidates:
                blackboard_key = candidates[0]
            else:
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        blackboard_key = var_name
                        break
                if not blackboard_key:
                    return {
                        "error": f"No blackboard variable found. Available: {all_variables}",
                        "available_variables": all_variables
                    }
        
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {"error": f"Blackboard key '{blackboard_key}' not found.", "available_variables": all_variables}

        blackboard_obj = get_result.get("content", {}).get("value")
        facts = blackboard_obj.get("facts", blackboard_obj) if isinstance(blackboard_obj, dict) else blackboard_obj
        metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"} if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj else {}

        prompt = self._build_prompt(facts, context, metadata, all_variables)

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/xendex-mcp-server",
                    "X-Title": "Xendex MCP Liquidity Analyst",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a liquidity analyst AI agent. Analyze cash flow, working capital, current ratios, quick ratios, and liquidity positions. Identify liquidity risks and opportunities."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                analysis = result["choices"][0]["message"]["content"]
                
                hypothesis_obj = {
                    "analyst": "liquidity_analyst",
                    "analysis": analysis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append to blackboard hypotheses if available
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj:
                    current_bb = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
                    if "error" not in current_bb:
                        bb_value = current_bb.get("content", {}).get("value", {})
                        if isinstance(bb_value, dict) and "hypotheses" in bb_value and isinstance(bb_value["hypotheses"], list):
                            bb_value["hypotheses"].append(hypothesis_obj)
                            self.memory_tool.handle(session_id, {"action": "set", "key": blackboard_key, "value": bb_value})
                            added_to_blackboard = True
                
                if save_to:
                    self.memory_tool.handle(session_id, {
                        "action": "set",
                        "key": save_to,
                        "value": hypothesis_obj
                    })

                return {
                    "content": {
                        "analysis": analysis,
                        "facts_analyzed": facts,
                        "blackboard_key": blackboard_key,
                        "blackboard_metadata": metadata,
                        "model_used": model,
                        "saved_to": save_to,
                        "added_to_blackboard": added_to_blackboard,
                        "usage": result.get("usage", {})
                    }
                }
            return {"error": "No response generated from the model"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _build_prompt(self, facts: Any, context: str, metadata: Dict[str, Any], all_variables: List[str]) -> str:
        prompt_parts = ["# Liquidity Analysis Task\n"]
        if all_variables:
            prompt_parts.append(f"Available variables: {', '.join(all_variables)}\n")
        if metadata:
            prompt_parts.append(f"## Metadata:\n```json\n{json.dumps(metadata, indent=2)}\n```\n")
        prompt_parts.append(f"## Facts:\n```json\n{json.dumps(facts, indent=2) if isinstance(facts, (dict, list)) else str(facts)}\n```\n")
        if context:
            prompt_parts.append(f"## Context:\n{context}\n")
        prompt_parts.append(
            "\n## Task:\nAnalyze liquidity position including:\n"
            "1. Current and quick ratio analysis\n"
            "2. Cash flow assessment\n"
            "3. Working capital evaluation\n"
            "4. Liquidity risks and mitigation strategies\n"
            "5. Short-term vs long-term liquidity position"
        )
        return "".join(prompt_parts)


@dataclass
class QOEAnalystTool:
    """AI agent that performs Quality of Earnings analysis."""
    
    config: Dict[str, Any]
    memory_tool: Optional['MemoryStoreTool'] = None
    mutate_tool: Optional['MemoryMutateTool'] = None

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "AI agent that performs Quality of Earnings analysis."
        )

        self.api_base: str = "local"
        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {}

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        blackboard_key = args.get("blackboard_key")
        context = args.get("context", "")
        api_key = args.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        save_to = args.get("save_to")
        model = args.get("model", "deepseek/deepseek-r1-0528:free")
        timeout = args.get("timeout", 25)

        if not api_key:
            return {"error": "OpenRouter API key required. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY environment variable."}

        if not self.memory_tool:
            return {"error": "MemoryStoreTool not configured for this analyst"}

        keys_result = self.memory_tool.handle(session_id, {"action": "keys"})
        if "error" in keys_result:
            return keys_result
        all_variables = keys_result.get("content", {}).get("keys", [])
        
        if not all_variables:
            return {"error": "No variables found in memory store. Please create variables first."}
        
        # Auto-discover blackboard
        if not blackboard_key:
            candidates = []
            for k in all_variables:
                get_result = self.memory_tool.handle(session_id, {"action": "get", "key": k})
                if "error" not in get_result:
                    val = get_result.get("content", {}).get("value")
                    if isinstance(val, dict) and "facts" in val:
                        candidates.append(k)
            if candidates:
                blackboard_key = candidates[0]
            else:
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        blackboard_key = var_name
                        break
                if not blackboard_key:
                    return {
                        "error": f"No blackboard variable found. Available: {all_variables}",
                        "available_variables": all_variables
                    }
        
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {"error": f"Blackboard key '{blackboard_key}' not found.", "available_variables": all_variables}

        blackboard_obj = get_result.get("content", {}).get("value")
        facts = blackboard_obj.get("facts", blackboard_obj) if isinstance(blackboard_obj, dict) else blackboard_obj
        metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"} if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj else {}

        prompt = self._build_prompt(facts, context, metadata, all_variables)

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/xendex-mcp-server",
                    "X-Title": "Xendex MCP QoE Analyst",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a Quality of Earnings (QoE) analyst AI agent. Evaluate earnings quality, sustainability, one-time items, accounting policies, revenue recognition, and potential earnings manipulation red flags."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                analysis = result["choices"][0]["message"]["content"]
                
                hypothesis_obj = {
                    "analyst": "qoe_analyst",
                    "analysis": analysis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append to blackboard hypotheses if available
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj:
                    current_bb = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
                    if "error" not in current_bb:
                        bb_value = current_bb.get("content", {}).get("value", {})
                        if isinstance(bb_value, dict) and "hypotheses" in bb_value and isinstance(bb_value["hypotheses"], list):
                            bb_value["hypotheses"].append(hypothesis_obj)
                            self.memory_tool.handle(session_id, {"action": "set", "key": blackboard_key, "value": bb_value})
                            added_to_blackboard = True
                
                if save_to:
                    self.memory_tool.handle(session_id, {
                        "action": "set",
                        "key": save_to,
                        "value": hypothesis_obj
                    })

                return {
                    "content": {
                        "analysis": analysis,
                        "facts_analyzed": facts,
                        "blackboard_key": blackboard_key,
                        "blackboard_metadata": metadata,
                        "model_used": model,
                        "saved_to": save_to,
                        "added_to_blackboard": added_to_blackboard,
                        "usage": result.get("usage", {})
                    }
                }
            return {"error": "No response generated from the model"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _build_prompt(self, facts: Any, context: str, metadata: Dict[str, Any], all_variables: List[str]) -> str:
        prompt_parts = ["# Quality of Earnings (QoE) Analysis Task\n"]
        if all_variables:
            prompt_parts.append(f"Available variables: {', '.join(all_variables)}\n")
        if metadata:
            prompt_parts.append(f"## Metadata:\n```json\n{json.dumps(metadata, indent=2)}\n```\n")
        prompt_parts.append(f"## Facts:\n```json\n{json.dumps(facts, indent=2) if isinstance(facts, (dict, list)) else str(facts)}\n```\n")
        if context:
            prompt_parts.append(f"## Context:\n{context}\n")
        prompt_parts.append(
            "\n## Task:\nPerform Quality of Earnings analysis including:\n"
            "1. Earnings sustainability and quality assessment\n"
            "2. One-time items and adjustments identification\n"
            "3. Revenue recognition policy evaluation\n"
            "4. Cash vs accrual earnings comparison\n"
            "5. Red flags and earnings manipulation indicators\n"
            "6. Normalized earnings calculation"
        )
        return "".join(prompt_parts)


@dataclass
class AssetQualityAnalystTool:
    """AI agent that analyzes asset quality and credit risk."""
    
    config: Dict[str, Any]
    memory_tool: Optional['MemoryStoreTool'] = None
    mutate_tool: Optional['MemoryMutateTool'] = None

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "AI agent that analyzes asset quality and credit risk."
        )

        self.api_base: str = "local"
        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {}

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        blackboard_key = args.get("blackboard_key")
        context = args.get("context", "")
        api_key = args.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        save_to = args.get("save_to")
        model = args.get("model", "deepseek/deepseek-r1-0528:free")
        timeout = args.get("timeout", 25)

        if not api_key:
            return {"error": "OpenRouter API key required. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY environment variable."}

        if not self.memory_tool:
            return {"error": "MemoryStoreTool not configured for this analyst"}

        keys_result = self.memory_tool.handle(session_id, {"action": "keys"})
        if "error" in keys_result:
            return keys_result
        all_variables = keys_result.get("content", {}).get("keys", [])
        
        if not all_variables:
            return {"error": "No variables found in memory store. Please create variables first."}
        
        # Auto-discover blackboard
        if not blackboard_key:
            candidates = []
            for k in all_variables:
                get_result = self.memory_tool.handle(session_id, {"action": "get", "key": k})
                if "error" not in get_result:
                    val = get_result.get("content", {}).get("value")
                    if isinstance(val, dict) and "facts" in val:
                        candidates.append(k)
            if candidates:
                blackboard_key = candidates[0]
            else:
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        blackboard_key = var_name
                        break
                if not blackboard_key:
                    return {
                        "error": f"No blackboard variable found. Available: {all_variables}",
                        "available_variables": all_variables
                    }
        
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {"error": f"Blackboard key '{blackboard_key}' not found.", "available_variables": all_variables}

        blackboard_obj = get_result.get("content", {}).get("value")
        facts = blackboard_obj.get("facts", blackboard_obj) if isinstance(blackboard_obj, dict) else blackboard_obj
        metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"} if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj else {}

        prompt = self._build_prompt(facts, context, metadata, all_variables)

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/xendex-mcp-server",
                    "X-Title": "Xendex MCP Asset Quality Analyst",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an asset quality analyst AI agent. Assess credit risk, loan quality, investment portfolio health, non-performing assets, provisions, and asset impairment risks."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                analysis = result["choices"][0]["message"]["content"]
                
                hypothesis_obj = {
                    "analyst": "asset_quality_analyst",
                    "analysis": analysis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append to blackboard hypotheses if available
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj:
                    current_bb = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
                    if "error" not in current_bb:
                        bb_value = current_bb.get("content", {}).get("value", {})
                        if isinstance(bb_value, dict) and "hypotheses" in bb_value and isinstance(bb_value["hypotheses"], list):
                            bb_value["hypotheses"].append(hypothesis_obj)
                            self.memory_tool.handle(session_id, {"action": "set", "key": blackboard_key, "value": bb_value})
                            added_to_blackboard = True
                
                if save_to:
                    self.memory_tool.handle(session_id, {
                        "action": "set",
                        "key": save_to,
                        "value": hypothesis_obj
                    })

                return {
                    "content": {
                        "analysis": analysis,
                        "facts_analyzed": facts,
                        "blackboard_key": blackboard_key,
                        "blackboard_metadata": metadata,
                        "model_used": model,
                        "saved_to": save_to,
                        "added_to_blackboard": added_to_blackboard,
                        "usage": result.get("usage", {})
                    }
                }
            return {"error": "No response generated from the model"}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _build_prompt(self, facts: Any, context: str, metadata: Dict[str, Any], all_variables: List[str]) -> str:
        prompt_parts = ["# Asset Quality Analysis Task\n"]
        if all_variables:
            prompt_parts.append(f"Available variables: {', '.join(all_variables)}\n")
        if metadata:
            prompt_parts.append(f"## Metadata:\n```json\n{json.dumps(metadata, indent=2)}\n```\n")
        prompt_parts.append(f"## Facts:\n```json\n{json.dumps(facts, indent=2) if isinstance(facts, (dict, list)) else str(facts)}\n```\n")
        if context:
            prompt_parts.append(f"## Context:\n{context}\n")
        prompt_parts.append(
            "\n## Task:\nPerform asset quality analysis including:\n"
            "1. Credit risk assessment and ratings\n"
            "2. Non-performing asset (NPA) evaluation\n"
            "3. Loan portfolio quality and diversification\n"
            "4. Provision adequacy and impairment analysis\n"
            "5. Concentration risk identification\n"
            "6. Recovery rates and loss given default (LGD) estimates"
        )
        return "".join(prompt_parts)
