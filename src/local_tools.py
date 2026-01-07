from __future__ import annotations

import json
import os
import re
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

# Session-scoped in-memory store.
# Persists for the lifetime of the Python process.
_SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def _extract_content_from_response(content: str) -> str:
    """Extract actual analysis content from AI response, handling DeepSeek R1 reasoning tags."""
    if not content:
        return ""
    
    # DeepSeek R1 wraps reasoning in <think> tags
    # Case 1: Content is after </think> tag (preferred format)
    if "</think>" in content:
        parts = content.split("</think>")
        if len(parts) > 1:
            actual_content = parts[-1].strip()
            if actual_content:
                return actual_content
        
        # Case 2: If nothing after </think>, extract from inside <think>...</think>
        # This happens when the model puts everything in the thinking block
        if "<think>" in content:
            # Use regex to get content between first <think> and last </think>
            import re
            pattern = r'<think>(.*?)</think>'
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                # Get the last match (in case of multiple think blocks)
                inner_content = matches[-1].strip()
                if inner_content:
                    print(f"DEBUG: Extracting content from inside <think> tags (length: {len(inner_content)})")
                    return inner_content
    
    # Case 3: No think tags, or extraction failed - return everything
    # This is a fallback to ensure we never lose content
    result = content.strip()
    if result:
        print(f"DEBUG: No think tags found or extraction failed, returning raw content (length: {len(result)})")
    return result


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
                        "nested_list_append",
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

        # Nested list operations
        if action == "nested_list_append":
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
                # Get the list at the final path
                list_key = path[-1]
                if list_key not in current:
                    current[list_key] = []
                if not isinstance(current[list_key], list):
                    return {"error": f"Path '{'.'.join(path)}' does not point to a list"}
                current[list_key].append(value)
            except Exception as e:
                return {"error": str(e)}
            return {"content": {"key": key, "value": existing, "path": path, "appended": True}}

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
    critic_tool: Optional['CriticAgentTool'] = None  # Reference to critic agent

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
        enable_self_critique = args.get("enable_self_critique", False)
        max_iterations = args.get("max_iterations", 2)

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
        
        # Check if explicitly provided blackboard_key exists and has data
        original_blackboard_key = blackboard_key
        if blackboard_key:
            get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
            if "error" in get_result or get_result.get("content", {}).get("value") is None:
                # Explicitly provided key doesn't exist or is None, fall back to auto-discovery
                print(f"Warning: Blackboard key '{blackboard_key}' not found or is None. Attempting auto-discovery...")
                blackboard_key = None
        
        # Auto-discover blackboard if not specified or if explicit key failed
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
                        # Verify this variable has data
                        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": var_name})
                        if "error" not in get_result:
                            var_value = get_result.get("content", {}).get("value")
                            if var_value is not None:
                                blackboard_key = var_name
                                print(f"Found blackboard variable: {blackboard_key}")
                                break
                
                if not blackboard_key:
                    error_msg = f"No blackboard variable found. Available variables: {all_variables}."
                    if original_blackboard_key:
                        error_msg = f"Blackboard key '{original_blackboard_key}' not found or is None. " + error_msg
                    return {
                        "error": error_msg + " Please specify 'blackboard_key' parameter or create a variable with 'facts' property.",
                        "available_variables": all_variables,
                        "hint": "Create a blackboard object with facts property, or specify which variable to analyze"
                    }
        
        # Get blackboard object (we know it exists now)
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {
                "error": f"Blackboard key '{blackboard_key}' not found in memory store. Available keys: {all_variables}",
                "available_variables": all_variables
            }
        
        blackboard_obj = get_result.get("content", {}).get("value")
        if blackboard_obj is None:
            return {
                "error": f"Blackboard key '{blackboard_key}' is None. Please ensure the blackboard contains data.",
                "available_variables": all_variables
            }
        
        # Check if blackboard is an object with 'facts' property
        if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj:
            facts = blackboard_obj["facts"]
            if facts is None:
                return {
                    "error": f"Blackboard '{blackboard_key}' facts property is None. Please ensure facts contains data.",
                    "available_variables": all_variables
                }
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
                            "content": "You are a debt analyst AI agent. Your role is to analyze financial facts and generate insightful hypotheses about debt situations, risks, opportunities, and recommendations.\n\nIMPORTANT: After your reasoning process, provide your complete final analysis. Put your analysis in the main response, not just in thinking tags."
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
                choice = result["choices"][0]
                message = choice.get("message", {})
                raw_content = message.get("content", "")
                
                # OpenRouter separates reasoning from content for DeepSeek R1
                # If content is empty, check the reasoning field
                if not raw_content:
                    reasoning = message.get("reasoning", "")
                    if reasoning:
                        print(f"DEBUG debt: Content empty, using reasoning field (length: {len(reasoning)})")
                        raw_content = reasoning
                
                hypothesis = _extract_content_from_response(raw_content)
                
                # CRITICAL FIX: If extraction returns empty but we have content, use raw
                if not hypothesis and raw_content:
                    print(f"CRITICAL: Extraction failed for debt analyst, using raw content")
                    print(f"Raw content length: {len(raw_content)}")
                    print(f"First 1000 chars: {raw_content[:1000]}")
                    hypothesis = raw_content
                elif not hypothesis:
                    print(f"ERROR: Empty hypothesis AND empty raw content from debt analyst")
                
                # Enable self-critique and refinement if requested
                critique_history = []
                if enable_self_critique and self.critic_tool and max_iterations > 0:
                    for iteration in range(max_iterations):
                        # Call critic agent to evaluate the hypothesis
                        critique_result = self.critic_tool.handle(session_id, {
                            "response_text": hypothesis,
                            "agent_type": "debt_analyst",
                            "criteria": ["completeness", "accuracy", "depth", "actionability", "evidence"],
                            "context": context,
                            "api_key": api_key,
                            "model": model,
                            "timeout": timeout
                        })
                        
                        if "error" not in critique_result:
                            critique = critique_result.get("content", {}).get("critique", "")
                            critique_history.append({
                                "iteration": iteration + 1,
                                "critique": critique
                            })
                            
                            # Refine the hypothesis based on critique
                            refinement_prompt = self._build_refinement_prompt(
                                facts, context, metadata, all_variables, hypothesis, critique
                            )
                            
                            try:
                                refinement_response = requests.post(
                                    url="https://openrouter.ai/api/v1/chat/completions",
                                    headers={
                                        "Authorization": f"Bearer {api_key}",
                                        "Content-Type": "application/json",
                                        "HTTP-Referer": "https://github.com/xendex-mcp-server",
                                        "X-Title": "Xendex MCP Debt Analyst Refinement",
                                    },
                                    json={
                                        "model": model,
                                        "messages": [
                                            {
                                                "role": "system",
                                                "content": "You are a debt analyst AI agent. Refine your previous analysis based on critic feedback to improve quality and completeness."
                                            },
                                            {"role": "user", "content": refinement_prompt}
                                        ]
                                    },
                                    timeout=timeout
                                )
                                refinement_response.raise_for_status()
                                refinement_result = refinement_response.json()
                                
                                if "choices" in refinement_result and len(refinement_result["choices"]) > 0:
                                    refined_raw = refinement_result["choices"][0]["message"]["content"]
                                    hypothesis = _extract_content_from_response(refined_raw)
                                    critique_history[-1]["refined_hypothesis"] = hypothesis
                                else:
                                    break  # Could not refine, use current hypothesis
                            except Exception as e:
                                critique_history[-1]["refinement_error"] = str(e)
                                break  # Error during refinement, use current hypothesis
                        else:
                            break  # Critic failed, use current hypothesis
                
                # Create hypothesis object
                hypothesis_obj = {
                    "analyst": "debt_analyst",
                    "hypothesis": hypothesis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {}),
                    "self_critique_enabled": enable_self_critique,
                    "critique_iterations": len(critique_history) if enable_self_critique else 0,
                    "critique_history": critique_history if critique_history else None
                }
                
                # Append hypothesis to blackboard.hypotheses using tool call
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj and self.mutate_tool:
                    append_result = self.mutate_tool.handle(session_id, {
                        "action": "nested_list_append",
                        "key": blackboard_key,
                        "path": ["hypotheses"],
                        "value": hypothesis_obj
                    })
                    if "error" not in append_result:
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
                        "self_critique_enabled": enable_self_critique,
                        "critique_iterations": len(critique_history) if enable_self_critique else 0,
                        "critique_history": critique_history if critique_history else None,
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
            try:
                prompt_parts.append(json.dumps(metadata, indent=2, default=str))
            except Exception as e:
                prompt_parts.append(f"Error serializing metadata: {str(e)}")
                prompt_parts.append(str(metadata))
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Financial Data / Facts:",
            ""
        ])

        # Format facts based on type with better error handling
        try:
            if isinstance(facts, (dict, list)):
                prompt_parts.append("```json")
                prompt_parts.append(json.dumps(facts, indent=2, default=str, ensure_ascii=False))
                prompt_parts.append("```")
            elif isinstance(facts, str):
                prompt_parts.append(facts)
            else:
                prompt_parts.append(f"Data Type: {type(facts).__name__}")
                prompt_parts.append(str(facts))
        except Exception as e:
            prompt_parts.append(f"[Error formatting facts: {str(e)}]")
            prompt_parts.append(f"Raw facts (type: {type(facts).__name__}): {str(facts)[:1000]}")

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

    def _build_refinement_prompt(self, facts: Any, context: str, metadata: Dict[str, Any], all_variables: List[str], 
                                  previous_hypothesis: str, critique: str) -> str:
        """Build a prompt for refining the hypothesis based on critique."""
        prompt_parts = [
            "# Hypothesis Refinement Task",
            "",
            "## Original Analysis Context:",
            "### Memory Store Context:"
        ]
        
        if all_variables:
            prompt_parts.append(f"Available variables: {', '.join(all_variables)}")
            prompt_parts.append("")
        
        if metadata:
            prompt_parts.append("### Blackboard Metadata:")
            prompt_parts.append("```json")
            try:
                prompt_parts.append(json.dumps(metadata, indent=2, default=str))
            except Exception:
                prompt_parts.append(str(metadata))
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "### Financial Data / Facts:",
            ""
        ])
        
        try:
            if isinstance(facts, (dict, list)):
                prompt_parts.append("```json")
                prompt_parts.append(json.dumps(facts, indent=2, default=str, ensure_ascii=False))
                prompt_parts.append("```")
            else:
                prompt_parts.append(str(facts))
        except Exception:
            prompt_parts.append(str(facts)[:1000])
        
        if context:
            prompt_parts.extend([
                "",
                "### Additional Context:",
                context
            ])
        
        prompt_parts.extend([
            "",
            "## Your Previous Hypothesis:",
            "```",
            previous_hypothesis,
            "```",
            "",
            "## Critic's Feedback:",
            "```",
            critique,
            "```",
            "",
            "## Task:",
            "Refine your debt analysis hypothesis by addressing the critic's feedback. Maintain the strengths identified while improving weaknesses, filling gaps, and strengthening your evidence and recommendations. Provide a complete, revised analysis that incorporates the constructive feedback.",
            ""
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
        
        # Check if explicitly provided blackboard_key exists and has data
        original_blackboard_key = blackboard_key
        if blackboard_key:
            get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
            if "error" in get_result or get_result.get("content", {}).get("value") is None:
                print(f"Warning: Blackboard key '{blackboard_key}' not found or is None. Attempting auto-discovery...")
                blackboard_key = None
        
        # Auto-discover blackboard if not specified or if explicit key failed
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
                print(f"Auto-discovered blackboard variable: {blackboard_key}")
            else:
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": var_name})
                        if "error" not in get_result and get_result.get("content", {}).get("value") is not None:
                            blackboard_key = var_name
                            print(f"Found blackboard variable: {blackboard_key}")
                            break
                if not blackboard_key:
                    error_msg = f"No blackboard variable found. Available: {all_variables}"
                    if original_blackboard_key:
                        error_msg = f"Blackboard key '{original_blackboard_key}' not found or is None. " + error_msg
                    return {
                        "error": error_msg,
                        "available_variables": all_variables
                    }
        
        # Get blackboard object (we know it exists now)
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {"error": f"Blackboard key '{blackboard_key}' not found.", "available_variables": all_variables}

        blackboard_obj = get_result.get("content", {}).get("value")
        if blackboard_obj is None:
            return {
                "error": f"Blackboard key '{blackboard_key}' is None. Please ensure the blackboard contains data.",
                "available_variables": all_variables
            }
        
        # Extract facts and metadata
        if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj:
            facts = blackboard_obj["facts"]
            if facts is None:
                return {
                    "error": f"Blackboard '{blackboard_key}' facts property is None. Please ensure facts contains data.",
                    "available_variables": all_variables
                }
            metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"}
        else:
            facts = blackboard_obj
            metadata = {}

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
                            "content": "You are a liquidity analyst AI agent. Analyze cash flow, working capital, current ratios, quick ratios, and liquidity positions. Identify liquidity risks and opportunities.\n\nIMPORTANT: After your reasoning process, provide your complete final analysis. Put your analysis in the main response, not just in thinking tags."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                # Debug: Log the entire choice structure
                choice = result["choices"][0]
                print(f"DEBUG liquidity: Full choice keys: {choice.keys()}")
                message = choice.get("message", {})
                print(f"DEBUG liquidity: Message keys: {message.keys()}")
                
                raw_content = message.get("content", "")
                print(f"DEBUG liquidity: Raw content type: {type(raw_content)}, length: {len(raw_content) if raw_content else 0}")
                
                # OpenRouter separates reasoning from content for DeepSeek R1
                # If content is empty, check the reasoning field
                if not raw_content:
                    reasoning = message.get("reasoning", "")
                    if reasoning:
                        print(f"DEBUG liquidity: Content empty, using reasoning field (length: {len(reasoning)})")
                        raw_content = reasoning
                
                analysis = _extract_content_from_response(raw_content)
                
                # CRITICAL FIX: If extraction returns empty but we have content, use raw
                if not analysis and raw_content:
                    print(f"CRITICAL: Extraction failed for liquidity analyst, using raw content")
                    print(f"Raw content length: {len(raw_content)}")
                    print(f"First 1000 chars: {raw_content[:1000]}")
                    # Use the raw content as-is (includes thinking tags, but better than nothing)
                    analysis = raw_content
                elif not analysis:
                    print(f"ERROR: Empty analysis AND empty raw content from liquidity analyst")
                
                hypothesis_obj = {
                    "analyst": "liquidity_analyst",
                    "analysis": analysis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append to blackboard hypotheses using tool call
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj and self.mutate_tool:
                    append_result = self.mutate_tool.handle(session_id, {
                        "action": "nested_list_append",
                        "key": blackboard_key,
                        "path": ["hypotheses"],
                        "value": hypothesis_obj
                    })
                    if "error" not in append_result:
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
        """Build the analysis prompt from facts and context."""
        prompt_parts = [
            "# Liquidity Analysis Task",
            "",
            "## Memory Store Context:"
        ]
        
        if all_variables:
            prompt_parts.append(f"Available variables in memory: {', '.join(all_variables)}")
            prompt_parts.append("")
        
        if metadata:
            prompt_parts.append("## Blackboard Metadata:")
            prompt_parts.append("```json")
            try:
                prompt_parts.append(json.dumps(metadata, indent=2, default=str))
            except Exception as e:
                prompt_parts.append(f"Error serializing metadata: {str(e)}")
                prompt_parts.append(str(metadata))
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Financial Data / Facts:",
            ""
        ])

        # Format facts based on type with better error handling
        try:
            if isinstance(facts, (dict, list)):
                prompt_parts.append("```json")
                prompt_parts.append(json.dumps(facts, indent=2, default=str, ensure_ascii=False))
                prompt_parts.append("```")
            elif isinstance(facts, str):
                prompt_parts.append(facts)
            else:
                prompt_parts.append(f"Data Type: {type(facts).__name__}")
                prompt_parts.append(str(facts))
        except Exception as e:
            prompt_parts.append(f"[Error formatting facts: {str(e)}]")
            prompt_parts.append(f"Raw facts (type: {type(facts).__name__}): {str(facts)[:1000]}")

        if context:
            prompt_parts.extend([
                "",
                "## Additional Context:",
                context
            ])

        prompt_parts.extend([
            "",
            "## Task:",
            "Based on the facts above, analyze liquidity position including:",
            "1. Current and quick ratio analysis",
            "2. Cash flow assessment and trends",
            "3. Working capital evaluation",
            "4. Liquidity risks and mitigation strategies",
            "5. Short-term vs long-term liquidity position",
            "6. Any assumptions or additional information needed",
            "",
            "Provide a structured, detailed analysis."
        ])

        return "\n".join(prompt_parts)


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
        
        # Check if explicitly provided blackboard_key exists and has data
        original_blackboard_key = blackboard_key
        if blackboard_key:
            get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
            if "error" in get_result or get_result.get("content", {}).get("value") is None:
                print(f"Warning: Blackboard key '{blackboard_key}' not found or is None. Attempting auto-discovery...")
                blackboard_key = None
        
        # Auto-discover blackboard if not specified or if explicit key failed
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
                print(f"Auto-discovered blackboard variable: {blackboard_key}")
            else:
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": var_name})
                        if "error" not in get_result and get_result.get("content", {}).get("value") is not None:
                            print(f"Found blackboard variable: {blackboard_key}")
                            break
                if not blackboard_key:
                    error_msg = f"No blackboard variable found. Available: {all_variables}"
                    if original_blackboard_key:
                        error_msg = f"Blackboard key '{original_blackboard_key}' not found or is None. " + error_msg
                    return {
                        "error": error_msg,
                        "available_variables": all_variables
                    }
        
        # Get blackboard object (we know it exists now)
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {"error": f"Blackboard key '{blackboard_key}' not found.", "available_variables": all_variables}

        blackboard_obj = get_result.get("content", {}).get("value")
        if blackboard_obj is None:
            return {
                "error": f"Blackboard key '{blackboard_key}' is None. Please ensure the blackboard contains data.",
                "available_variables": all_variables
            }
        
        # Extract facts and metadata
        if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj:
            facts = blackboard_obj["facts"]
            if facts is None:
                return {
                    "error": f"Blackboard '{blackboard_key}' facts property is None. Please ensure facts contains data.",
                    "available_variables": all_variables
                }
            metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"}
        else:
            facts = blackboard_obj
            metadata = {}

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
                            "content": "You are a Quality of Earnings (QoE) analyst AI agent. Evaluate earnings quality, sustainability, one-time items, accounting policies, revenue recognition, and potential earnings manipulation red flags.\n\nIMPORTANT: After your reasoning process, provide your complete final analysis. Put your analysis in the main response, not just in thinking tags."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                raw_content = result["choices"][0]["message"]["content"]
                analysis = _extract_content_from_response(raw_content)
                
                # Debug: Log if analysis is empty
                if not analysis:
                    print(f"WARNING: Empty analysis extracted from QoE analyst response")
                    print(f"Raw content length: {len(raw_content)}")
                
                hypothesis_obj = {
                    "analyst": "qoe_analyst",
                    "analysis": analysis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append to blackboard hypotheses using tool call
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj and self.mutate_tool:
                    append_result = self.mutate_tool.handle(session_id, {
                        "action": "nested_list_append",
                        "key": blackboard_key,
                        "path": ["hypotheses"],
                        "value": hypothesis_obj
                    })
                    if "error" not in append_result:
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
        """Build the analysis prompt from facts and context."""
        prompt_parts = [
            "# Quality of Earnings (QoE) Analysis Task",
            "",
            "## Memory Store Context:"
        ]
        
        if all_variables:
            prompt_parts.append(f"Available variables in memory: {', '.join(all_variables)}")
            prompt_parts.append("")
        
        if metadata:
            prompt_parts.append("## Blackboard Metadata:")
            prompt_parts.append("```json")
            try:
                prompt_parts.append(json.dumps(metadata, indent=2, default=str))
            except Exception as e:
                prompt_parts.append(f"Error serializing metadata: {str(e)}")
                prompt_parts.append(str(metadata))
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Financial Data / Facts:",
            ""
        ])

        # Format facts based on type with better error handling
        try:
            if isinstance(facts, (dict, list)):
                prompt_parts.append("```json")
                prompt_parts.append(json.dumps(facts, indent=2, default=str, ensure_ascii=False))
                prompt_parts.append("```")
            elif isinstance(facts, str):
                prompt_parts.append(facts)
            else:
                prompt_parts.append(f"Data Type: {type(facts).__name__}")
                prompt_parts.append(str(facts))
        except Exception as e:
            prompt_parts.append(f"[Error formatting facts: {str(e)}]")
            prompt_parts.append(f"Raw facts (type: {type(facts).__name__}): {str(facts)[:1000]}")

        if context:
            prompt_parts.extend([
                "",
                "## Additional Context:",
                context
            ])

        prompt_parts.extend([
            "",
            "## Task:",
            "Based on the facts above, perform Quality of Earnings analysis including:",
            "1. Earnings sustainability and quality assessment",
            "2. One-time items and adjustments identification",
            "3. Revenue recognition policy evaluation",
            "4. Cash vs accrual earnings comparison",
            "5. Red flags and earnings manipulation indicators",
            "6. Normalized earnings calculation",
            "7. Any assumptions or additional information needed",
            "",
            "Provide a structured, detailed analysis."
        ])

        return "\n".join(prompt_parts)


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
        
        # Check if explicitly provided blackboard_key exists and has data
        original_blackboard_key = blackboard_key
        if blackboard_key:
            get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
            if "error" in get_result or get_result.get("content", {}).get("value") is None:
                print(f"Warning: Blackboard key '{blackboard_key}' not found or is None. Attempting auto-discovery...")
                blackboard_key = None
        
        # Auto-discover blackboard if not specified or if explicit key failed
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
                print(f"Auto-discovered blackboard variable: {blackboard_key}")
            else:
                for var_name in all_variables:
                    if "blackboard" in var_name.lower():
                        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": var_name})
                        if "error" not in get_result and get_result.get("content", {}).get("value") is not None:
                            blackboard_key = var_name
                            print(f"Found blackboard variable: {blackboard_key}")
                            break
                if not blackboard_key:
                    error_msg = f"No blackboard variable found. Available: {all_variables}"
                    if original_blackboard_key:
                        error_msg = f"Blackboard key '{original_blackboard_key}' not found or is None. " + error_msg
                    return {
                        "error": error_msg,
                        "available_variables": all_variables
                    }
        
        get_result = self.memory_tool.handle(session_id, {"action": "get", "key": blackboard_key})
        if "error" in get_result:
            return {"error": f"Blackboard key '{blackboard_key}' not found.", "available_variables": all_variables}

        blackboard_obj = get_result.get("content", {}).get("value")
        if blackboard_obj is None:
            return {
                "error": f"Blackboard key '{blackboard_key}' is None. Please ensure the blackboard contains data.",
                "available_variables": all_variables
            }
        
        # Extract facts and metadata
        if isinstance(blackboard_obj, dict) and "facts" in blackboard_obj:
            facts = blackboard_obj["facts"]
            if facts is None:
                return {
                    "error": f"Blackboard '{blackboard_key}' facts property is None. Please ensure facts contains data.",
                    "available_variables": all_variables
                }
            metadata = {k: v for k, v in blackboard_obj.items() if k != "facts"}
        else:
            facts = blackboard_obj
            metadata = {}

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
                            "content": "You are an asset quality analyst AI agent. Assess credit risk, loan quality, investment portfolio health, non-performing assets, provisions, and asset impairment risks.\n\nIMPORTANT: After your reasoning process, provide your complete final analysis. Put your analysis in the main response, not just in thinking tags."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                raw_content = result["choices"][0]["message"]["content"]
                analysis = _extract_content_from_response(raw_content)
                
                # Debug: Log if analysis is empty
                if not analysis:
                    print(f"WARNING: Empty analysis extracted from asset quality analyst response")
                    print(f"Raw content length: {len(raw_content)}")
                
                hypothesis_obj = {
                    "analyst": "asset_quality_analyst",
                    "analysis": analysis,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Append to blackboard hypotheses using tool call
                added_to_blackboard = False
                if isinstance(blackboard_obj, dict) and "hypotheses" in blackboard_obj and self.mutate_tool:
                    append_result = self.mutate_tool.handle(session_id, {
                        "action": "nested_list_append",
                        "key": blackboard_key,
                        "path": ["hypotheses"],
                        "value": hypothesis_obj
                    })
                    if "error" not in append_result:
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
        """Build the analysis prompt from facts and context."""
        prompt_parts = [
            "# Asset Quality Analysis Task",
            "",
            "## Memory Store Context:"
        ]
        
        if all_variables:
            prompt_parts.append(f"Available variables in memory: {', '.join(all_variables)}")
            prompt_parts.append("")
        
        if metadata:
            prompt_parts.append("## Blackboard Metadata:")
            prompt_parts.append("```json")
            try:
                prompt_parts.append(json.dumps(metadata, indent=2, default=str))
            except Exception as e:
                prompt_parts.append(f"Error serializing metadata: {str(e)}")
                prompt_parts.append(str(metadata))
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## Financial Data / Facts:",
            ""
        ])

        # Format facts based on type with better error handling
        try:
            if isinstance(facts, (dict, list)):
                prompt_parts.append("```json")
                prompt_parts.append(json.dumps(facts, indent=2, default=str, ensure_ascii=False))
                prompt_parts.append("```")
            elif isinstance(facts, str):
                prompt_parts.append(facts)
            else:
                prompt_parts.append(f"Data Type: {type(facts).__name__}")
                prompt_parts.append(str(facts))
        except Exception as e:
            prompt_parts.append(f"[Error formatting facts: {str(e)}]")
            prompt_parts.append(f"Raw facts (type: {type(facts).__name__}): {str(facts)[:1000]}")

        if context:
            prompt_parts.extend([
                "",
                "## Additional Context:",
                context
            ])

        prompt_parts.extend([
            "",
            "## Task:",
            "Based on the facts above, perform asset quality analysis including:",
            "1. Credit risk assessment and ratings",
            "2. Non-performing asset (NPA) evaluation",
            "3. Loan portfolio quality and diversification",
            "4. Provision adequacy and impairment analysis",
            "5. Concentration risk identification",
            "6. Recovery rates and loss given default (LGD) estimates",
            "7. Any assumptions or additional information needed",
            "",
            "Provide a structured, detailed analysis."
        ])

        return "\n".join(prompt_parts)


@dataclass
class ConsolidatedCriticTool:
    """Batch critique multiple analyst responses for cost-efficient cross-analysis."""
    
    config: Dict[str, Any]
    memory_tool: Optional['MemoryStoreTool'] = None

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "Batch critique multiple analyst responses together for cost efficiency."
        )

        self.api_base: str = "local"
        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {}

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        analyst_responses = args.get("analyst_responses", [])
        critique_mode = args.get("critique_mode", "cross_analysis")
        use_cheap_model = args.get("use_cheap_model", True)
        context = args.get("context", "")
        api_key = args.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        save_to = args.get("save_to")
        timeout = args.get("timeout", 60)
        
        # Model selection based on cost preference
        if args.get("model"):
            model = args["model"]
        else:
            model = "deepseek/deepseek-r1-0528:free"

        if not api_key:
            return {"error": "OpenRouter API key required. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY environment variable."}

        if not analyst_responses or len(analyst_responses) == 0:
            return {"error": "No analyst responses provided. Please provide at least one analyst response."}

        # Collect all analyst responses
        collected_responses = []
        for resp_obj in analyst_responses:
            agent_type = resp_obj.get("agent", "unknown")
            response_text = resp_obj.get("response")
            response_key = resp_obj.get("response_key")
            
            # Try to get response from memory if key provided
            if not response_text and response_key and self.memory_tool:
                get_result = self.memory_tool.handle(session_id, {"action": "get", "key": response_key})
                if "error" not in get_result:
                    response_value = get_result.get("content", {}).get("value")
                    if isinstance(response_value, dict):
                        response_text = response_value.get("analysis") or response_value.get("hypothesis") or str(response_value)
                    else:
                        response_text = str(response_value)
            
            if response_text:
                collected_responses.append({
                    "agent": agent_type,
                    "response": response_text
                })

        if not collected_responses:
            return {"error": "Could not retrieve any analyst responses. Check response_key values or provide response text."}

        # Build consolidated critique prompt
        prompt = self._build_consolidated_prompt(collected_responses, critique_mode, context)

        # Call OpenRouter API
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/xendex-mcp-server",
                    "X-Title": "Xendex MCP Consolidated Critic",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a senior financial analyst reviewing multiple analyst reports. Your role is to identify conflicts, inconsistencies, gaps, and provide consolidated feedback across all analyses. Focus on cross-validation and synthesis."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})
                raw_content = message.get("content", "")
                
                # OpenRouter separates reasoning from content for DeepSeek R1
                # If content is empty, check the reasoning field
                if not raw_content:
                    reasoning = message.get("reasoning", "")
                    if reasoning:
                        print(f"DEBUG consolidated_critic: Content empty, using reasoning field (length: {len(reasoning)})")
                        raw_content = reasoning
                
                critique = _extract_content_from_response(raw_content)
                
                # Debug: Log if critique is empty
                if not critique:
                    print(f"WARNING: Empty critique extracted from consolidated critic response")
                    print(f"Raw content length: {len(raw_content)}")
                
                critique_obj = {
                    "consolidated_critic": "consolidated_analyst_critique",
                    "critique": critique,
                    "critique_mode": critique_mode,
                    "num_analysts_reviewed": len(collected_responses),
                    "analysts": [r["agent"] for r in collected_responses],
                    "model": model,
                    "cost_efficient": use_cheap_model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Save the critique if requested
                if save_to and self.memory_tool:
                    self.memory_tool.handle(session_id, {
                        "action": "set",
                        "key": save_to,
                        "value": critique_obj
                    })

                return {
                    "content": {
                        "critique": critique,
                        "critique_mode": critique_mode,
                        "num_analysts_reviewed": len(collected_responses),
                        "analysts_reviewed": [r["agent"] for r in collected_responses],
                        "model_used": model,
                        "cost_efficient": use_cheap_model,
                        "saved_to": save_to if save_to else None,
                        "usage": result.get("usage", {}),
                        "estimated_cost_savings": f"{len(collected_responses)}x vs individual critiques" if use_cheap_model else "N/A"
                    }
                }
            return {"error": "No critique generated from the model"}

        except Exception as e:
            return {"error": f"Consolidated critique failed: {str(e)}"}

    def _build_consolidated_prompt(self, responses: List[Dict[str, str]], mode: str, context: str) -> str:
        """Build a consolidated critique prompt for multiple analyst responses."""
        prompt_parts = [
            "# Consolidated Multi-Analyst Critique Task",
            "",
            f"## Critique Mode: {mode.replace('_', ' ').title()}",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "## Company/Scenario Context:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "## Analyst Reports to Review:",
            ""
        ])
        
        for i, resp in enumerate(responses, 1):
            agent_name = resp["agent"].replace("_", " ").title()
            prompt_parts.extend([
                f"### {i}. {agent_name} Report:",
                "```",
                resp["response"],
                "```",
                ""
            ])
        
        # Mode-specific instructions
        if mode == "cross_analysis":
            prompt_parts.extend([
                "## Task: Cross-Analysis Critique",
                "Analyze all reports together and provide:",
                "",
                "1. **Consistency Check**: Do the analysts agree on key metrics and conclusions?",
                "2. **Contradiction Detection**: Identify any conflicting assessments or recommendations",
                "3. **Gaps Analysis**: What critical aspects are missing across all reports?",
                "4. **Synthesis Opportunities**: Where can insights be combined for stronger conclusions?",
                "5. **Priority Issues**: Rank the most critical findings across all analyses",
                "6. **Unified Recommendations**: Consolidated action items based on all inputs",
                ""
            ])
        elif mode == "conflicts_only":
            prompt_parts.extend([
                "## Task: Conflict Identification",
                "Focus ONLY on identifying contradictions and inconsistencies:",
                "",
                "1. **Direct Contradictions**: Where do analysts disagree on facts or assessments?",
                "2. **Implicit Conflicts**: Recommendations that contradict each other",
                "3. **Severity Rating**: How critical is each conflict?",
                "4. **Resolution Guidance**: How should conflicts be resolved?",
                ""
            ])
        else:  # comprehensive
            prompt_parts.extend([
                "## Task: Comprehensive Review",
                "Provide a thorough critique covering:",
                "",
                "1. **Individual Report Quality**: Assess each analyst's work",
                "2. **Cross-Validation**: Verify claims across reports",
                "3. **Completeness**: Are all financial aspects adequately covered?",
                "4. **Risk Blind Spots**: Risks mentioned by some but not all analysts",
                "5. **Evidence Quality**: Strength of supporting data/rationale",
                "6. **Actionability**: How implementable are the recommendations?",
                "7. **Holistic View**: Integrated assessment from all perspectives",
                ""
            ])
        
        prompt_parts.extend([
            "## Output Format:",
            "Provide structured feedback that can be used to improve analyses or inform decision-making.",
            ""
        ])

        return "\n".join(prompt_parts)


@dataclass
class CriticAgentTool:
    """AI agent that critiques other agents' responses to help them improve their analysis."""
    
    config: Dict[str, Any]
    memory_tool: Optional['MemoryStoreTool'] = None
    all_tools: Optional[Dict[str, Any]] = None  # Reference to all available tools

    def __post_init__(self) -> None:
        self.name: str = str(self.config.get("name") or "")
        if not self.name:
            raise ValueError("Local tool config missing required field 'name'")

        self.description: str = str(
            self.config.get("description")
            or "AI agent that critiques and evaluates responses from other analyst agents."
        )

        self.api_base: str = "local"
        self.input_schema: Dict[str, Any] = self.config.get("input_schema") or {}

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = arguments or {}
        response_key = args.get("response_key")
        response_text = args.get("response_text")
        agent_type = args.get("agent_type", "general")
        criteria = args.get("criteria", ["completeness", "accuracy", "depth", "actionability", "evidence"])
        context = args.get("context", "")
        api_key = args.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        save_to = args.get("save_to")
        model = args.get("model", "deepseek/deepseek-r1-0528:free")
        timeout = args.get("timeout", 60)

        if not api_key:
            return {"error": "OpenRouter API key required. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY environment variable."}

        # Get the response to critique
        response_content = None
        if response_text:
            response_content = response_text
        elif response_key and self.memory_tool:
            get_result = self.memory_tool.handle(session_id, {"action": "get", "key": response_key})
            if "error" in get_result:
                return {"error": f"Could not find response with key '{response_key}': {get_result.get('error')}"}
            
            response_value = get_result.get("content", {}).get("value")
            if isinstance(response_value, dict):
                # Extract the analysis/hypothesis from the response object
                response_content = response_value.get("analysis") or response_value.get("hypothesis") or str(response_value)
            else:
                response_content = str(response_value)
        else:
            # Auto-discover recent analyst outputs
            if not self.memory_tool:
                return {"error": "No response provided and MemoryStoreTool not configured"}
            
            keys_result = self.memory_tool.handle(session_id, {"action": "keys"})
            if "error" in keys_result:
                return {"error": "Could not retrieve memory keys to auto-discover response"}
            
            all_variables = keys_result.get("content", {}).get("keys", [])
            # Look for analyst outputs (typically have _analysis or _hypothesis suffixes)
            analyst_vars = [k for k in all_variables if any(x in k.lower() for x in ["analyst", "hypothesis", "analysis"])]
            
            if not analyst_vars:
                return {
                    "error": "No response provided. Please specify 'response_key' or 'response_text'.",
                    "available_variables": all_variables
                }
            
            # Use the first analyst variable found
            response_key = analyst_vars[0]
            get_result = self.memory_tool.handle(session_id, {"action": "get", "key": response_key})
            response_value = get_result.get("content", {}).get("value")
            if isinstance(response_value, dict):
                response_content = response_value.get("analysis") or response_value.get("hypothesis") or str(response_value)
            else:
                response_content = str(response_value)

        if not response_content:
            return {"error": "No response content found to critique"}

        # Build the critique prompt
        prompt = self._build_prompt(response_content, agent_type, criteria, context)

        # Call OpenRouter API
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/xendex-mcp-server",
                    "X-Title": "Xendex MCP Critic Agent",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a rigorous critic and peer reviewer for financial analysis. Your role is to evaluate analytical responses for quality, completeness, logical consistency, and actionability. Provide constructive feedback that helps analysts improve their work."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})
                raw_content = message.get("content", "")
                
                # OpenRouter separates reasoning from content for DeepSeek R1
                # If content is empty, check the reasoning field
                if not raw_content:
                    reasoning = message.get("reasoning", "")
                    if reasoning:
                        print(f"DEBUG critic_agent: Content empty, using reasoning field (length: {len(reasoning)})")
                        raw_content = reasoning
                
                critique = _extract_content_from_response(raw_content)
                
                # Debug: Log if critique is empty
                if not critique:
                    print(f"WARNING: Empty critique extracted from critic agent response")
                    print(f"Raw content length: {len(raw_content)})")
                
                critique_obj = {
                    "critic": "critic_agent",
                    "critique": critique,
                    "target_response_key": response_key,
                    "agent_type": agent_type,
                    "criteria": criteria,
                    "context": context,
                    "model": model,
                    "timestamp": result.get("created"),
                    "usage": result.get("usage", {})
                }
                
                # Save the critique if requested
                if save_to and self.memory_tool:
                    self.memory_tool.handle(session_id, {
                        "action": "set",
                        "key": save_to,
                        "value": critique_obj
                    })

                return {
                    "content": {
                        "critique": critique,
                        "target_response_key": response_key,
                        "agent_type": agent_type,
                        "criteria": criteria,
                        "model_used": model,
                        "saved_to": save_to if save_to else None,
                        "usage": result.get("usage", {})
                    }
                }
            return {"error": "No critique generated from the model"}

        except Exception as e:
            return {"error": f"Critique failed: {str(e)}"}

    def _build_prompt(self, response_content: str, agent_type: str, criteria: List[str], context: str) -> str:
        """Build the critique prompt."""
        prompt_parts = [
            "# Response Critique Task",
            "",
            f"## Agent Type: {agent_type}",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "## Context:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "## Response to Critique:",
            "```",
            response_content,
            "```",
            "",
            "## Evaluation Criteria:",
        ])
        
        for criterion in criteria:
            prompt_parts.append(f"- {criterion.replace('_', ' ').title()}")
        
        prompt_parts.extend([
            "",
            "## Task:",
            "Provide a comprehensive critique of the above response. Your critique should include:",
            "",
            "1. **Strengths**: What the response does well",
            "2. **Weaknesses**: Gaps, logical flaws, or unsupported claims",
            "3. **Missing Elements**: What critical aspects are not addressed",
            "4. **Specific Improvements**: Concrete suggestions for enhancement",
            "5. **Risk Assessment**: Potential issues if recommendations are followed",
            "6. **Overall Rating**: Score from 1-10 with justification",
            "",
            "Be constructive but rigorous. Identify specific areas for improvement and provide actionable feedback.",
            ""
        ])

        return "\n".join(prompt_parts)
