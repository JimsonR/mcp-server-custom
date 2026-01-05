from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
