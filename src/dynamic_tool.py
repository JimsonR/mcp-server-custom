from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

import json
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from loguru import logger

from src.elicitation import ElicitationHelper
from src.registry_client import get_http_client, new_http_client


elicitation_helper = ElicitationHelper()

class DynamicToolHandler:
    """Handler for dynamic tool operations based on prompts."""

    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("Dynamic tool config must be an object")

        self.config: Dict[str, Any] = config

        self.name: str = str(config.get("name") or "")
        if not self.name:
            raise ValueError("Dynamic tool config missing required field 'name'")

        self.description: str = str(config.get("description") or "Dynamic tool")
        self.api_base: str = str(config.get("api_base") or "").rstrip("/")
        if not self.api_base:
            raise ValueError(f"Dynamic tool '{self.name}' missing required field 'api_base'")

        self.endpoint: str = str(config.get("endpoint") or "/execute")
        self.method: str = str(config.get("method") or "POST").upper()

        self.input_schema: Dict[str, Any] = config.get("input_schema") or {}
        self.internal_schema: Optional[Dict[str, Any]] = config.get("internal_schema")
        self.output_schema: Optional[Dict[str, Any]] = config.get("output_schema")

        self._check_availability()

    def handle(self, session_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the dynamic tool request."""
        logger.info(f"Handling dynamic tool: {self.name}")

        try:
            args: Dict[str, Any] = arguments or {}

            if self.input_schema:
                schema_for_elicitation = self.internal_schema or self.input_schema
                elicitation_result = elicitation_helper.elicit_inputs(
                    session_id,
                    args,
                    schema_for_elicitation,
                )

                if not elicitation_result.get("complete", False):
                    # main.py expects content dict with {"elicitation": True, ...}
                    return {"content": elicitation_result.get("content", {})}

                args = elicitation_result.get("content", {})

            return self._execute_tool(session_id, args)

        except Exception as e:
            logger.error(f"Error in {self.name} handler: {e}")
            return {"error": str(e)}
    
    def _execute_tool(self, session_id:str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool via configured endpoint."""
        logger.info(f"Executing tool {self.name} with arguments: {arguments}")

        url = self.config.get("url")
        if not url:
            url = urljoin(self.api_base + "/", self.endpoint.lstrip("/"))

        headers = {"Content-Type": "application/json"}

        # Parse and validate user input acording to input schema
        parsed_payload = self._parse_input(arguments)
        logger.info(f"Parsed payload for tool {self.name}: {parsed_payload}")
        if "error" in parsed_payload:
            return parsed_payload

        http_client = get_http_client(session_id) or new_http_client(session_id)
        logger.info(f"Making {self.method} request to {url} with payload: {parsed_payload}")

        try:
            if self.method == "GET":
                response = http_client.get(url, params=parsed_payload, headers=headers, timeout=300)
            elif self.method in {"POST", "PUT", "PATCH", "DELETE"}:
                response = http_client.request(
                    self.method, url, json=parsed_payload, headers=headers, timeout=300
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")
            logger.info(f"Request completed")
            response.raise_for_status()
        except Exception as e:
            logger.info(f"Http request failed: {e}")
            raise
        logger.info(f"Response status: {response.status_code}, content: {response.text}")

        try:
            result = response.json()
            logger.info(f"Response JSON parsed successfully")
            return {"content": result}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response")
            return {"content": response.text or "Success"}

    def _parse_input(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Parse user input according to input schema and create api payload"""
        if not self.input_schema:
            return arguments
        
        # Parse input according to internal schema structure (validation already done in elicitation)
        return self._build_payload(arguments, self.internal_schema or self.input_schema)
    
    def _build_payload(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build payload based on schema structure."""
        payload = {}
        for key, value in schema.get('properties', {}).items():
            if key in data:
                if value.get('type') == 'object':
                    payload[key] = self._build_payload(data[key], value)
                else:
                    payload[key] = data[key]
        return payload
    
    def _check_availability(self):
        """ Check if tool endpoint is available"""
        try:
            health_url = f"{self.api_base}/health_check"
            response = requests.get(health_url, timeout=5)
            if response.status_code != 200:
                logger.warning(
                    f"Tool {self.name} health check returned {response.status_code}"
                )
        except Exception as e:
            logger.warning(f"Tool {self.name} health check failed: {e}")