from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()

from typing import Dict, Any, List
from loguru import logger


class ElicitationHelper:
    """
    Helper for gathering user inputs through guided questions.
    """

    def __init__(self):
        self.session_data: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = 1000  # Prevent memory leaks

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------
    def _cleanup_sessions(self):
        if len(self.session_data) > self.max_sessions:
            sessions_to_remove = list(self.session_data.keys())[: len(self.session_data) // 2]
            for session_id in sessions_to_remove:
                del self.session_data[session_id]

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def elicit_inputs(
        self,
        session_id: str,
        arguments: Dict[str, Any],
        input_schema: Dict[str, Any],
    ) -> Dict[str, Any]:

        if session_id == "http-session":
            session_id = "elicitation-session"

        if session_id not in self.session_data:
            self._cleanup_sessions()
            self.session_data[session_id] = {}

        session = self.session_data[session_id]

        # Merge partial updates (supports dot notation)
        session = self._deep_merge(session, arguments)
        self.session_data[session_id] = session

        logger.info(f"Session data after update: {session}")

        validation_data = session.copy()

        if self._has_complete_input_for_schema(validation_data, input_schema):
            del self.session_data[session_id]
            return {
                "content": validation_data,
                "complete": True,
            }

        missing_info = self._identify_missing_info_for_schema(
            validation_data, input_schema
        )

        questions = self._generate_questions_for_schema(
            missing_info, session, input_schema
        )

        return {
            "content": {
                "elicitation": True,
                "questions": questions,
                "current_data": session,
                "next_steps": "Please provide the requested information to continue.",
            },
            "complete": False,
        }

    # ------------------------------------------------------------------
    # Deep merge with dot-notation support
    # ------------------------------------------------------------------
    def _deep_merge(
        self, base: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, handling nested structures
        and dot-notation keys like:
        tf_vars.StorageAccount.subnet.name
        """
        result = base.copy()

        for key, value in update.items():
            if "." in key:
                self._set_nested_value(result, key, value)
            elif (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _set_nested_value(
        self, data: Dict[str, Any], key_path: str, value: Any
    ):
        """
        Set nested dictionary value using dot notation.
        """
        keys = key_path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:

        missing = []
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required_fields:
            field_path = f"{path}.{field}" if path else field

            if field not in data:
                missing.append(field_path)
                continue

            field_schema = properties.get(field, {})

            if field_schema.get("type") == "object":
                if not isinstance(data[field], dict):
                    missing.append(field_path)
                    continue

                if "patternProperties" in field_schema:
                    if not data[field]:
                        missing.append(field_path)
                        continue

                    for key, value in data[field].items():
                        pattern_schema = next(
                            iter(field_schema["patternProperties"].values())
                        )
                        missing.extend(
                            self._validate_against_schema(
                                value,
                                pattern_schema,
                                f"{field_path}.{key}",
                            )
                        )
                else:
                    missing.extend(
                        self._validate_against_schema(
                            data[field], field_schema, field_path
                        )
                    )

        return missing

    def _has_complete_input_for_schema(
        self, arguments: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        return len(self._validate_against_schema(arguments, schema, "")) == 0

    def _identify_missing_info_for_schema(
        self, arguments: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        missing_fields = self._validate_against_schema(arguments, schema, "")
        return {field: f"Missing required field: {field}" for field in missing_fields}

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------
    def _generate_questions_for_schema(
        self,
        missing_info: Dict[str, Any],
        session: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        questions = []
        missing_fields = list(missing_info.keys())

        # If tf_vars missing → ask service-specific questions
        if "tf_vars" in missing_fields:
            services = session.get("services", [])
            for service in services:
                questions.extend(self._generate_service_questions(service))
            return questions[:3]

        for field_path in missing_fields:
            questions.append(
                {
                    "field": field_path,
                    "question": self._get_question_text(field_path),
                    "type": self._get_field_type(field_path),
                    "example": self._get_example_for_field(field_path),
                }
            )

        return questions[:3]

    def _generate_service_questions(self, service: str) -> List[Dict[str, Any]]:
        return [
            {
                "field": f"tf_vars.{service}.namespace",
                "question": f"What namespace/prefix should be used for {service}?",
                "type": "string",
                "example": "finance-mcp",
            },
            {
                "field": f"tf_vars.{service}.location",
                "question": f"Which Azure region for {service}?",
                "type": "string",
                "example": "eastus2",
            },
            {
                "field": f"tf_vars.{service}.tags.environment",
                "question": f"What environment is this {service} for?",
                "type": "string",
                "example": "dev",
            },
            {
                "field": f"tf_vars.{service}.tags.project",
                "question": f"What project is this {service} for?",
                "type": "string",
                "example": "my-project",
            },
            {
                "field": f"tf_vars.{service}.subnet.name",
                "question": f"What subnet name for {service}?",
                "type": "string",
                "example": "my-subnet",
            },
            {
                "field": f"tf_vars.{service}.subnet.resource_group_name",
                "question": f"What resource group contains the subnet for {service}?",
                "type": "string",
                "example": "my-vnet-rg",
            },
            {
                "field": f"tf_vars.{service}.subnet.virtual_network_name",
                "question": f"What virtual network name for {service}?",
                "type": "string",
                "example": "my-vnet",
            },
        ]

    # ------------------------------------------------------------------
    # Question helpers
    # ------------------------------------------------------------------
    def _get_question_text(self, field_path: str) -> str:
        if field_path == "services":
            return "Which Azure services do you want to deploy?"
        if "namespace" in field_path:
            return "What namespace/prefix should be used?"
        if "location" in field_path:
            return "Which Azure region should be used?"
        if "environment" in field_path:
            return "What environment is this for?"
        if "project" in field_path:
            return "What is the project name?"
        if "subnet.name" in field_path:
            return "What is the subnet name?"
        if "subnet.resource_group_name" in field_path:
            return "What resource group contains the subnet?"
        if "subnet.virtual_network_name" in field_path:
            return "What is the virtual network name?"

        return f"Please provide {field_path.split('.')[-1]}"

    def _get_field_type(self, field_path: str) -> str:
        if field_path == "services":
            return "array"
        return "string"

    def _get_example_for_field(self, field_path: str) -> Any:
        if "services" in field_path:
            return ["App Service"]
        if "namespace" in field_path:
            return "finance-ai"
        if "location" in field_path:
            return "eastus2"
        if "environment" in field_path:
            return "dev"
        if "project" in field_path:
            return "my-project"
        if "subnet.name" in field_path:
            return "my-subnet"
        if "subnet.resource_group_name" in field_path:
            return "my-vnet-rg"
        if "subnet.virtual_network_name" in field_path:
            return "my-vnet"

        return "<value>"


elicitation_helper = ElicitationHelper()
