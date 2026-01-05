import json
import html
from typing import Dict, Any

class PromptParser:
    """Parse natural language prompts using system prompt to extract structured parameters."""

    def parse_prompt(self, prompt: str, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse prompt and extract parameters based on schema using system prompt."""
        decoded_prompt = html.unescape(prompt)
        system_prompt = self._build_system_prompt(input_schema)
        
        # Simulate LLM parsing with prompt (replace with actual LLM call)
        extracted = self._parse_with_system_prompt(decoded_prompt, system_prompt, input_schema)
        extracted["original_prompt"] = decoded_prompt
        return extracted

    def _build_system_prompt(self, schema: Dict[str, Any]) -> str:
        """Build system prompt for parsing based on schema."""
        return f"""You are a cloud infrastructure parameter extraction system. Extract structured parameters from user prompts.

Schema: {json.dumps(schema, indent=2)}

Extract any relevant information from the user prompt that matches the schema structure.
Support all cloud providers (Azure, AWS, GCP, etc.) and their services.

For services, identify any cloud service mentioned (storage, compute, database, networking, etc.).
For locations/regions, extract any geographic region mentioned.
For environments, identify dev/test/staging/prod mentions.
For projects, extract project names or identifiers.
For networking, extract subnet, VPC, resource group information.

Return only valid JSON matching the schema structure. If no relevant information found, return empty object {{}}.

Examples:
- "Deploy S3 bucket" -> {{"services": ["S3"]}}
- "Create Azure storage in eastus for dev" -> {{"services": ["Storage Account"], "tf_vars": {{"location": "eastus", "tags": {{"environment": "dev"}}}}}}
- "Setup GCP compute instance" -> {{"services": ["Compute Engine"]}}"""

    def _parse_with_system_prompt(self, prompt: str, system_prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse prompt using system prompt logic (simplified implementation)."""
        extracted = {}
        # Extract services dynamically
        services = self._extract_services_dynamic(prompt)
        if services:
            extracted["services"] = services
            
        # Extract tf_vars for services
        tf_vars = {}
        for service in services:
            vars_dict = self._extract_vars_dynamic(prompt)
            if vars_dict:
                tf_vars[service] = vars_dict
        
        if tf_vars:
            extracted["tf_vars"] = tf_vars
            
        return extracted

    def _extract_services_dynamic(self, prompt: str) -> list:
        """Extract Azure services from prompt."""
        services = []
        prompt_lower = prompt.lower()
        
        # Azure service keywords
        service_keywords = {
            # Storage services
            "storage": "Storage Account",
            "blob": "Storage Account",
            "storage account": "Storage Account",
            # Compute services
            "virtual machine": "Virtual Machine",
            "vm": "Virtual Machine",
            # Container services
            "container": "Container Instances",
            "aks": "Kubernetes Service",
            "kubernetes": "Kubernetes Service",
            # Database services
            "sql": "SQL Database",
            "cosmos": "Cosmos DB",
            "database": "SQL Database",
            # Web services
            "web app": "App Service",
            "app service": "App Service",
            # Security services
            "key vault": "Key Vault",
            # Networking
            "vnet": "Virtual Network",
            "virtual network": "Virtual Network",
            "load balancer": "Load Balancer",
            # Serverless
            "function": "Function App",
            "azure function": "Function App"
        }
        
        for keyword, service_name in service_keywords.items():
            if keyword in prompt_lower and service_name not in services:
                services.append(service_name)
                
        return services

    def _extract_vars_dynamic(self, prompt: str) -> Dict[str, Any]:
        """Extract only clearly mentioned variables, let elicitation ask for missing ones."""
        vars_dict = {}
        words = prompt.lower().split()
        
        # Extract namespace if clearly stated
        for i, word in enumerate(words):
            if word == "namespace" and i + 1 < len(words):
                vars_dict["namespace"] = words[i + 1].rstrip(',.')
                break
                
        # Extract location/region if clearly stated
        for i, word in enumerate(words):
            if word == "region" and i + 1 < len(words):
                vars_dict["location"] = words[i + 1].rstrip(',.')
                break
                
        # Fallback to Azure region patterns
        if "location" not in vars_dict:
            azure_regions = ["eastus", "westus", "centralus", "eastus2", "westus2", "northeurope", "westeurope", "southeastasia", "eastasia"]
            for region in azure_regions:
                if region in prompt.lower():
                    vars_dict["location"] = region
                    break
                    
        # Extract environment if clearly stated
        for i, word in enumerate(words):
            if word == "environment" and i + 1 < len(words):
                if "tags" not in vars_dict:
                    vars_dict["tags"] = {}
                vars_dict["tags"]["environment"] = words[i + 1].rstrip(',.')
                break
                
        # Fallback to common environment patterns
        if "tags" not in vars_dict or "environment" not in vars_dict.get("tags", {}):
            envs = ["dev", "test", "staging", "prod"]
            for env in envs:
                if env in prompt.lower():
                    if "tags" not in vars_dict:
                        vars_dict["tags"] = {}
                    vars_dict["tags"]["environment"] = env
                    break
                    
        # Extract project if clearly stated
        for i, word in enumerate(words):
            if word == "project" and i + 1 < len(words):
                next_word = words[i + 1].rstrip(',.')
                if "tags" not in vars_dict:
                    vars_dict["tags"] = {}
                vars_dict["tags"]["project"] = next_word
                break
                
        # Extract subnet info if clearly mentioned
        subnet_info = {}
        
        # Extract subnet name
        for i, word in enumerate(words):
            if word == "subnet" and i + 1 < len(words):
                subnet_info["name"] = words[i + 1].rstrip(',.')
                break
                
        # Extract resource group from patterns like "resource group myrg"
        for i, word in enumerate(words):
            if word == "resource" and i + 1 < len(words) and words[i + 1] == "group" and i + 2 < len(words):
                subnet_info["resource_group_name"] = words[i + 2].rstrip(',.')
                break
                
        # Extract virtual network from patterns like "virtual network myvnet"
        for i, word in enumerate(words):
            if word == "virtual" and i + 1 < len(words) and words[i + 1] == "network" and i + 2 < len(words):
                subnet_info["virtual_network_name"] = words[i + 2].rstrip(',.')
                break
                
        # Only add subnet if any subnet info was found
        if subnet_info:
            vars_dict["subnet"] = subnet_info
            
        return vars_dict

# Initialization as seen in the last image
prompt_parser = PromptParser()