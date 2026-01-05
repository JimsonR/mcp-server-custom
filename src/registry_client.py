from src.utility import add_site_packages_to_sys_path
add_site_packages_to_sys_path()
import requests
from typing import Any, Dict, Optional
from loguru import logger
import threading

active_http_clients: Dict[str, requests.Session] = {}
client_lock = threading.Lock()

def new_http_client(session_id: str , terraform_skip_tls_verify: bool = False) -> requests.Session:
    """
    Create a new HTTP client session with optional TLS verification skipping.
    """
    client = create_http_client(terraform_skip_tls_verify)

    with client_lock:
        active_http_clients[session_id] = client
    
    logger.info(f"Created new HTTP client for session_id: {session_id}")
    return client

def get_http_client(session_id: str) -> Optional[requests.Session]:
    """
    Retrieve an existing HTTP client session by session_id.
    """
    with client_lock:
        return active_http_clients.get(session_id)
    
def delete_http_client(session_id: str) -> None:
    """
    Delete an existing HTTP client session by session_id.
    """
    with client_lock:
        active_http_clients.pop(session_id, None)

def create_http_client(terraform_skip_tls_verify: bool = False) -> requests.Session:
    """
    Create a new HTTP client session with optional TLS verification skipping.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Xendex-MCP-Server-AI/1.0'
    })

    if terraform_skip_tls_verify:
        session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session

def get_http_client_from_headers(session_id: str,terraform_skip_tls_verify: bool = False) -> requests.Session:
    """
    create an HTTP client for the session.
    """
    return new_http_client(session_id, terraform_skip_tls_verify)
