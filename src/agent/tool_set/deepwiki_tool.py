import json
import logging
import os
import time
from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
import requests

from src.agent import runtime_config
from src.agent.constant import REPO_MAP_DIR
from src.agent.runtime_config import RuntimeType
from src.agent.logging_config import get_logger

# Setup logging
logger = get_logger(__name__)


@tool
def ask_repository_agent(query: str, config: RunnableConfig = None) -> str:
    """
    Ask repository agent a question about the content of the current repository.

    This tool allows you to pose natural language questions about a repository's codebase, documentation, or other internal content. 
    A repository agent will answer your query based on the repository knowledge.

    When to Use:
    - Use this tool to understand repository-specific logic, structures, components, or decisions.
    - Appropriate for sementic search, simple lookups and complex reasoning tasks involving repository internals.

    When not to use:
    - Avoid using this tool to ask questions if you can achieve the same goal by using other tools.
    - The repository agent is READ ONLY, it cannot modify the repository.

    How to Use:
    - Provide a clear natural language query about the repository.
    - The response will reflect accurate, context-based information derived from the repository.

    Examples:
    - query = "What does the ConfigManager class do?"
    - query = "How does the authentication system handle multi-tenant user isolation in the login flow?"

    Args:
        query (str): A question about the repository.

    Returns:
        str: The answer from repository agent derived from repository content.
    """
    
    instance_id = config.get("configurable", {}).get("instance_id")
    assert instance_id, "instance_id is not set in the config"
    repo_dir = os.path.join(REPO_MAP_DIR, instance_id)
    
    logger.info(f"Asking {instance_id} at {repo_dir}")

    url = "http://localhost:8008/chat/completions/stream"
    
    payload = {
      "repo_url": repo_dir,
      "provider":"google",
      "model":"gemini-2.5-flash",
      "messages": [
          {
              "role": "user",
              "content": query,
          }
      ]
    }
    response = requests.post(url, json=payload, stream=True)
    if response.ok:
        return response.text
    else:
        logger.error(f"Error: {response.text}")
    return None


