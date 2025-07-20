import subprocess
import time
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from src.agent import runtime_config
from src.agent.tool_set.constant import *

def get_runtime_config(config: Optional[RunnableConfig] = None) -> Any:
    """Helper function to safely get runtime config.
    
    First tries to get runtime_object from config["configurable"]["runtime_object"].
    Falls back to the global rc if not available.
    
    Args:
        config: RunnableConfig object that might contain runtime_object
        
    Returns:
        RuntimeConfig instance
    """
    if config and isinstance(config, dict) and "configurable" in config:
        runtime_obj = config["configurable"].get("runtime_object")
        if runtime_obj:
            return runtime_obj
    
    # Fall back to global rc
    return runtime_config.RuntimeConfig()

def run_shell_local(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN_CHAR,
    truncate_notice: str = CONTENT_TRUNCATED_NOTICE,
) -> tuple[int, str, str]:
    """Run a shell command synchronously with a timeout.

    Args:
        cmd: The shell command to run.
        timeout: The maximum time to wait for the command to complete.
        truncate_after: The maximum number of characters to return for stdout and stderr.

    Returns:
        A tuple containing the return code, stdout, and stderr.
    """

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = process.communicate(timeout=timeout)

        return (
            process.returncode or 0,
            maybe_truncate(stdout, truncate_after=truncate_after, truncate_notice=truncate_notice),
            maybe_truncate(
                stderr,
                truncate_after=truncate_after,
                truncate_notice=CONTENT_TRUNCATED_NOTICE,  # Use generic notice for stderr
            ),
        )
    except subprocess.TimeoutExpired:
        process.kill()
        elapsed_time = time.time() - start_time
        raise TimeoutError(f"Command '{cmd}' timed out after {elapsed_time:.2f} seconds")
