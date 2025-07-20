import os
import subprocess
import time
import asyncio
import uuid
import logging
from pathlib import Path
from typing import Annotated, List, Optional, Union
import ast
from git import Repo
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from swerex.runtime.abstract import CreateBashSessionRequest, BashAction, Command, WriteFileRequest, CloseBashSessionRequest
from swerex.deployment.docker import DockerDeployment
from src.agent import runtime_config
from src.agent.runtime_config import RuntimeType
from src.agent.constant import PATCH_RESULT_DIR, RUNTIME_DIR
from src.agent.tool_set.constant import MAX_LIST_FILES, MAX_RESPONSE_LEN_CHAR, FILE_CONTENT_TRUNCATED_NOTICE
from src.agent.tool_set.utils import get_runtime_config
from src.agent.tool_set.edit_tool import str_replace_editor
from swerex.exceptions import CommandTimeoutError
from src.agent.logging_config import get_logger


def prepare_input_dir(in_dir, config: RunnableConfig = None):
    rc = get_runtime_config(config)
    assert rc.initialized

    if in_dir == ".":
        in_dir = ""
    return os.path.join(rc.proj_path, in_dir)
    
    
def prepare_output_dir(out_dir, config: RunnableConfig = None):
    rc = get_runtime_config(config)
    assert rc.initialized

    return out_dir.replace(rc.proj_path + "/", "")

# Setup logging
logger = get_logger(__name__)


@tool
def search_files_by_keywords(
    directory: Annotated[str, "The root directory to search in, can also be a file path if you want to search keywords within a single file"],
    keywords: Annotated[Union[list[str], str], "A list of regex patterns to look for, or a JSON string representation of such a list. Each pattern should be at least three characters long. Patterns are treated as regex by default"],
    config: RunnableConfig = None
):
    """
    Recursively searches for files in the given directory (including subdirectories) that contain the specified regex patterns or keywords in their filename or content using ripgrep for faster performance.

    Args:
        directory (str): The root directory to search in, can also be a file path if you want to search keywords within a single file.
        keywords (Union[list[str], str]): A list of regex patterns to look for, or a JSON string representation of such a list. Each pattern should be at least three characters long. Patterns are treated as regex by default.

    Returns:
        dict: A dictionary where each keyword/pattern maps to a list of file paths and their corresponding line ranges that match the search.
    """

    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None
    
    # Handle case where keywords comes as a JSON string instead of a list
    if isinstance(keywords, str):
        try:
            import json
            keywords = json.loads(keywords)
            if not isinstance(keywords, list):
                return "ArgumentError: The keywords parameter must be a list of strings"
        except (json.JSONDecodeError, TypeError):
            return "ArgumentError: The keywords parameter must be a list of strings, or a valid JSON string representation of a list"
    
    # Input validation
    if not keywords:
        return "ArgumentError: The keywords must be a non-empty list"
    
    if len(keywords) > 10:
        return "ArgumentError: The number of keywords must be less than 10"
    
    # Ensure all keywords are strings
    for i, keyword in enumerate(keywords):
        if not isinstance(keyword, str):
            return f"ArgumentError: All keywords must be strings, but item {i} is {type(keyword).__name__}"
        if len(keyword) < 3:
            return f"ArgumentError: The keyword '{keyword}' must be at least 3 characters long"
    
    # Normalize the directory path - using the original prepare_input_dir function
    directory = prepare_input_dir(directory, config)
    
    # Check if ripgrep is installed
    try:
        subprocess.run(["rg", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return "Error: ripgrep is not installed or not available in PATH. Please install ripgrep for this tool to work."
    
    # Make sure directory exists
    if not os.path.exists(directory):
        return f"Error: Directory or file '{directory}' does not exist"
    else:
        log_output.append(f"--search under: {directory}")
    
    # Initialize results dictionary
    results = {}
    
    # Search for each keyword
    for keyword in keywords:
        matching_files = []
        
        try:
            # Build ripgrep command for content search
            rg_cmd = ["rg", "--line-number"]
            
            # Add filename matching capability (search in filenames too)
            if os.path.isfile(directory):
                # Check if the filename contains the keyword
                if keyword in os.path.basename(directory):
                    matching_files.append(prepare_output_dir(directory, config))
            else:
                try:
                    file_pattern_cmd = ["rg", "--files", "-g", f"*{keyword}*"]
                    file_proc = subprocess.run(
                        file_pattern_cmd + [directory], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    # Add files that match by name
                    for file_path in file_proc.stdout.splitlines():
                        matching_files.append(prepare_output_dir(file_path, config))
                except subprocess.SubprocessError as e:
                    logger.error(f"Error searching filenames: {e}")
            
            # Content search command
            search_target = directory
            
            # Execute ripgrep for content search
            process = subprocess.run(
                rg_cmd + [keyword, search_target],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Process the output
            file_line_map = {}
            for line in process.stdout.splitlines():
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        file_path = parts[0]
                        try:
                            line_number = int(parts[1])
                            if file_path not in file_line_map:
                                file_line_map[file_path] = {"first": line_number, "last": line_number}
                            else:
                                file_line_map[file_path]["last"] = line_number
                        except ValueError:
                            # Skip lines where line number can't be parsed
                            continue
                except Exception as e:
                    logger.error(f"Error: processing line '{line}': {e}")
            
            # Format results
            for file_path, line_info in file_line_map.items():
                output_path = prepare_output_dir(file_path, config)
                if line_info["first"] == line_info["last"]:
                    matching_files.append(f"{output_path}: line {line_info['first']}")
                else:
                    matching_files.append(f"{output_path}: line {line_info['first']}-{line_info['last']}")
                    
                # Check if we've reached the maximum files limit
                if len(matching_files) >= MAX_LIST_FILES:
                    matching_files.insert(0, f"Note: Too many files found. Only the first {MAX_LIST_FILES} files are returned, please narrow down your search")
                    break
                    
        except Exception as e:
            matching_files.append(f"Error: {str(e)}")
        
        results[keyword] = matching_files
        log_output.append(f"----keyword: {keyword} Matches: {len(matching_files)}")
    
    logger.info("\n".join(log_output))
    return results



@tool
def view_directory(dir_path: str = "./", depth: Optional[int] = None, config: RunnableConfig = None) -> List[str]:
    """View the file structure of the repository, including directories (marked with /).
    Automatically reduces depth if entries exceed MAX_LIST_FILES.

    Args:
        dir_path (str): Starting directory. Defaults to './'.
        depth (Optional[int]): Maximum depth. None for unlimited. Defaults to None.

    Returns:
        List[str]: Sorted list of directories (with /) and files.
    """
    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None

    log_output.append(f"--view_directory: {dir_path} depth: {depth}")

    rc = get_runtime_config(config)
    assert rc.initialized

    # Normalize dir_path to ensure proper filtering
    #
    if dir_path.startswith("./"):
        processed_dir = dir_path[2:]
    else:
        processed_dir = dir_path

    if processed_dir:
        processed_dir = processed_dir.rstrip("/") + "/"

    # Fetch all files in the repository
    file_list = []
    if rc.runtime_type == RuntimeType.LOCAL:
        repo = Repo(rc.proj_path)
        file_list = [entry.path for entry in repo.commit().tree.traverse()]
    elif rc.runtime_type == RuntimeType.SWEREX:
        runtime = rc.swe_rex_deployment.runtime
        result = asyncio.run(runtime.run_in_session(BashAction(command="git ls-files",check="ignore")))
        file_list = [line.strip() for line in result.output.splitlines()]
    else:
        raise ValueError("Unsupported runtime type")

    # Filter out .git and its subfolders/files
    file_list = [p for p in file_list if not (p == ".git" or p.startswith(".git/"))]

    # Filter out all hidden files and directories (those starting with a dot at any level)
    file_list = [p for p in file_list if not (os.path.basename(p).startswith(".") or any(part.startswith(".") for part in p.split("/")))]

    # Collect files and directories with their depths
    all_files = []  # Format: (full_path, depth)
    all_dirs = set()  # Format: (full_dir_path, depth)

    for path in file_list:
        # Filter files outside the target directory
        if not path.startswith(processed_dir):
            continue

        # Calculate file depth
        rel_path = path[len(processed_dir) :] if processed_dir else path
        file_depth = rel_path.count("/")
        all_files.append((path, file_depth))

        # Generate parent directories from the file path
        dir_components = rel_path.split("/")[:-1]  # Exclude filename
        current_dir = []
        for component in dir_components:
            current_dir.append(component)
            dir_rel_path = "/".join(current_dir)
            dir_depth = dir_rel_path.count("/")  # Depth is based on slashes
            full_dir_path = f"{processed_dir}{dir_rel_path}/"
            all_dirs.add((full_dir_path, dir_depth))

    # Function to filter entries by depth
    def filter_entries(max_depth: Optional[int]) -> List[str]:
        # Filter files
        filtered_files = [path for path, d in all_files if (max_depth is None) or (d <= max_depth)]
        # Filter directories
        filtered_dirs = [dir_path for dir_path, d in all_dirs if (max_depth is None) or (d <= max_depth)]
        # Combine and deduplicate
        entries = list(set(filtered_dirs + filtered_files))
        return sorted(entries)  # Alphabetical order

    # Check initial entry count
    initial_entries = filter_entries(depth)
    if len(initial_entries) <= MAX_LIST_FILES:
        log_output.append(f"----depth: {depth} entries: {len(initial_entries)}")
        logger.info("\n".join(log_output))
        return "\n".join(initial_entries)

    # Automatically reduce depth
    start_depth = (
        depth
        if depth is not None
        else max(max((d for _, d in all_files), default=0), max((d for _, d in all_dirs), default=0))
    )

    for d in range(start_depth, -1, -1):
        adjusted_entries = filter_entries(d)
        if len(adjusted_entries) <= MAX_LIST_FILES:

            log_output.append(f"----reduced depth: {d} entries: {len(adjusted_entries)}")
            logger.info("\n".join(log_output))
            return f"Note: Reduced depth to {d} with {len(adjusted_entries)} entries:\n" + "\n".join(adjusted_entries)

    # Fallback (depth 0)
    final_entries = filter_entries(0)
    
    log_output.append(f"----fallback depth: 0 entries: {len(final_entries)}")
    logger.info("\n".join(log_output))

    return f"Note: Limited to depth 0 with {len(final_entries)} entries\n" + "\n".join(final_entries)




@tool
def view_file_structure(
    file_path: Annotated[
        str,
        "Path to a Python file relative to the project root (e.g., 'src/main.py')"
    ],
    config: RunnableConfig = None
) -> str:
    """Extracts and displays the hierarchical structure of a Python file. Ideally suited for files that are too large to view directly.
    
    Parses the specified Python file and returns a formatted representation of:
    - Classes with line numbers and docstrings
    - Methods with parameters, line numbers, and docstrings
    - Functions with parameters, line numbers, and docstrings
    
    Line numbers can help identify the exact range of code within a file, which can then be viewed using the `view` command with range in the `str_replace_editor` tool.
    
    The indentation in the output indicates the nesting level:
    
    Class: ClassName (line X)
    -- Doc: Class docstring
    -- Method: method_name(param1, param2) (line Y)
    ---- Doc: Method docstring
    Function: function_name(param1, param2) (line Z)
    -- Doc: Function docstring
    
    Args:
        file_path: Path to a Python file relative to the project root. Must point to an existing .py file.
    
    Returns:
        A formatted string representing the file's structure with indentation showing the hierarchy.
    """

    # if not python file, return error message
    if not file_path.endswith('.py'):
        return "View file structure failed. Currenly only support python file."

    rc = get_runtime_config(config)
    assert rc.initialized

    if rc.runtime_type == RuntimeType.LOCAL:
        full_file_path = os.path.join(rc.proj_path, file_path)
        if not os.path.isfile(full_file_path):
            raise ValueError(f"file_name: '{file_path}' doesn't exist!")
        with open(full_file_path, encoding="utf-8") as f:
            file_content = f.read()  # FIXME : add line number
    if rc.runtime_type == RuntimeType.DOCKER:
        file_content = rc.docker_container.exec_run(f"cat {file_path}", tty=True, stdin=True)
        file_content = file_content.output.decode("utf-8")
    if rc.runtime_type == RuntimeType.SWEREX:
        runtime = rc.swe_rex_deployment.runtime
        file_content = asyncio.run(runtime.run_in_session(BashAction(command=f"cat {file_path}")))
        file_content = file_content.output

    return parse_content_structure(file_content)

@tool
def view_file_content(
    file_name: Annotated[
        str,
        "File name relative to git root, candidates can be retrieved by `view_directory`",
    ],
    view_range: Annotated[
        Optional[List[int]],
        "Optional parameter [start_line, end_line] to specify the range of lines to view",
    ] = None,
    config: RunnableConfig = None,
) -> str:
    """
    Read the content of the specified file.
    Parameters:
        file_name (str): File name relative to the git root directory.
        view_range (Optional[List[int]]): Optional list containing [start_line, end_line] to limit the lines displayed.
    Usage:
        - LLM should initially attempt to read the entire file content.
        - If the file is too large, LLM can use the `view_file_structure` tool to identify relevant code ranges,
          and then call this tool again specifying the `view_range` to read only the necessary lines.
    Returns:
        str: Content of the file or the specified line range.
    """

    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None

    rc = get_runtime_config(config)
    assert rc.initialized
    
    if view_range:
        return str_replace_editor.invoke({"command": "view", "path": file_name, "view_range": view_range},config = config)
    else:
        return str_replace_editor.invoke({"command": "view", "path": file_name},config = config)


@tool("bash")
def run_shell_cmd(
    command: Annotated[str, "A shell command to be run"],
    config: RunnableConfig = None,
) -> str:
    # """Run a shell command and return the stdout results, your working directory is the root of the project. Cannot run interactive commands like `pdb` or `gdb`."""/
    """Run commands in a bash shell
* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.

Args:
    command (str): A shell command to be run

Returns:
    str: The stdout results of the command
    """
    
    # Get runtime config
    rc = get_runtime_config(config)
    assert rc.initialized
    
    # Get project path from runtime config
    proj_path = rc.proj_path
    logger.info(f"run_shell_cmd using project path: {proj_path}")

    if rc.runtime_type == RuntimeType.LOCAL:
        import subprocess

        process = subprocess.Popen(
            "/bin/bash",
            cwd=proj_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
        )
        out, err = process.communicate(command)
        return out
    
    elif rc.runtime_type == RuntimeType.SWEREX:
        runtime = rc.swe_rex_deployment.runtime
        try:
            cmd_output = asyncio.run(runtime.run_in_session(BashAction(command=command,check="silent", timeout=60)))
        except CommandTimeoutError:
            # Close the current session and create a new one, then rerun the command
            logger.warning("Timeout error, trying closing session and creating new one")
            try:
                asyncio.run(runtime.close_session(CloseBashSessionRequest()))
            except Exception as e:
                logger.error(f"Error closing session after timeout: {e}")
            try:
                asyncio.run(runtime.create_session(CreateBashSessionRequest()))
                asyncio.run(runtime.run_in_session(BashAction(command="cd /",check="silent", timeout=10)))
                asyncio.run(runtime.run_in_session(BashAction(command="cd testbed",check="silent", timeout=10)))
            except Exception as e:
                logger.error(f"Error creating new session after timeout: {e}")
            # Retry the command once, but catch timeout error on retry as well
            try:
                cmd_output = asyncio.run(runtime.run_in_session(BashAction(command=command,check="silent", timeout=120)))
            except CommandTimeoutError as e:
                logger.error(f"Command timed out again after session recreation: {e}")
                return f"Error: Command '{command}' timed out after 120 seconds, try using other command"
        return cmd_output.output
    else:
        raise ValueError("Unsupported runtime type")



@tool
def think(thought: Annotated[str, "The thought to log."], config: RunnableConfig = None) -> str:
    """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""
    return "Your thought has been logged. Please continue your work.\n"

