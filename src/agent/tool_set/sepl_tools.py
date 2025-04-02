import os
from pathlib import Path
import subprocess
import time
import uuid
from typing import Annotated, List, Optional

from git import Repo
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from agent import runtime_config
from agent.constant import PATCH_RESULT_DIR, RUNTIME_DIR

MAX_LIST_FILES = 50  # the maximum number of files to return
MAX_RESPONSE_LEN_CHAR: int = 32000


@tool
def view_directory(dir_path: str = "./", depth: Optional[int] = None) -> List[str]:
    """View the file structure of the repository, including directories (marked with /).
    Automatically reduces depth if entries exceed 50.

    Args:
        dir_path (str): Starting directory. Defaults to './'.
        depth (Optional[int]): Maximum depth. None for unlimited. Defaults to None.

    Returns:
        List[str]: Sorted list of directories (with /) and files.
    """
    rc = runtime_config.RuntimeConfig()
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
    if rc.runtime_type == runtime_config.RuntimeType.LOCAL:
        repo = Repo(rc.proj_path)
        file_list = [entry.path for entry in repo.commit().tree.traverse()]
    else:
        raise ValueError("Unsupported runtime type")

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
        filtered_files = [
            path for path, d in all_files if (max_depth is None) or (d <= max_depth)
        ]
        # Filter directories
        filtered_dirs = [
            dir_path
            for dir_path, d in all_dirs
            if (max_depth is None) or (d <= max_depth)
        ]
        # Combine and deduplicate
        entries = list(set(filtered_dirs + filtered_files))
        return sorted(entries)  # Alphabetical order

    # Check initial entry count
    initial_entries = filter_entries(depth)
    if len(initial_entries) <= 50:
        return initial_entries

    # Automatically reduce depth
    start_depth = (
        depth
        if depth is not None
        else max(
            max((d for _, d in all_files), default=0),
            max((d for _, d in all_dirs), default=0),
        )
    )

    for d in range(start_depth, -1, -1):
        adjusted_entries = filter_entries(d)
        if len(adjusted_entries) <= 50:
            print(f"Note: Reduced depth to {d} with {len(adjusted_entries)} entries")
            return [
                f"Note: Reduced depth to {d} with {len(adjusted_entries)} entries"
            ] + adjusted_entries

    # Fallback (depth 0)
    final_entries = filter_entries(0)
    print(f"Note: Limited to depth 0 with {len(final_entries)} entries")
    return [
        f"Note: Limited to depth 0 with {len(final_entries)} entries"
    ] + final_entries


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
    rc = runtime_config.RuntimeConfig()
    assert rc.initialized
    print(
        'view_file_content: path:%s file_name="%s" view_range=%s'
        % (rc.proj_path, file_name, view_range)
    )
    if rc.runtime_type == runtime_config.RuntimeType.LOCAL:
        full_file_path = os.path.join(rc.proj_path, file_name)
        if not os.path.isfile(full_file_path):
            raise ValueError(f"file_name: '{file_name}' doesn't exist!")
        with open(full_file_path, encoding="utf-8") as f:
            lines = f.readlines()
            if view_range:
                start_line, end_line = view_range
                lines = lines[start_line - 1 : end_line]
                lines = [f"{line}" for i, line in enumerate(lines)]
            else:
                lines = [f"{i + 1}\t{line}" for i, line in enumerate(lines)]
            file_content = "".join(lines)
    else:
        raise NotImplementedError

    # FILE_CONTENT_TRUNCATED_NOTICE = '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with the `search_file_by_keywords` tool or `view_file_structure` tool in order to find the line numbers of what you are looking for, and then use this tool with view_range.</NOTE>'
    FILE_CONTENT_TRUNCATED_NOTICE = "<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with the `search_file_by_keywords` tool or view the file structure below in order to find the line numbers of what you are looking for, and then use this tool with view_range.</NOTE>"
    if len(file_content) > MAX_RESPONSE_LEN_CHAR:
        truncated = True
    else:
        truncated = False
    snippet_content = (
        file_content
        if not truncated
        else file_content[:MAX_RESPONSE_LEN_CHAR] + FILE_CONTENT_TRUNCATED_NOTICE
    )
    snippet_content = snippet_content.expandtabs()

    if view_range:
        start_line, end_line = view_range
        snippet_content = "\n".join(
            [
                f"{i + start_line:6}\t{line}"
                for i, line in enumerate(snippet_content.split("\n"))
            ]
        )

    return snippet_content


def extract_git_diff_local():
    """Executes and returns the `git diff` command in a local runtime environment."""
    rc = runtime_config.RuntimeConfig()
    print("extracting git diff local")
    rc.pretty_print_runtime()
    assert rc.initialized
    assert rc.runtime_type == runtime_config.RuntimeType.LOCAL

    import subprocess

    process = subprocess.Popen(
        "/bin/bash",
        cwd=rc.proj_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        shell=True,
    )
    out, err = process.communicate(
        "git -c core.fileMode=false diff --exit-code --no-color"
    )
    return out


# %%
def save_git_diff():
    print("Saving git diff")
    rc = runtime_config.RuntimeConfig()

    git_diff_output_before = extract_git_diff_local()
    instance_id = rc.proj_name.replace("/", "+")

    patch_path = (
        os.path.join(PATCH_RESULT_DIR, instance_id + "@" + str(int(time.time())))
        + ".patch"
    )

    with open(patch_path, "w", encoding="utf-8") as save_file:
        save_file.write(git_diff_output_before)
    # print(f"Saved patch content to {patch_path}")
    return git_diff_output_before


# %%
@tool
def run_shell_cmd(
    commands: Annotated[
        List[str], "A list of shell commands to be run in sequential order"
    ],
    config: RunnableConfig,
) -> str:
    """Run a list of shell commands in sequential order and return the stdout results, your working directory is the root of the project"""

    proj_path = config.get("configurable", {}).get("proj_path")
    if proj_path is None:
        rc = runtime_config.RuntimeConfig()
        assert rc.initialized
        proj_path = rc.proj_path
        print(f"use global runtime config project path: {proj_path}")
    else:
        print(f"use configrable config project path: {proj_path}")

    if rc.runtime_type == runtime_config.RuntimeType.LOCAL:
        import subprocess

        process = subprocess.Popen(
            "/bin/bash",
            cwd=proj_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
        )
        out, err = process.communicate("\n".join(commands))
        return out

    else:
        raise NotImplementedError


if __name__ == "__main__":
    runtime_config = runtime_config.RuntimeConfig()
    runtime_config.load_from_github_issue_url(
        "https://github.com/gitpython-developers/GitPython/issues/1977"
    )


# %%
