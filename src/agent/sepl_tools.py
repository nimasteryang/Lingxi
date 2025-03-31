import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Annotated, List, Optional
import ast
from git import Repo
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from agent import runtime_config
from agent.runtime_config import RuntimeType
from agent.constant import PATCH_RESULT_DIR, RUNTIME_DIR

MAX_LIST_FILES = 50  # the maximum number of files to return
MAX_RESPONSE_LEN_CHAR: int = 32000

def prepare_input_dir(in_dir):
    rc = runtime_config.RuntimeConfig()
    assert rc.initialized

    if in_dir == ".":
        in_dir = ""
    return os.path.join(rc.proj_path, in_dir)


def prepare_output_dir(out_dir):
    rc = runtime_config.RuntimeConfig()
    assert rc.initialized

    return out_dir.replace(rc.proj_path + "/", "")

@tool
def search_files_by_keywords(directory: str, keywords: list[str]):
    """
    Recursively searches for files in the given directory (including subdirectories) that contain the specified regex patterns or keywords in their filename or content using ripgrep for faster performance.

    Args:
        directory (str): The root directory to search in, can also be a file path if you want to search keywords within a single file.
        keywords (list[str]): A list of regex patterns to look for, each pattern should be at least three characters long. Patterns are treated as regex by default.

    Returns:
        dict: A dictionary where each keyword/pattern maps to a list of file paths and their corresponding line ranges that match the search.
    """
    
    # Input validation
    if not keywords:
        return "ArgumentError: The keywords must be a non-empty list"
    
    if len(keywords) > 10:
        return "ArgumentError: The number of keywords must be less than 10"
    
    for keyword in keywords:
        if len(keyword) < 3:
            return f"ArgumentError: The keyword '{keyword}' must be at least 3 characters long"
    
    # Normalize the directory path - using the original prepare_input_dir function
    directory = prepare_input_dir(directory)
    
    # Check if ripgrep is installed
    try:
        subprocess.run(["rg", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return "Error: ripgrep is not installed or not available in PATH. Please install ripgrep for this tool to work."
    
    # Make sure directory exists
    if not os.path.exists(directory):
        return f"Error: Directory or file '{directory}' does not exist"
    
    # Initialize results dictionary
    results = {}
    
    # Search for each keyword
    for keyword in keywords:
        print(f"Searching for regex pattern: {keyword}")
        matching_files = []
        
        try:
            # Build ripgrep command for content search
            rg_cmd = ["rg", "--line-number"]
            
            # Add filename matching capability (search in filenames too)
            if os.path.isfile(directory):
                # Check if the filename contains the keyword
                if keyword in os.path.basename(directory):
                    matching_files.append(prepare_output_dir(directory))
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
                        matching_files.append(prepare_output_dir(file_path))
                except subprocess.SubprocessError as e:
                    print(f"Error searching filenames: {e}")
            
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
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        file_path, line_number = parts[0], int(parts[1])
                        
                        if file_path not in file_line_map:
                            file_line_map[file_path] = {"first": line_number, "last": line_number}
                        else:
                            file_line_map[file_path]["last"] = line_number
                except Exception as e:
                    print(f"Error processing line '{line}': {e}")
            
            # Format results
            for file_path, line_info in file_line_map.items():
                output_path = prepare_output_dir(file_path)
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
        print(f"Search for '{keyword}' found {len(matching_files)} results")
    
    return results

@tool
def search_files_by_name_or_content(directory, substring):
    """
    Recursively searches for files in the given directory (including subdirectories)
    that contain the specified substring in their filename or content.

    Args:
        directory (str): The root directory to search in.
        substring (str): The substring to look for, should at least be three characters long like 'abc'

    Returns:
        list: A list of file paths that match the search criteria.
    """
    if len(substring) < 3:
        return "Failed. The substring must be at least 3 characters long"

    print(
        'search_files_by_name_or_content: path:%s dir="%s" substring:%s'
        % (runtime_config.RuntimeConfig().proj_path, directory, substring)
    )

    directory = prepare_input_dir(directory)
    matching_files = []

    # if directory is a file, search for the substring in the file
    if os.path.isfile(directory):
        if substring in directory:
            matching_files.append(prepare_output_dir(directory))
        try:
            with open(directory, encoding="utf-8", errors="ignore") as f:
                line_number = 0
                first_occurrence_line_number = 0
                last_occurrence_line_number = 0
                found_substring = False
                for line in f:
                    line_number += 1
                    if substring in line:
                        found_substring = True
                        if first_occurrence_line_number == 0:
                            first_occurrence_line_number = line_number
                        last_occurrence_line_number = line_number
                if found_substring:
                    if first_occurrence_line_number == last_occurrence_line_number:
                        matching_files.append(
                            prepare_output_dir(directory) + f": line {first_occurrence_line_number}"
                        )
                    else:
                        matching_files.append(
                            prepare_output_dir(directory)
                            + f": line {first_occurrence_line_number}-{last_occurrence_line_number}"
                        )
        except Exception as e:
            print(f"Skipping {directory}: {e}")

    else:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)

                # Check if the filename contains the substring
                if substring in file:
                    matching_files.append(prepare_output_dir(file_path))
                    continue  # No need to check content if filename matches

                # Check if the file content contains the substring
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        line_number = 0
                        first_occurrence_line_number = 0
                        last_occurrence_line_number = 0
                        found_substring = False
                        for line in f:
                            line_number += 1
                            if substring in line:
                                found_substring = True
                                if first_occurrence_line_number == 0:
                                    first_occurrence_line_number = line_number
                                last_occurrence_line_number = line_number
                        if found_substring:
                            if first_occurrence_line_number == last_occurrence_line_number:
                                matching_files.append(
                                    prepare_output_dir(file_path) + f": line {first_occurrence_line_number}"
                                )
                            else:
                                matching_files.append(
                                    prepare_output_dir(file_path)
                                    + f": line {first_occurrence_line_number}-{last_occurrence_line_number}"
                                )
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    print("search_files_by_name_or_content Input: %s Output:[%s]" % (substring, len(matching_files)))

    if len(matching_files) > MAX_LIST_FILES:
        return [
            f"Note: Too many files found. Only the first {MAX_LIST_FILES} files are returned, please narrow down your search"
        ] + matching_files[:MAX_LIST_FILES]
    return matching_files


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
    if rc.runtime_type == RuntimeType.LOCAL:
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
        filtered_files = [path for path, d in all_files if (max_depth is None) or (d <= max_depth)]
        # Filter directories
        filtered_dirs = [dir_path for dir_path, d in all_dirs if (max_depth is None) or (d <= max_depth)]
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
        else max(max((d for _, d in all_files), default=0), max((d for _, d in all_dirs), default=0))
    )

    for d in range(start_depth, -1, -1):
        adjusted_entries = filter_entries(d)
        if len(adjusted_entries) <= 50:
            print(f"Note: Reduced depth to {d} with {len(adjusted_entries)} entries")
            return [f"Note: Reduced depth to {d} with {len(adjusted_entries)} entries"] + adjusted_entries

    # Fallback (depth 0)
    final_entries = filter_entries(0)
    print(f"Note: Limited to depth 0 with {len(final_entries)} entries")
    return [f"Note: Limited to depth 0 with {len(final_entries)} entries"] + final_entries


@tool
def view_directory_path(path: Annotated[str, "File path relative to git root"]) -> List[str]:
    """View the file structure of the repository at path"""

    # code.interact('view_directory_path', local=dict(locals(), **globals()))
    rc = runtime_config.RuntimeConfig()
    assert rc.initialized

    # code.interact('view_directory_path', local=dict(locals(), **globals()))

    repo = Repo(rc.proj_path)
    repo_root = repo.working_tree_dir  # Get the root of the repository

    absolute_path = prepare_input_dir(path)

    if not os.path.exists(absolute_path):
        print("The path '{path}' does not exist in the repository.")
        raise ValueError(f"The path '{path}' does not exist in the repository.")

    fnames = os.listdir(absolute_path)
    ret = []
    for fname in fnames:
        ret.append(prepare_output_dir(os.path.join(absolute_path, fname)))

    # import code
    # code.interact("view_directory Path:%s Abspath:%s Output:%s" % (path, absolute_path, ret), local=dict(locals(), **globals()))

    # List and return the files and directories in the target path
    return ret


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
    print('view_file_content: path:%s file_name="%s" view_range=%s' % (rc.proj_path, file_name, view_range))
    if rc.runtime_type == RuntimeType.LOCAL:
        full_file_path = os.path.join(rc.proj_path, file_name)
        if not os.path.isfile(full_file_path):
            raise ValueError(f"file_name: '{file_name}' doesn't exist!")
        with open(full_file_path, encoding="utf-8") as f:
            lines = f.readlines()
            if view_range:
                start_line, end_line = view_range
                lines = lines[start_line - 1 : end_line]
                lines = [f"{i + start_line}\t{line}" for i, line in enumerate(lines)]
            else:
                lines = [f"{i + 1}\t{line}" for i, line in enumerate(lines)]
            file_content = "".join(lines)
    else:
        raise NotImplementedError

    # FILE_CONTENT_TRUNCATED_NOTICE = '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with the `search_file_by_keywords` tool or `view_file_structure` tool in order to find the line numbers of what you are looking for, and then use this tool with view_range.</NOTE>'
    FILE_CONTENT_TRUNCATED_NOTICE = '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with the `search_file_by_keywords` tool or view the file structure below in order to find the line numbers of what you are looking for, and then use this tool with view_range.</NOTE>'
    if len(file_content) > MAX_RESPONSE_LEN_CHAR:
        truncated = True
    else:
        truncated = False
    snippet_content = file_content if not truncated else file_content[:MAX_RESPONSE_LEN_CHAR] + FILE_CONTENT_TRUNCATED_NOTICE
    snippet_content = snippet_content.expandtabs()

    if view_range:
        start_line, end_line = view_range
        snippet_content = "\n".join(
            [f"{i + start_line:6}\t{line}" for i, line in enumerate(snippet_content.split("\n"))]
        )

    return snippet_content

@tool
def view_files_content(
    file_names: Annotated[
        list[str],
        "List of file names relative to git root, each file name can be retrieved by `view_directory`",
    ],
) -> dict[str, str]:
    """
    Read multiple files and return their contents.

    Args:
        file_names: A list of file names relative to git root to read.

    Returns:
        dict: A dictionary mapping file names to their contents.
    """
    results = {}

    for file_name in file_names:
        try:
            content = view_file_content(file_name)
            results[file_name] = content
        except ValueError as e:
            results[file_name] = f"Error reading file: {str(e)}"

    return results


def apply_git_diff_local(patch):
    rc = runtime_config.RuntimeConfig()
    assert rc.initialized, "Configuration is not initialized!"
    assert rc.runtime_type == RuntimeType.LOCAL

    GIT_APPLY_CMDS = [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --batch --fuzz=5 -p1",
    ]

    import subprocess

    misc_ident = str(uuid.uuid1())
    tmp_patch_name = f"tmp_patch_{misc_ident[:4]}.patch"
    tmp_f_name = Path(f"{RUNTIME_DIR}/tmp/{tmp_patch_name}")

    os.makedirs(tmp_f_name.parent, exist_ok=True)

    with open(tmp_f_name, "w", encoding="utf-8") as f:
        f.write(patch)

    for git_apply_cmd in GIT_APPLY_CMDS:
        process = subprocess.Popen(
            "/bin/bash",
            cwd=rc.proj_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
        )
        err_code, _ = process.communicate(f"cat {tmp_f_name} | {git_apply_cmd}\necho $?")
        print(err_code)
        if err_code.splitlines()[-1].strip() == "0":
            applied_patch = True
            return 0, "Patch successfully applied"
        else:
            print(f"Failed to apply patch to container: {git_apply_cmd}")
    return -2, "Patch failed to apply"


def apply_git_diff(patch):
    rc = runtime_config.RuntimeConfig()
    assert rc.initialized, "Configuration is not initialized!"

    if rc.runtime_type == RuntimeType.LOCAL:
        return apply_git_diff_local(patch)
    else:
        raise NotImplementedError

def extract_git_diff_local():
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
    out, err = process.communicate("git -c core.fileMode=false diff --exit-code --no-color")
    return out


# %%
def save_git_diff():
    print("Saving git diff")
    rc = runtime_config.RuntimeConfig()

    
    git_diff_output_before = extract_git_diff_local()
    instance_id = rc.proj_name.replace("/", "+")

    patch_path = os.path.join(PATCH_RESULT_DIR, instance_id + "@" + str(int(time.time()))) + ".patch"

    with open(patch_path, "w", encoding="utf-8") as save_file:
        save_file.write(git_diff_output_before)
    # print(f"Saved patch content to {patch_path}")
    return git_diff_output_before


# %%
@tool
def run_shell_cmd(
    commands: Annotated[List[str], "A list of shell commands to be run in sequential order"],
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
        out, err = process.communicate("\n".join(commands))
        return out

    else:
        raise NotImplementedError
if __name__ == "__main__":
    rc = runtime_config.RuntimeConfig()
    rc.load_from_github_issue_url("https://github.com/gitpython-developers/GitPython/issues/1977")

    # write a test for get_file_signature with above config
    # print(view_directory({'dir_path': './sphinx/'}))
    # print(view_file_content({'file_name': 'django/forms/fields.py'}))
    print("-" * 50)
    print(view_file_structure({'file_path': 'django/forms/fields.py'}))
    
    # print(view_file_content({'file_name': 'sphinx/roles.py', 'view_range': [200, 400]}))
    # print(view_file_structure({'file_path': 'sphinx/roles.py'}))

    # print("=" * 50)
    # rc.pretty_print_runtime()
    # print("=" * 50)
    # regex = r"viewcode(_enable_epub)?\b"
    # print("=" * 50)
    # print(search_files_by_keywords({'directory': './', 'keywords': [regex]}))

    # print(apply_git_diff(PLACE_HOLDER_PATCH))

    # print(extract_git_diff_local())

# %%
