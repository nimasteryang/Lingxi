import json
import os
import re
from dataclasses import dataclass
from pprint import pprint
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agent.runtime_config import RuntimeConfig


@dataclass
class ConfigPreset:
    project_name: str
    issue_id: int
    checkout_commit: str
    issue_description: str


def get_cve_id_from_preset(preset):
    if "38821" in preset:
        cve = "CVE-2024-38821"
    elif "log4j" in preset:
        cve = "CVE-2021-44228"
    elif "50164" in preset:
        cve = "CVE-2023-50164"
    else:
        assert "unknown cve", preset
    return cve


def cve_from_str(input_str):
    cve_pattern = r"CVE-\d{4}-\d{4,7}"
    return re.findall(cve_pattern, input_str)


def graph_print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            if hasattr(m, "pretty_print"):
                m.pretty_print()
            else:
                if type(m) is list and type(m[0]) is dict and "text" in m[0]:
                    pprint(m[0]["text"])
                    if len(m) > 1:
                        pprint(m[1])
                else:
                    pprint(m)


def message_processor_mk(messages):
    """Based on XY's `message_processor` with few additions. It is used for newest version of summarizer/summary."""
    message_json = {}
    step = 0

    index = 0
    found_human_feedback = False
    messages = [vars(message) if not isinstance(message, dict) else message for message in messages]
    for message in messages:
        if message["name"] == "human_feedback":
            found_human_feedback = True
            index = messages.index(message)

    if not found_human_feedback:
        index = 0

    messages = messages[index:]

    for message in messages:
        if message["type"] != "tool":
            # if message['content'] is string:
            if isinstance(message["content"], str):
                if step == 0:
                    step += 1
                    continue  # skip human input as its duplicated
                name = message["name"] if "name" in message else ""
                message_json[f"Step {step} {name}:"] = {"detail": message["content"]}
            else:
                detail_cnt = 1
                details = {}
                for content in message["content"]:
                    details[f"Detail {detail_cnt} {content['type']}:"] = content
                    detail_cnt += 1
                name = message["name"] if "name" in message else ""
                message_json[f"Step {step} {name}:"] = {"detail": details}

            step += 1
    return message_json


# project knowledge utils
function_node_types = ["function_definition", "method_declaration", "constructor_declaration"]
class_node_types = ["class_definition", "class_declaration"]
root_node_types = ["module", "program"]
class_node_types_for_space = ["class", "identifier", "void_type", "modifiers"]
comment_node_types = ["block_comment", "line_comment"]
argument_node_types = ["argument_list"]


def parse_code_tree_sitter(code: str, parser):
    return parser.parse(bytes(code, "utf8"))


def find_class_name_in_ancestors(code, node):
    """Traverse upwards in the tree to find the nearest ancestor that is a class_definition"""
    current_node = node.parent
    while current_node:
        if current_node.type in class_node_types:
            # Extract and return the class name from the class node
            for child in current_node.children:
                if child.type == "identifier":
                    return code[child.start_byte : child.end_byte]
        current_node = current_node.parent
    return None  # No class definition found in ancestors


def append_method_docstring(content, node, parent, language):
    """Append method docstring to the node content. Currently only needed/supports Java as Java docstring appears outside the method."""
    if (
        language == "java"
        and parent is not None
        and parent.children.index(node) - 1 >= 0
        and parent.children[parent.children.index(node) - 1].type in comment_node_types
    ):
        return parent.children[parent.children.index(node) - 1].text.decode() + "\n" + content
    return content


def extract_code_tags(code, parser, language):
    """Tag code into for categories: global code, functions, classes, and methods."""
    tree = parse_code_tree_sitter(code, parser)
    root_node = tree.root_node
    chunks = {
        "global_code": "",
        "functions": {},
        "classes": {},
        "methods": {},
    }
    split_code = code.split("\n")

    def traverse(node, parent=None):
        node_type = node.type

        # Check if the node is a function definition
        if node_type in function_node_types:
            function_name = node.child_by_field_name("name").text.decode()

            # Check if this function is inside a class (using ancestor search)
            class_name = find_class_name_in_ancestors(code, node)
            chunk_key = ""
            func_key = ""
            if class_name:  # If class name was found, add to appropriate chunk dict key
                chunk_key = "methods"
                func_key = f"{class_name}:{function_name}()"
            else:
                chunk_key = "functions"
                func_key = f"{function_name}()"

            func_code_block = "\n".join(split_code[node.start_point[0] : node.end_point[0] + 1]).strip()

            # check for and append docstring
            func_code_block = append_method_docstring(func_code_block, node, parent, language)

            chunks[chunk_key][func_key] = func_code_block

        # Check if the node is a class definition # Also avoid individually indexing nested classes (check parent type)
        elif node_type in class_node_types and (parent is None or parent and parent.type not in "block"):
            class_name = node.child_by_field_name("name").text.decode()

            # Collect the class body (excluding methods)
            class_code_lines = split_code[node.start_point[0] : node.end_point[0] + 1]
            class_methods = [c for c in node.children if c.type in function_node_types]
            class_body_code = ""

            if language != "java":  # Add spaces between the text depending on the node type
                for i, c in enumerate(node.children[:-1]):
                    class_body_code += c.text.decode()
                    if (
                        c.type in class_node_types_for_space
                        and node.children[i + 1].type not in argument_node_types
                    ):
                        class_body_code += " "
            class_body_code = class_body_code.strip() + "\n"

            # Only include class-level code, not methods
            for i, c in enumerate(node.children[-1].children):
                if c.type not in function_node_types + ["decorated_definition"]:  # This excludes the methods
                    if (
                        language == "java"
                        and i + 1 < len(node.children[-1].children) - 1
                        and c.type == "block_comment"
                        and node.children[-1].children[i + 1].type in function_node_types
                    ):  # Skip method docstrings
                        continue
                    class_body_code += "\n".join(split_code[c.start_point[0] : c.end_point[0] + 1]) + "\n"

            # If there is class-level code (non-method), store it
            if class_body_code:
                # check for and append docstring
                class_body_code = append_method_docstring(class_body_code, node, parent, language)

                chunks["classes"][class_name] = class_body_code.strip()

            # Handle class methods separately (they should go in methods_in_classes)
            for method in class_methods:
                method_name = None
                for child in method.children:
                    if child.type == "identifier":
                        method_name = code[child.start_byte : child.end_byte]
                        break

                method_key = f"{class_name}:{method_name}()"
                method_code = "\n".join(split_code[method.start_point[0] : method.end_point[0] + 1])

                # check for and append docstring
                method_code = append_method_docstring(
                    method_code, method, node, language
                )  # In this case node is  In this case node is the parentthe parent

                chunks["methods"][method_key] = method_code.strip()

        # If it's top-level code outside of a function or class (global code)
        elif (
            parent is not None
            and parent.type in root_node_types
            and node.type not in function_node_types + class_node_types + root_node_types
        ):
            # Only include top-level code outside of functions or classes
            chunks["global_code"] += (
                "\n".join(split_code[node.start_point[0] : node.end_point[0] + 1]) + "\n"
            )

        # Recursively traverse the child nodes
        for child in node.children:
            traverse(child, node)

    # Start traversing the tree from the root node
    traverse(root_node)

    chunks["global_code"] = chunks["global_code"].strip()

    return chunks


def calculate_stage_messages(state, key="previous_messages_len"):
    assert key in state, f"{key} not found in state var"
    previous_messages_len = state.get(key)
    return len(state["messages"]) - previous_messages_len


def setup_debug_logger(filename="log.txt"):
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    # Step 3: Create a file handler that writes log messages to a file
    file_handler = logging.FileHandler("log.txt")  # You can specify your file name here
    # Step 4: Create a log formatter to specify the format of the log messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Step 5: Attach the formatter to the handler
    file_handler.setFormatter(formatter)
    # Step 6: Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


def dump_langgraph_config(config_dict, meta_info, type_, save_directory):
    """
    config_dict is the langgraph config
    type_ is the type of this particular save. Expected to send the graph name
    save_directory is the root of the logging directory.
    """
    os.makedirs(os.path.join(save_directory, type_), exist_ok=True)

    config_dict["meta_info"] = meta_info

    assert "configurable" in config_dict
    assert "thread_id" in config_dict["configurable"]

    store_file_path = os.path.join(save_directory, type_, config_dict["configurable"]["thread_id"] + ".json")

    assert not os.path.isfile(store_file_path)

    with open(store_file_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    print(f"Config dumped to:\n\t{store_file_path}")


def ensure_config(state):
    """
    helper function for restoring a runtimestate
    """
    print("*" * 30)
    rc = RuntimeConfig()
    if not rc.initialized:
        print("*" * 30)
        print("Configuration not setup detected!")
        print("Loading using state storted `runtime_info`")
        pprint(state["runtime_info"])
        rc.load_from_dump_config(state["runtime_info"])

