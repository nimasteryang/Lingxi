# %%
import glob
import os
import re

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# %%


class RepoKnowledge:
    def __init__(self):
        self.build_dirs = ["build"]
        self.test_dirs = ["test"]
        self.PY_LANGUAGE = Language(tspython.language())
        self.JAVA_LANGUAGE = Language(tsjava.language())

        self.parsers = {
            "py": Parser(self.PY_LANGUAGE),
            "java": Parser(self.JAVA_LANGUAGE),
        }

        self.java_class_query = self.JAVA_LANGUAGE.query(
            """(class_declaration
            (modifiers) @class.mods
            name: (identifier) @class.def
            body: (class_body) @class.body
        )"""
        )
        self.java_comment_types = ["block_comment", "line_comment"]
        self.java_method_query = self.JAVA_LANGUAGE.query("(method_declaration) @method.dec")
        self.java_method_details = self.JAVA_LANGUAGE.query(
            """
            (modifiers) @method.mods
            (void_type) @method.void_type
            parameters: (formal_parameters) @method.args
            body: (block) @method.block
        """
        )

    def retrieve_signatures_java(self, file_path, content, tree):
        file_map = {}
        func_map = {}
        class_results = self.java_class_query.captures(tree.root_node)  # Get all classes
        for i, class_result in enumerate(class_results["class.body"]):  # For each class
            class_mods = class_results["class.mods"][i].text.decode().replace("\n", "").strip()
            class_def = class_results["class.def"][i].text.decode().replace("\n", "").strip()
            class_body = class_results["class.body"][i]
            class_comment = ""
            for child in class_body.children:
                if child.type in self.java_comment_types:
                    class_comment = child.text.decode().replace("\n", "").strip()
                    break
            class_signature = f"{class_mods} class {class_def} {class_comment}".replace(
                "\n", ""
            ).strip()  # Classes don't have args in Java but their constructors do. I am not sure if we should get these constructor params here or just define construcotrs as methods @Shawn?
            method_results = self.java_method_query.captures(class_body)
            class_method_signatures = []
            for j in range(len(method_results["method.dec"])):
                method_result = method_results["method.dec"][j]
                method_details = self.java_method_details.captures(method_result)

                method_body = method_details["method.block"][0]
                method_mods = method_details["method.mods"][0].text.decode().replace("\n", "").strip()
                method_type = (
                    method_details["method.void_type"][0].text.decode().replace("\n", "").strip()
                    if "method.void_type" in method_details
                    else ""
                )
                method_args = method_details["method.args"][0].text.decode().replace("\n", "").strip()
                method_def = method_result.child_by_field_name("name").text.decode()
                method_comment = ""
                for child in method_body.children:
                    if child.type in self.java_comment_types:
                        method_comment = child.text.decode().replace("\n", "").strip()
                        break

                method_signature = f"function {method_mods} {method_type} {method_def}{method_args} {method_comment}".strip()
                class_method_signatures.append(method_signature)
            func_map[class_signature] = class_method_signatures
        file_map[file_path] = content
        return file_map, func_map

    def create_map(self, repo_path, lang="py"):
        self.repo_path = repo_path if repo_path.endswith("/") else repo_path + "/"
        self.lang = lang
        parser = self.parsers[self.lang]
        if lang == "py":
            self.function_map, self.file_map = self.map_file_to_class_and_functions(self.repo_path)
            return self.function_map, self.file_map
        else:
            file_map = {}
            func_map = {}
            for file_path in glob.glob(repo_path + "**/*." + self.lang, recursive=True):
                with open(file_path, "r+", encoding="latin-1") as file:
                    content = file.read()
                tree = parser.parse(content.encode())
                i_file_map, i_func_map = self.retrieve_signatures_java(file_path, content, tree)
                func_map.update(i_func_map)
                file_map.update(i_file_map)
            return func_map, file_map

    def extract_docstring(self, text):
        """EXtracts the docstring from a given text."""
        text = text.decode()
        docstring = ""
        lines = text.split("\n")

        first_i = -1
        for line_i, line in enumerate(lines):
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                first_i = line_i
                break

        lines = lines[first_i:]

        if len(lines) > 1:
            if lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''"):
                docstring = lines[0].strip()[3:]
                for line in lines[1:]:
                    if line.strip().startswith('"""') or line.strip().startswith("'''"):
                        docstring += line.strip()[:-3]
                        break
                    docstring += line.strip()
        if docstring.strip():
            return "#" + docstring.strip().replace("\n", "")
        return ""

    def get_class_and_function_signatures(self, source_code):
        """Extracts sclass and function definitions with signatures and docstirngs from"""
        tree = self.parsers["py"].parse(source_code.encode())
        root_node = tree.root_node
        classes_and_functions = []

        def extract_function_signature(node):
            """EXtract function name, signature, and dcstring if abailable."""
            func_name = node.child_by_field_name("name").text.decode()
            parmas_node = node.child_by_field_name("parameters")
            params = parmas_node.text.decode()
            params = re.sub(r"\s+", " ", params)

            # Extract docstring if present
            docstring = self.extract_docstring(node.text)
            func_signature = f"function {func_name}{params} {docstring}".replace("\n", "")
            return func_signature

        def extract_class_content(node):
            """EXtracts all functiosn iwthin a class, including their signaturtes and docstrings"""
            class_content = []
            block_node = node.children[-1]
            reparsed = self.parsers["py"].parse(
                source_code[block_node.start_byte : block_node.end_byte].encode()
            )

            # Traverse class node's children to find methods
            for child in reparsed.root_node.children:
                if child.type == "function_definition":
                    func_signature = extract_function_signature(child)
                    class_content.append(func_signature)
            return class_content

        for node in root_node.children:
            if node.type == "class_definition":
                class_name = node.child_by_field_name("name").text.decode()
                params_node = node.child_by_field_name("superclasses")
                params = params_node.text.decode() if params_node is not None else ""
                params = re.sub(r"\s+", " ", params)

                first_func_i = node.text.find(b"def")
                docstring = self.extract_docstring(node.text[:first_func_i])

                # Extract functiosn within the class
                class_content = extract_class_content(node)
                class_signature = f"class {class_name}{params} {docstring}".replace("\n", "")
                classes_and_functions.append((class_signature, class_content))

            elif node.type == "function_definition":
                func_signature = extract_function_signature(node)
                classes_and_functions.append(func_signature)

        return classes_and_functions

    def map_file_to_class_and_functions(self, directory_path):
        """MAp each Python file in dir to its list of class and function definitiions"""
        file_structure_map = {}
        file_content_map = {}

        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(file_path, directory_path)
                if any([relative_path.startswith(build_dir) for build_dir in self.build_dirs]):
                    continue
                if file_name.endswith(".py"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        source_code = file.read()

                    class_and_function_signatures = self.get_class_and_function_signatures(source_code)
                    if class_and_function_signatures:
                        file_structure_map[relative_path] = class_and_function_signatures
                    file_content_map[relative_path] = source_code
        return file_structure_map, file_content_map


# %%
if __name__ == "__main__":
    repo_knowledge = RepoKnowledge()
    example_file = "/Users/xuyang/Documents/GitHub/codexray/src/agent/tool_set/oheditor.py"
    with open(example_file, "r", encoding="utf-8") as file:
        source_code = file.read()
    class_and_funcs =   repo_knowledge.get_class_and_function_signatures(source_code)
    for class_and_func in class_and_funcs:
        print(class_and_func)
        for func in class_and_func:
            print(func)
            print("-"*100)
    # file_structure_map, file_content_map = repo_knowledge.map_file_to_class_and_functions('/Users/xuyang/Documents/GitHub/codexray/src/agent/tool_set')
    # print(file_structure_map)
    # for content in file_structure_map['oheditor.py']:
        # print(content)
        # print("-"*100)
    # print(file_content_map['edit_tool.py'])

    # PY_LANGUAGE = Language(tspython.language())
    # JAVA_LANGUAGE = Language(tsjava.language())

    # parsers = {
    #     "py": Parser(PY_LANGUAGE),
    #     "java": Parser(JAVA_LANGUAGE),
    # }
    # parser = parsers["py"]
    # with open("./graph_with_context_manager.py", "r+", encoding="latin-1") as file:
    #     content = file.read()
    # tree = parser.parse(content.encode())
    # py_func_defs = PY_LANGUAGE.query(
    #     """(function_definition) @func_defs
    #     """
    # )
    # py_func_details = PY_LANGUAGE.query(
    #     """
    #         name: (identifier) @func_name
    #         parameters: (parameters) @func_args
    #         body: (block) @func_block
    #     """
    # )
    # func_defs = py_func_defs.captures(tree.root_node).get("func_defs", [])
    # print(func_defs)
    # for func_def in func_defs:
    #     func_details = py_func_details.captures(func_def)
    #     func_name = func_details.get("func_name", [])
    #     if func_name:
    #         func_name = func_name[0].text.decode()
    #     print(func_name)

    import pdb

    # pdb.set_trace()



# %%
