"""
Defines several constant variables used throughout the prototype.
These constants include configuration settings, default values, and
other immutable parameters used across different components of the system.
"""

import os

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser




RUNTIME_DIR = os.path.join(os.environ["HOME"], "Tmp", "swe-runtime")

DOCKER_MAP_DIR = os.path.join(RUNTIME_DIR, "docker_map")
os.makedirs(DOCKER_MAP_DIR, exist_ok=True)

REPO_MAP_DIR = os.path.join(RUNTIME_DIR, "repo_map")
os.makedirs(REPO_MAP_DIR, exist_ok=True)

PATCH_RESULT_DIR = os.path.join(RUNTIME_DIR, "results")
os.makedirs(PATCH_RESULT_DIR, exist_ok=True)

REQUEST_TIMEOUT = 30

# Tree-sitter parser and query definitions used for indexing
PY_LANGUAGE = Language(tspython.language())
JAVA_LANGUAGE = Language(tsjava.language())

tree_sitter_parsers = {
    "py": Parser(PY_LANGUAGE),
    "java": Parser(JAVA_LANGUAGE),
}

query_py_func_defs = PY_LANGUAGE.query(
    """(function_definition) @defs
    """
)
query_py_func_details = PY_LANGUAGE.query(
    """
        name: (identifier) @name
        parameters: (parameters) @args
        body: (block) @block
    """
)

query_java_method_decs = JAVA_LANGUAGE.query("(method_declaration) @defs")
query_java_construcor_decs = JAVA_LANGUAGE.query("(constructor_declaration) @defs")
query_java_method_details = JAVA_LANGUAGE.query(
    """
    name: (identifier) @name
    (modifiers) @mods
    (void_type) @void_type
    parameters: (formal_parameters) @args
    body: (block) @block
"""
)

func_queries = {"py": query_py_func_defs, "java": query_java_method_decs}
func_detail_queries = {"py": query_py_func_details, "java": query_java_method_details}

PLACE_HOLDER_PATCH = """diff --git a/_random_file_1bx7.txt b/_random_file_1bx7.txt
new file mode 100644
index 00000000..3372b06d
--- /dev/null
+++ b/_random_file_1bx7.txt
@@ -0,0 +1 @@
+random text fillering, no meaning
"""
