import os

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

RUNTIME_DIR = os.path.join(os.environ["HOME"], "Tmp", "swe-runtime")

DOCKER_MAP_DIR = os.path.join(RUNTIME_DIR, "docker_map")
os.makedirs(DOCKER_MAP_DIR, exist_ok=True)

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets/")
PRESET_OPTIONS = os.listdir(PRESET_DIR)

MEMBERS = ["problem_decoder", "solution_mapper", "problem_solver", "reviewer"]

CHECKPOINTER_DB = os.path.join(RUNTIME_DIR, "checkpointer.db")

PATCH_RESULT_DIR = os.path.join(RUNTIME_DIR, "results")
os.makedirs(PATCH_RESULT_DIR, exist_ok=True)

MEMORY_CONFIG = {"configurable": {"thread_id": "1"}}

SPRING_FRAMEWORK_ISSUES = """
There is a new issue report in the project located at "/home/xuyang/Tmp/swe-runtime/spring-projects/spring-framework". Here is the original issue report:
WebTestClient has a special case when a Void response is expected. Method WebTestClient.ResponseSpec#returnResult(java.lang.Class<T>) has this documentation comment:

    Note that when {@code Void.class} is passed in, the response body is consumed and released.

The same special case logic is not applied to the overloaded version of this method: WebTestClient.ResponseSpec#returnResult(ParameterizedTypeReference<T>). This is a bit unexpected; and caused a memory leak in my tests when I swapped a Class parameter for a ParameterizedTypeReference parameter. The following sample code shows the problem (additional GC is forced to make the leak happen faster; netty's leak detection happens on GC):

import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.netty.http.client.HttpClient;

class Scratch {
    public static void main(String[] args) throws Exception {
        var client = WebTestClient.bindToServer(new ReactorClientHttpConnector(HttpClient.create()))
                .build();
        var b = client.get().uri("http://localhost:8000");
        long count = 0;
        while(true) {
            count += 1;
            if (count % 10000 == 0) {
                System.out.printf("Done %d requests forcing a G\n", count);
                System.gc();
            }
            b.exchange()
                    .expectStatus().isEqualTo(200)
                    //.returnResult(Void.class);                                       // <-- doesn't leak
                    .returnResult(ParameterizedTypeReference.forType(Void.class)); // <-- leaks
        }
    }
}

The workaround is to use the Class signature, or consume the response body some other way using e.g. by specifying .consumeWith(...).
"""

GIT_PYTHON_ISSUES = """
There is a new issue report created on 2021-05-04T10:05:33Z in the project located at "/home/xuyang/Tmp/swe-runtime/gitpython-developers/GitPython". Here is the original issue report:

Get patch text from Commit class like (commit.stats)

Current we can get status for the commit (comparing parent). But can't get patch text.

I would like to have commit.patch to get patch text. Result will be similar to git diff shell command.

```shell
git diff <c1> <c2>
```

```python
all_commits = repo.iter_commits('master')
for commit in all_commits:
    print(commit.stats)
```

If the community is okay with this addition, i can work on this and raise a PR.

Thanks in advance,
Durai Pandian

"""

GIT_PYTHON_ISSUES_MK = f"""There is a new issue report created on 2021-05-04T10:05:33Z in the project located at "{RUNTIME_DIR}/gitpython-developers/GitPython". Here is the original issue report:

```
Description:
I am trying to create or open a repo while also changing the .git default path to custom name like .custom_git. However, I a meeting an exception.

Example Code
from pathlib import Path

import git

repo_path = Path("my_repo")
custom_git_dir = Path(".custom_git")

os.environ["GIT_DIR"] = str(custom_git_dir)

repo = git.Repo.init(repo_path)
Error
Traceback (most recent call last):
  File "pythonProject\\main.py", line 11, in <module>
    repo = git.Repo.init(repo_path)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pythonProject\\.venv\\Lib\\site-packages\\git\\repo\\base.py", line 1329, in init
    return cls(path, odbt=odbt)
           ^^^^^^^^^^^^^^^^^^^^
  File "pythonProject\\.venv\\Lib\\site-packages\\git\\repo\\base.py", line 289, in __init__
    raise InvalidGitRepositoryError(epath)
git.exc.InvalidGitRepositoryError: pythonProject\temp
The directory .custom_git gets created but the error happens. The same exception is raised when trying git.Repo(repo_path) instead of initialization with .innit(repo_path).
```
"""

GIT_PYTHON_ISSUES_XY = """
There is a new issue report created on 2021-05-04T10:05:33Z in the project located at "/home/xuyang/Tmp/swe-runtime/gitpython-developers/GitPython". Here is the original issue report:

```
Description:
I am trying to create or open a repo while also changing the .git default path to custom name like .custom_git. However, I a meeting an exception.

Example Code
from pathlib import Path

import git

repo_path = Path("my_repo")
custom_git_dir = Path(".custom_git")

os.environ["GIT_DIR"] = str(custom_git_dir)

repo = git.Repo.init(repo_path)
Error
Traceback (most recent call last):
  File "pythonProject\\main.py", line 11, in <module>
    repo = git.Repo.init(repo_path)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pythonProject\\.venv\\Lib\\site-packages\\git\\repo\\base.py", line 1329, in init
    return cls(path, odbt=odbt)
           ^^^^^^^^^^^^^^^^^^^^
  File "pythonProject\\.venv\\Lib\\site-packages\\git\\repo\\base.py", line 289, in __init__
    raise InvalidGitRepositoryError(epath)
git.exc.InvalidGitRepositoryError: pythonProject\temp
The directory .custom_git gets created but the error happens. The same exception is raised when trying git.Repo(repo_path) instead of initialization with .innit(repo_path).
"""

base_path = os.path.join(os.path.dirname(__file__), "samples")


def load_file_content(f_name):
    with open(os.path.join(base_path, f_name), encoding="utf-8") as f:
        content = f.read()
    return content


CVE_LOAD_DICT = {
    "CVE38821": "CVE-2024-38821_openai.md",
    "CVE47685": "CVE-2024-47685_openai.md",
    "CVE48208": "CVE-2024-48208_openai.md",
    "CVE44228": "CVE-2021-44228.md",
    "CVE50164": "CVE-2023-50164.md",
}

for var_name, file_name in CVE_LOAD_DICT.items():
    globals()[var_name] = load_file_content(file_name)


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
