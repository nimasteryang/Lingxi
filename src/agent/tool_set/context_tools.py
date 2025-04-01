from glob import glob
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from agent import runtime_config
from agent.constant import func_queries, query_java_construcor_decs, tree_sitter_parsers
from agent.llm import llm
from agent.parsers import relevant_file_explanations_parser
from agent.prompt import (
    RELEVANT_FILE_EXPLANATION_SYSTEM_PROMPT,
)
from agent.runtime_config import load_env_config
from agent.utils import UndefinedValueError

load_env_config()

if "OPENAI_API_KEY" not in os.environ:
    raise UndefinedValueError("OPENAI_API_KEY")

EMBEDDING_FUNCTION = OpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-small",
)
PROJECT_KNOWLEDGE_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=256, separators=["\n"]
)


def create_project_knowledge(
    project_dir: str,
    collection_name="project_knowledge_db",
    file_types=("*.java", "*.py"),
    batch_size=1000,
):
    """Creates the Project Knowledge component. Indexes all Python files in the given directory.

    Args:
        project_dir: The path of the project to index.
    """

    print(f"Creating project knowledge for {project_dir!r}")
    repo = (
        project_dir.split("/")[-1]
        if not project_dir.endswith("/")
        else project_dir.split("/")[-2]
    )
    rc = runtime_config.RuntimeConfig()

    persist_directory = os.path.join(rc.runtime_dir, collection_name + "_" + repo)

    print(f"{persist_directory=}")
    if os.path.isdir(persist_directory):
        project_knowledge_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=EMBEDDING_FUNCTION,
            collection_name=collection_name,
        )
    else:
        print(f"Creating project knowledge for {repo} ({project_dir})")
        project_knowledge_db = None
        file_paths = []

        for file_type in file_types:
            file_paths += glob(
                os.path.join(project_dir, "**/" + file_type), recursive=True
            )

        total_files = len(file_paths)

        file_batches = [
            file_paths[i : i + batch_size]
            for i in range(0, len(file_paths), batch_size)
        ]
        print(
            f"Preparing to process {total_files} total files in {len(file_batches)} batches"
        )

        for file_batch_idx, file_batch in enumerate(tqdm(file_batches)):
            file_document_batch = []
            func_document_batch = []
            for file_path in tqdm(file_batch):
                with open(file_path, encoding="utf-8") as pyfile:
                    file_content = pyfile.read()

                # File processing
                relative_file_path = file_path.replace(project_dir + "/", "")
                file_document_batch.append(
                    Document(
                        page_content=file_content,
                        metadata={"file_path": relative_file_path, "type": "file"},
                    )
                )

                # Func processing
                file_type_ext = relative_file_path.split(".")[-1]
                parser = tree_sitter_parsers[file_type_ext]
                tree = parser.parse(file_content.encode())

                func_defs = (
                    func_queries[file_type_ext].captures(tree.root_node).get("defs", [])
                )
                if (
                    file_type_ext == "java"
                ):  # Java contains a "constructor_declaration" node separate from the already queried "method_declarations" nodes
                    constructor_defs = query_java_construcor_decs.captures(
                        tree.root_node
                    ).get("defs", [])
                    func_defs = constructor_defs + func_defs
                for func_def in func_defs:
                    func_content = func_def.text.decode()
                    func_name = func_def.child_by_field_name("name").text.decode()
                    func_document_batch.append(
                        Document(
                            page_content=func_content,
                            metadata={
                                "file_path": relative_file_path,
                                "func_name": func_name,
                                "type": "func",
                            },
                        )
                    )

            # Chunk the docs
            file_document_batch_split = PROJECT_KNOWLEDGE_TEXT_SPLITTER.split_documents(
                file_document_batch
            )
            func_document_batch_split = PROJECT_KNOWLEDGE_TEXT_SPLITTER.split_documents(
                func_document_batch
            )

            print(
                f"Inserting {len(file_document_batch_split)} file and {len(func_document_batch_split)} func chunked documents for batch {file_batch_idx}"
            )
            # Insert chunked docs Chroma
            if project_knowledge_db is None:
                project_knowledge_db = Chroma.from_documents(
                    file_document_batch_split + func_document_batch_split,
                    EMBEDDING_FUNCTION,
                    collection_name=collection_name,
                    persist_directory=persist_directory,
                )
            else:
                project_knowledge_db.add_documents(
                    file_document_batch_split + func_document_batch_split
                )

    # Retrieve docs for log
    total_files, total_funcs = (
        len(project_knowledge_db.get(where={"type": "file"})["ids"]),
        len(project_knowledge_db.get(where={"type": "func"})["ids"]),
    )
    print(
        f"Connected to DB {persist_directory}:{collection_name} containing {total_files} total files and {total_funcs} total func documents."
    )

    # create VectorStoreRetriever
    project_knowledge_retriever = project_knowledge_db.as_retriever()
    return project_knowledge_retriever, project_knowledge_db


@tool
def search_relevant_files(query: str, k=10):
    """Given a query search string (for example, the issue report description, filenames, etc), search for relevant code snippets of files in the project by calculating embedding similarity between the query and code snippets in a vector database.

    Args:
        query: A search string (for example, the issue report description, filenames, etc), to be used to find relevant files and functions.
    """
    rc = runtime_config.RuntimeConfig()

    project_knowledge_retriever, _ = create_project_knowledge(rc.proj_path)

    relevant_docs = project_knowledge_retriever.get_relevant_documents(query, k=k)

    full_result = []
    return_string = f"Top {k} most relevant files: \n\n"
    print("-----RELEVANT DOCS-----")
    for doc in relevant_docs:
        return_string += doc.metadata["file_path"] + "\n"
        # if "func_name" in doc.metadata and doc.metadata["type"] == "func":
        if "name" in doc.metadata:
            full_result.append(
                {
                    # "file_path": doc.metadata["file_path"] + ":" + doc.metadata["name"] + "()",
                    "file_path": doc.metadata["file_path"] + ":" + doc.metadata["name"],
                    "code_snippet": doc.page_content,
                }
            )

        else:
            full_result.append(
                {
                    "file_path": doc.metadata["file_path"],
                    "code_snippet": doc.page_content,
                }
            )

    return_string = return_string.strip()

    explain_prompt = RELEVANT_FILE_EXPLANATION_SYSTEM_PROMPT.substitute(
        search_term=query, k=k, full_result=full_result
    ).strip()
    generate_explanation = llm.invoke([HumanMessage(explain_prompt)])

    explanations = relevant_file_explanations_parser.invoke(
        generate_explanation.content
    )
    return explanations


def summarizer(stage_msgs_processed):
    """Summarize the information of previous chat history to gain addtiional information or remember what you were doing."""
    stage_message_keys = list(stage_msgs_processed.keys())
    stage_messages = {}
    for k in stage_message_keys:
        stage_messages[k] = stage_msgs_processed[k]

    summary_prompt = f"Summarize the messages in the following conversation. Be sure to include aggregated details of the key steps and or goals of the message. Include the names of the agents and tools features in the steps. If the agent did not describe its process but used a tool mention the used tool(s). Also include any raw content such as problem statements, solution plans, generated code and or patches if applicable. Be sure to only output your result. Here are the message(s):\n```{stage_messages}\n\nHere is an example of the result:\n```\nStep 1: The user submitted the issue to be resolved.\nStep 2. The supervisor delegated the task to the problem_decoder\nStep 3. The problem_decoder asked the context_manager for help\nStep 4. The context_manager searched for relevant files in the codebase, including file1.py, file2.py.\nStep 5. The context_manager viewed the file file5.py using `view_file_content`.```"

    response = llm.invoke(
        [
            HumanMessage(
                summary_prompt.strip(),
                name="summarizer_agent",
            )
        ]
    )

    summary = response.content

    return summary
