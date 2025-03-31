import copy
import datetime
import importlib
import os
from glob import glob
from string import Template
from typing import Annotated

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import InjectedState
from tqdm import tqdm

from agent import runtime_config
from agent.runtime_config import load_env_config
from agent.constant import func_queries, query_java_construcor_decs, tree_sitter_parsers
from agent.llm import llm
from agent.parsers import relevant_file_explanations_parser, relevant_message_ids_parser
from agent.prompt import (
    FILTER_CONVO_HISTORY_SYSTEM_PROMPT,
    FILTER_CONVO_HISTORY_SYSTEM_PROMPT_V2,
    RELEVANT_FILE_EXPLANATION_SYSTEM_PROMPT,
)
from agent.repo_knowledge_tools import RepoKnowledge
from agent.sepl_tools import (
    save_git_diff,
    view_directory,
    view_file_content,
)
from agent.utils import (
    extract_code_tags,
    message_processor_mk,
)

load_env_config()

EMBEDDING_FUNCTION = OpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-small",
)
PROJECT_KNOWLEDGE_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=256, separators=["\n"]
)


def convert_iso_8601_to_timestamp(date: str):
    """Convert a date string in ISO 8601 format to timestamp. ChromaDB can only do less/greater than comparisons to ints/floats."""
    return datetime.datetime.timestamp(
        datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    )


def create_project_knowledge_v2(
    project_dir: str,
    collection_name="project_knowledge_db_v2",
    file_types=("*.java", "*.py"),
    file_batch_size=1000,
    chunk_batch_size=1000,
):
    """Creates the Project Knowledge component. This version indexes all Python files in the given directory and splits code into four categories: global code, function code, class code, and method code. Processes files in batches and inserts chunked documents in batches to avoid memory issues.

    Args:
        project_dir: The path of the project to index.
    """

    repo = project_dir.split("/")[-1] if not project_dir.endswith("/") else project_dir.split("/")[-2]
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
            file_paths += glob(os.path.join(project_dir, "**/" + file_type), recursive=True)

        total_files = len(file_paths)

        file_batches = [
            file_paths[i : i + file_batch_size] for i in range(0, len(file_paths), file_batch_size)
        ]
        print(f"Preparing to process {total_files} total files in {len(file_batches)} batches")

        for file_batch_idx, file_batch in enumerate(tqdm(file_batches)):
            file_batch_tags = {
                "global_code": [],
                "functions": [],
                "classes": [],
                "methods": [],
            }

            for file_path in tqdm(file_batch):
                with open(file_path, encoding="utf-8") as pyfile:
                    file_content = pyfile.read()

                relative_file_path = file_path.replace(project_dir + "/", "")

                file_type_ext = relative_file_path.split(".")[-1]
                parser = tree_sitter_parsers[
                    file_type_ext
                ]  # Get correct parser according to language / filetype

                code_tags = extract_code_tags(file_content, parser, file_type_ext)

                for tag, result in code_tags.items():
                    if isinstance(result, dict):
                        for sub_tag, sub_tag_code in result.items():
                            if (
                                tag == "classes" and "\n" not in sub_tag_code
                            ):  # NOTE: Decision here to skip classes that have no code other than method definitions - this prevents the DB returning documents like "class TestDB(TestBase):"
                                continue
                            file_batch_tags[tag].append(
                                Document(
                                    page_content=sub_tag_code,
                                    metadata={
                                        "file_path": relative_file_path,
                                        "name": sub_tag,
                                        "type": tag,
                                    },
                                )
                            )
                    else:  # global code
                        file_batch_tags[tag].append(
                            Document(
                                page_content=result,
                                metadata={
                                    "file_path": relative_file_path,
                                    "type": tag,
                                },
                            )
                        )

            # Chunk the docs in the batch
            chunked_docs = []
            for tag, tag_code in file_batch_tags.items():
                chunked_tag_code = PROJECT_KNOWLEDGE_TEXT_SPLITTER.split_documents(tag_code)
                print(
                    f"Inserting {len(chunked_tag_code)} {tag} chunked documents for batch {file_batch_idx}"
                )
                chunked_docs.extend(chunked_tag_code)

            # Insert chunked docs Chroma
            chunked_doc_batches = [
                chunked_docs[i : i + chunk_batch_size] for i in range(0, len(chunked_docs), chunk_batch_size)
            ]

            for chunked_doc_batch in tqdm(chunked_doc_batches):
                if project_knowledge_db is None:
                    project_knowledge_db = Chroma.from_documents(
                        chunked_doc_batch,
                        EMBEDDING_FUNCTION,
                        collection_name=collection_name,
                        persist_directory=persist_directory,
                    )
                else:
                    project_knowledge_db.add_documents(chunked_doc_batch)

    # Retrieve docs for log
    db_get = project_knowledge_db.get()
    get_key = list(db_get.keys())[0]
    total_docs = len(db_get[get_key])
    print(f"Connected to DB {persist_directory}:{collection_name} containing {total_docs} docs")

    # create VectorStoreRetriever
    project_knowledge_retriever = project_knowledge_db.as_retriever()
    return project_knowledge_retriever, project_knowledge_db


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
    repo = project_dir.split("/")[-1] if not project_dir.endswith("/") else project_dir.split("/")[-2]
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
            file_paths += glob(os.path.join(project_dir, "**/" + file_type), recursive=True)

        total_files = len(file_paths)

        file_batches = [file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)]
        print(f"Preparing to process {total_files} total files in {len(file_batches)} batches")

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

                func_defs = func_queries[file_type_ext].captures(tree.root_node).get("defs", [])
                if (
                    file_type_ext == "java"
                ):  # Java contains a "constructor_declaration" node separate from the already queried "method_declarations" nodes
                    constructor_defs = query_java_construcor_decs.captures(tree.root_node).get("defs", [])
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
            file_document_batch_split = PROJECT_KNOWLEDGE_TEXT_SPLITTER.split_documents(file_document_batch)
            func_document_batch_split = PROJECT_KNOWLEDGE_TEXT_SPLITTER.split_documents(func_document_batch)

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
                project_knowledge_db.add_documents(file_document_batch_split + func_document_batch_split)

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


def create_domain_knowledge():
    print("creating domian knowledge")
    # Load swebench train
    swebench_train = load_dataset("princeton-nlp/SWE-bench", split="train", cache_dir=config.RUNTIME_DIR)

    # Split into documents
    documents = []
    try:
        for instance in swebench_train.to_list()[:100]:
            instance["created_at"] = convert_iso_8601_to_timestamp(instance["created_at"])
            documents.append(Document(page_content=instance["problem_statement"], metadata=instance))
    except TypeError as te:
        print(te)
        print(instance)

    # split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=256, separators=[" ", ",", "\n"]
    )
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    db = Chroma.from_documents(docs, EMBEDDING_FUNCTION, collection_name="domain_knowledge_db")

    # create VectorStoreRetriever
    retriever = db.as_retriever()
    return retriever, db


# domain_knowledge_retriever, domain_knowledge_db = create_domain_knowledge()


@tool
def domain_knowledge(issue_report: str, issue_created_at: str):
    """Given a full issue report (without modification), search for a historically similar issue, and return its corresponding description and patch fix.

    Args:
        issue_report: The full issue report, to be used to find similar issues.
        issue_created_at: THe date the issue was created at.
    """

    domain_knowledge_retriever = None
    relevant_docs = domain_knowledge_retriever.get_relevant_documents(
        issue_report,
        k=1,
        filter={"created_at": {"$lt": convert_iso_8601_to_timestamp(issue_created_at)}},
    )

    if isinstance(relevant_docs, list):
        relevant_docs = relevant_docs[0]
    assert isinstance(relevant_docs, Document)

    # FOr now let's take a subset of the issue desc and patch since the result can be very long
    return Template(
        """"Similar issue report description:
```$issue_desc````

Similar issue report patch fix:
```$patch```"""
    ).substitute(
        issue_desc=relevant_docs.metadata["problem_statement"][:9999],
        patch=relevant_docs.metadata["patch"][:9999],
    )


@tool
def search_relevant_files(query: str, k=10, version=1):
    """Given a query search string (for example, the issue report description, filenames, etc), search for relevant code snippets of files in the project by calculating embedding similarity between the query and code snippets in a vector database.

    Args:
        query: A search string (for example, the issue report description, filenames, etc), to be used to find relevant files and functions.
    """
    rc = runtime_config.RuntimeConfig()
    print(version)
    print(rc.proj_path)
    if version == 2:
        project_knowledge_retriever, _ = create_project_knowledge_v2(rc.proj_path)
    else:
        project_knowledge_retriever, _ = create_project_knowledge(rc.proj_path)

    relevant_docs = project_knowledge_retriever.get_relevant_documents(query, k=k)
    # if isinstance(relevant_docs, list) and relevant_docs:
    #     relevant_docs = relevant_docs[0]
    # assert isinstance(relevant_docs, Document)

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
    print(full_result)
    print("-----RELEVANT DOCS-----")
    return_string = return_string.strip()

    explain_prompt = RELEVANT_FILE_EXPLANATION_SYSTEM_PROMPT.substitute(
        search_term=query, k=k, full_result=full_result
    ).strip()
    print("-----RELEVANT DOCS EXPLANATIONS-----")
    generate_explanation = llm.invoke([HumanMessage(explain_prompt)])
    print(generate_explanation)
    print("-----RELEVANT DOCS EXPLANATIONS-----")

    explanations = relevant_file_explanations_parser.invoke(
        generate_explanation.content
    )  # json.loads(generate_explanation.content)
    return explanations


def get_stage_messages(state: Annotated[dict, InjectedState]):
    previous_messages = message_processor_mk(state["messages"])  # Process messages via XY's approach
    previous_processed_messages_len = state.get("previous_processed_messages_len")
    last_agent = state.get("next_agent")  # We use next agent as last agent since it hasn't been updated yet
    stage_message_keys = list(previous_messages.keys())[
        previous_processed_messages_len:
    ]  # Subset the list of keys for this stage
    stage_messages = {}
    for k in stage_message_keys:
        stage_messages[k] = previous_messages[k]
    return stage_messages


def summarizer(stage_msgs_processed, last_agent=None):
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

    rc = runtime_config.RuntimeConfig()


    return summary


def filter_convo_history(next_agent: str, state: Annotated[dict, InjectedState], version=1):
    """Modify the chat history to include only the most relevant information for the next agent."""

    previous_messages = []
    for message in state["messages"]:
        previous_messages.append({"message": message.content, "id": message.id})

    prompt = FILTER_CONVO_HISTORY_SYSTEM_PROMPT.substitute(next_agent=next_agent, messages=previous_messages)
    if version == 2:
        try:
            next_agent_info = importlib.import_module(
                f"agent.prompt.{next_agent}"
            )  # Import the agent prompt file
            next_agent_possible_prompts = [
                k for k in list(next_agent_info.__dict__.keys()) if "SYSTEM_PROMPT" in k
            ]  # Get all vars with "SYSTEM_PROMPT" - should be ordered by their definition in Python 3.7+
            next_agent_prompt = next_agent_info.__dict__[
                next_agent_possible_prompts[-1]
            ]  # Get the last system prompt
            prompt = FILTER_CONVO_HISTORY_SYSTEM_PROMPT_V2.substitute(
                next_agent=next_agent,
                next_agent_prompt=next_agent_prompt,
                messages=previous_messages,
            )
        except AttributeError as ae:
            print("importlib Import failed: agent.prompt.{next_agent}\n{ae}")
        except IndexError as ie:
            print(f"No system prompts found via keys: {list(next_agent_info.__dict__.keys())}\n{ie}")
    response = llm.invoke([HumanMessage(prompt.strip(), name="relevant_convo_history")])

    filtered_message_ids = relevant_message_ids_parser.invoke(response.content.strip())
    print("## START filter_context_history")
    print(f"IDs before filter: {filtered_message_ids}")
    print([bool(hasattr(msg, "tool_calls") and msg.tool_calls) for msg in state["messages"]])
    print([str(type(msg)).split(".")[-1] for msg in state["messages"]])

    print("\nMsgs in the filtered msgs")
    print([state["messages"].index(msg) for msg in state["messages"] if msg.id in filtered_message_ids])
    print(
        [
            bool(hasattr(msg, "tool_calls") and msg.tool_calls)
            for msg in state["messages"]
            if msg.id in filtered_message_ids
        ]
    )
    print([str(type(msg)).split(".")[-1] for msg in state["messages"] if msg.id in filtered_message_ids])

    filtered_messages = []

    for i, message in enumerate(state["messages"]):
        if message.id in filtered_message_ids:
            if (
                i > 0
                and isinstance(message, ToolMessage)
                and isinstance(state["messages"][i - 1], AIMessage)
                and (
                    (filtered_messages and state["messages"][i - 1].id != filtered_messages[-1].id)
                    or not filtered_messages
                )
            ):
                filtered_messages.append(
                    state["messages"][i - 1]
                )  # Langgraph requires the AIMessage preceeding a ToolMessage to be present in the state - so we need to add it before adding the ToolMessage
            if (
                isinstance(message, AIMessage)
                and i + 1 < len(state["messages"])
                and isinstance(state["messages"][i + 1], ToolMessage)
                and state["messages"][i + 1].id not in filtered_message_ids
            ):  # As above, Langgraph requires the ToolMessage proceeding a AIMessage to be present in the state - so append it to filtered_message_ids so it will be added next
                filtered_message_ids.append(state["messages"][i + 1].id)

            filtered_messages.append(message)

    if (
        not filtered_messages or state["messages"] and state["messages"][0].id != filtered_messages[0].id
    ):  # If the initial prompt was not included in the filtered messages, be sure to include it
        filtered_messages = [state["messages"][0]] + filtered_messages
    if (
        not filtered_messages
        or state["messages"]
        and filtered_messages
        and state["messages"][-1].id != filtered_messages[-1].id
    ):  # If the final message from the previous agent was not included in the filtered messages, be sure to include it
        filtered_messages.append(state["messages"][-1])

    print(f"\nIDs after filter: {[x.id for x in filtered_messages]}")
    print([state["messages"].index(msg) for msg in filtered_messages])
    print([bool(hasattr(msg, "tool_calls") and msg.tool_calls) for msg in filtered_messages])
    print([str(type(msg)).split(".")[-1] for msg in filtered_messages])
    print(f"Final msg: {filtered_messages[-1]}")

    return filtered_messages


def update_relevant_convo_history(state: Annotated[dict, InjectedState]):
    relevant_convo_history = state.get("relevant_convo_history", "")
    if relevant_convo_history:
        relevant_convo_history += [state["messages"][-1]]  # Append the message from the supervisor
    return relevant_convo_history


def determine_relevant_convo_history_use(
    state: Annotated[dict, InjectedState],
    relevant_convo_history,
    summary,
    override=False,
    usage_type="summary",
):
    """Helper to decide whether to use the convo history or not. Also appends the message of the supervisor if it is to be used.

    Args:
        override (bool): If true, this function will simply use the state instead of relevant convo history.
    """

    def provide_summary(state):
        new_state = copy.deepcopy(state)
        summary_messages = (
            [state["messages"][0]] + [summary] + [state["messages"][-1]]
        )  # Initial prompt, summary, supervisor msg
        new_state["messages"] = summary_messages
        print("=" * 50)
        print("Providing summary")
        print(summary_messages)
        print("=" * 50)
        return new_state

    def provide_relevant_convo(state):
        new_state = copy.deepcopy(state)
        new_state["messages"] = relevant_convo_history
        print("=" * 50)
        print("Providing relevant convo history")
        print(relevant_convo_history)
        print("=" * 50)
        return new_state

    def provide_stage_convo(state):
        new_state = copy.deepcopy(state)
        previous_messages_len = state.get("previous_messages_len")

        new_state["messages"] = new_state["messages"][previous_messages_len:]
        print("=" * 50)
        print("Providing stage message history:")
        print(new_state["messages"])
        print("=" * 50)
        return new_state

    new_state = {}
    if (
        usage_type == "summary"
        and summary
        and state.get("last_agent", "") != state.get("next_agent", "")
        and not override
    ):
        new_state = provide_summary(state)
    elif (
        usage_type == "history"
        and relevant_convo_history
        and state.get("last_agent", "") != state.get("next_agent", "")
        and not override
    ):
        new_state = provide_relevant_convo
    elif (
        state.get("last_agent")
        and state.get("next_agent")
        and state.get("last_agent", "") == state.get("next_agent", "")
    ):  # If agent was recalled, try using the stage msg instead of all msgs
        new_state = provide_stage_convo(state)

    if new_state and "messages" in new_state:
        return new_state
    print("Providing default state")
    return state


@tool
def repo_class_and_func_knowledge(filepath: str, langs=None):
    """Retrieve a summary (the definition headers) for all classes and methods or functions for a given file and its corresponding root directory.

    Args:
        filepath: Full path of the file of interest.
    """
    rc = runtime_config.RuntimeConfig()
    if langs is None:
        langs = ["py", "java"]
    repo_knowledge = RepoKnowledge()
    complete_function_map, complete_file_map = {}, {}
    for lang in langs:
        lang_function_map, lang_file_map = repo_knowledge.create_map(rc.proj_path, lang)
        complete_function_map.update(lang_function_map)
        complete_file_map.update(lang_file_map)
    file_name = filepath.split("/")[-1]
    for k in list(complete_function_map.keys()):
        if file_name in k:
            return complete_function_map[k]
    return ""


# @tool
# def view_context_keys(state):
#     """Check the context previously retrieved by the context_manager. This include files and their content that correspond to the current issue."""
#     pass


context_tools = [
    # domain_knowledge,
    search_relevant_files,  # search_relevant_files,
    # summarizer,
    # repo_class_and_func_knowledge, # I don't think we need this tool as of now. If we do, I need to reverify its output (see obsidian note)
    view_directory,  # Note the version of this function defined in sepl_tools.py includes pagination, so it won't break the OS
    view_file_content,
    # search_files_by_name_or_content, # This tool froze my OS, it needs pagination / response limiting
    # run_shell_cmd,
]


def context_manager_tool():
    pass
