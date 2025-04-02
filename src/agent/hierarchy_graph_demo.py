# %%
import os
from typing import Literal
import uuid


import dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from agent.llm import llm
from agent.prompt import (
    ISSUE_RESOLVE_REVIEWER_SYSTEM_PROMPT,
    ISSUE_RESOLVE_MAM_SYSTEM_PROMPT,
)
from agent.runtime_config import RuntimeConfig
from agent.state import CustomState
from agent.supervisor_graph_demo import issue_resolve_graph
from agent.tool_set.context_tools import search_relevant_files
from agent.tool_set.sepl_tools import view_file_content, view_directory, run_shell_cmd

rc = RuntimeConfig()

reviewer_tools = [
    view_directory,
    search_relevant_files,
    view_file_content,
    run_shell_cmd,
]

dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)


class MamRouter(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    options = ["issue_resolver", "reviewer", "FINISH"]
    next_agent: Literal[*options]
    thought: str


def mam_node(
    state: CustomState,
) -> Command[Literal["issue_resolve_graph", "reviewer_node"]]:
    messages = [
        {"role": "system", "content": ISSUE_RESOLVE_MAM_SYSTEM_PROMPT},
    ] + state["messages"]

    response = llm.with_structured_output(MamRouter, strict=True).invoke(messages)

    next_agent = response["next_agent"]
    print(f"Next agent: {next_agent}")
    if "reviewer" in next_agent:
        goto = "reviewer_node"
    elif "issue_resolver" in next_agent:
        goto = "issue_resolve_graph"
    elif "FINISH" in next_agent:
        goto = END
    else:
        raise ValueError(f"Invalid next agent: {next_agent}")

    return Command(
        update={
            "messages": [
                AIMessage(
                    content="MAM:\nThought: "
                    + response["thought"]
                    + "\nNext: "
                    + response["next_agent"]
                    + ".",
                    name="MAM",
                )
            ],
            "next_agent": goto if goto != END else None,
        },
        goto=goto,
    )


reviewer_agent = create_react_agent(
    llm,
    tools=reviewer_tools,
    state_modifier=ISSUE_RESOLVE_REVIEWER_SYSTEM_PROMPT,
)


def reviewer_node(state: CustomState) -> Command[Literal["mam_node"]]:
    result = reviewer_agent.invoke(state)
    new_messages = result["messages"][len(state["messages"]) :]
    last_message = new_messages[-1]
    # Add name to each AI message
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "reviewer_node"

    return Command(
        update={"messages": last_message},
        goto="mam_node",
    )


builder = StateGraph(CustomState)
builder.add_node(
    "issue_resolve_graph",
    issue_resolve_graph,
    destinations=({"mam_node": "mam_node-issue_resolve_graph"}),
)
builder.add_node(
    "mam_node",
    mam_node,
    destinations=(
        {
            "reviewer_node": "mam_node-reviewer_node",
            "issue_resolve_graph": "mam_node-issue_resolve_graph",
        }
    ),
)
builder.add_node(
    "reviewer_node",
    reviewer_node,
    destinations=({"mam_node": "mam_node-reviewer_node"}),
)
builder.add_edge(START, "mam_node")
builder.add_edge("issue_resolve_graph", "mam_node")


hierarchy_graph = builder.compile()


# # %%
if __name__ == "__main__":
    rc = RuntimeConfig()

    os.environ["LANGSMITH_TRACING"] = "true"
    thread = {
        "recursion_limit": 100,
        "run_id": uuid.uuid4(),
        "tags": ["interrupt"],
        "configurable": {"thread_id": "1"},
    }
    initial_input = {
        "messages": [
            HumanMessage(
                content="https://github.com/gitpython-developers/GitPython/issues/1413"
            )
        ],
        "preset": "https://github.com/gitpython-developers/GitPython/issues/1413",
        "human_in_the_loop": False,
    }

    for chunk in hierarchy_graph.stream(
        initial_input, config=thread, stream_mode="values"
    ):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            chunk["messages"][-1].pretty_print()
