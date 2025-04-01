# %%
import getpass
import os
import uuid
from typing import Literal

import dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
from agent.tool_set.context_tools import search_relevant_files, summarizer
from agent.runtime_config import RuntimeConfig
from agent.llm import llm
from agent.prompt import (
    ISSUE_RESOLVE_PROBLEM_DECODER_SYSTEM_PROMPT,
    ISSUE_RESOLVE_PROBLEM_SOLVER_SYSTEM_PROMPT,
    ISSUE_RESOLVE_REVIEWER_SYSTEM_PROMPT,
    ISSUE_RESOLVE_SOLUTION_MAPPER_SYSTEM_PROMPT,
    ISSUE_RESOLVE_SUPERVISOR_SYSTEM_PROMPT,
    
)
from agent.tool_set.sepl_tools import (
    save_git_diff,
    view_file_content,
    view_directory
)
from agent.state import CustomState
from agent.tool_set.edit_tool import str_replace_editor
from agent.tool_set.context_tools import search_relevant_files, summarizer
from agent.tool_set.sepl_tools import (
    save_git_diff,
    view_file_content,
    view_directory
)
from agent.utils import message_processor_mk

rc = RuntimeConfig()

problem_decoder_tools = [view_directory,search_relevant_files, view_file_content]
solution_mapper_tools = [view_directory, search_relevant_files, view_file_content]
problem_solver_tools = [view_directory, search_relevant_files, str_replace_editor]
reviewer_tools = [view_directory, search_relevant_files, view_file_content]


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


dotenv.load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
)

# anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

members = ["problem_decoder", "solution_mapper", "problem_solver"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next_agent: Literal[*options]
    thought: str


def input_handler_node(state: CustomState) -> Command[Literal["supervisor"]]:
    """in issue solving, input handler will take input of 1.swe-bench id, 2.issue link and setup the env accordingly"""
    input = state["preset"]
    if "/issues/" in input:
        # the input are github link
        rc.load_from_github_issue_url(input)
    else:
        print("error, enter a valid issue link")
        return Command(goto=END)
    issue_description = rc.issue_desc
    return Command(
        update={
            "messages": [HumanMessage(content=issue_description)],
            "last_agent": "input_handler",
        },
        goto="supervisor",
    )


def supervisor_node(state: CustomState) -> Command:
    messages = [
        {"role": "system", "content": ISSUE_RESOLVE_SUPERVISOR_SYSTEM_PROMPT},
    ] + state["messages"]

    response = llm.with_structured_output(Router, strict=True).invoke(messages)

    next_agent = response["next_agent"]
    goto = next_agent

    goto = END if "FINISH" in goto else goto
    last_agent = state["last_agent"]

    if "human_in_the_loop" in state:
        if state["human_in_the_loop"] and next_agent != last_agent and last_agent != "input_handler":
            stage_messages = message_processor_mk(state["messages"])
            summary = summarizer(stage_messages, last_agent=last_agent)
            return Command(
                update={
                    "summary": summary,
                    "messages": [
                        AIMessage(
                            content="Supervisor:\nThought: "
                            + response["thought"]
                            + "\nNext: "
                            + response["next_agent"]
                            + ".",
                            name="supervisor",
                        )
                    ],
                    "last_agent": last_agent,
                    "next_agent": goto if goto != END else None,
                },
                goto="human_feedback",
            )
    return Command(
        update={
            "messages": [
                AIMessage(
                    content="Supervisor:\nThought: "
                    + response["thought"]
                    + "\nNext: "
                    + response["next_agent"]
                    + ".",
                    name="supervisor",
                )
            ],
            "last_agent": last_agent,
            "next_agent": goto if goto != END else None,
        },
        goto=goto,
    )


def human_feedback_node(state: CustomState) -> Command[Literal[*members]]:
    next_agent = state["next_agent"]
    last_agent = state["last_agent"]
    summary = state["summary"]
    show_to_human = f"{summary}\n Please provide feedback on the last agent: "
    human_feedback = interrupt(show_to_human)
    feedback = human_feedback["feedback"]
    rerun = bool(human_feedback["rerun"])  # rerun is 0 or 1, if 1, then rerun

    if rerun:
        # now we rerun the last agent
        print(f"Human decide to rerun the last agent: {last_agent}")
        return Command(
            update={
                "messages": [
                    AIMessage(content=summary, name="conversation_summary"),
                    HumanMessage(content=feedback, name="human_feedback"),
                ],
                "next_agent": None,
            },
            goto=last_agent,
        )

    print(f"Human decide to continue to the next agent: {next_agent}")
    return Command(
        update={
            "messages": [
                AIMessage(content=summary, name="conversation_summary"),
                HumanMessage(content=feedback, name="human_feedback"),
            ]
        },
        goto=next_agent,
    )


problem_decoder_agent = create_react_agent(
    llm,
    tools=problem_decoder_tools,
    state_modifier=ISSUE_RESOLVE_PROBLEM_DECODER_SYSTEM_PROMPT,
)


def problem_decoder_node(state: CustomState) -> Command[Literal["supervisor"]]:
    result = problem_decoder_agent.invoke(state)
    new_messages = result["messages"][len(state["messages"]) :]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "problem_decoder"

    return Command(
        update={"messages": new_messages, "last_agent": "problem_decoder"},
        goto="supervisor",
    )


solution_mapper_agent = create_react_agent(
    llm,
    tools=solution_mapper_tools,
    state_modifier=ISSUE_RESOLVE_SOLUTION_MAPPER_SYSTEM_PROMPT,
)


def solution_mapper_node(state: CustomState) -> Command[Literal["supervisor"]]:
    print("Solution mapper node is running ~")
    result = solution_mapper_agent.invoke(state)
    new_messages = result["messages"][len(state["messages"]) :]

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "solution_mapper"

    return Command(
        update={"messages": new_messages, "last_agent": "solution_mapper"},
        goto="supervisor",
    )


problem_solver_agent = create_react_agent(
    llm,
    tools=problem_solver_tools,
    state_modifier=ISSUE_RESOLVE_PROBLEM_SOLVER_SYSTEM_PROMPT,
)


def problem_solver_node(state: CustomState) -> Command[Literal["supervisor"]]:
    result = problem_solver_agent.invoke(state)
    new_messages = result["messages"][len(state["messages"]) :]

    # Add name to each AI message
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            msg.name = "problem_solver"


    latest_patch = "Below is the latest code changes:\n" + save_git_diff()
    latest_patch = latest_patch.rstrip()
    print(f"Latest patch: {latest_patch}")

    return Command(
        update={
            "messages": new_messages
            + [
                AIMessage(
                    content=latest_patch,
                    name="problem_solver",
                )
            ],
            "last_agent": "problem_solver",
        },
        goto="supervisor",
    )




memory = MemorySaver()
builder = StateGraph(CustomState)
builder.add_edge(START, "input_handler")
builder.add_node("input_handler", input_handler_node, destinations=({"supervisor":"supervisor-input_handler"}))
builder.add_node("human_feedback", human_feedback_node,destinations=("problem_decoder", "solution_mapper", "problem_solver"))
builder.add_node("problem_decoder", problem_decoder_node,destinations=({"supervisor":"supervisor-decoder"}))
builder.add_node("solution_mapper", solution_mapper_node,destinations=({"supervisor":"supervisor-mapper"}))
builder.add_node("problem_solver", problem_solver_node,destinations=({"supervisor":"supervisor-solver"}))
builder.add_node("supervisor", supervisor_node, destinations=("problem_decoder", "solution_mapper", "problem_solver", END))


issue_resolve_graph = builder.compile(checkpointer=memory)


# # %%
if __name__ == "__main__":
    # set os env of LANGSMITH_TRACING to true
    rc = RuntimeConfig()

    # when using input_handler_node, no need to initialized
    os.environ["LANGSMITH_TRACING"] = "true"
    thread = {
        "recursion_limit": 100,
        "run_id": uuid.uuid4(),
        "tags": ["interrupt"],
        "configurable": {"thread_id": "1"},
    }
    initial_input = {"messages":[HumanMessage(content="https://github.com/gitpython-developers/GitPython/issues/1413")], "preset": "https://github.com/gitpython-developers/GitPython/issues/1413", "human_in_the_loop": False}

    for chunk in issue_resolve_graph.stream(initial_input, config=thread, stream_mode="values"):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            chunk["messages"][-1].pretty_print()
