import argparse
import asyncio
import datetime
from functools import lru_cache
import logging
import os
from typing import Any, Dict, Literal, Optional, cast
import uuid

from datasets import load_dataset
from langchain.schema.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command, RetryPolicy

from src.agent import runtime_config
from src.agent.constant import RUNTIME_DIR
from src.agent.logging_config import configure_logging, get_logger
from src.agent.prompt.problem_decoder import (
    SWE_PROBLEM_DECODER_AGGREGATOR_PROMPT_V11,
    SWE_PROBLEM_DECODER_SYSTEM_PROMPT_V11,
    SWE_PROBLEM_DECODER_USER_PROMPT_V11,
)
from src.agent.prompt.problem_solver import (
    SWE_PROBLEM_SOLVER_SYSTEM_PROMPT_V11,
    SWE_PROBLEM_SOLVER_USER_PROMPT_V11,
)
from src.agent.prompt.solution_mapper import (
    SWE_SOLUTION_MAPPER_SYSTEM_PROMPT_V11,
    SWE_SOLUTION_MAPPER_USER_PROMPT_V11,
)
from src.agent.state import State, messages_reducer
from src.agent.tool_set.deepwiki_tool import ask_repository_agent
from src.agent.tool_set.dev_knowledge import (
    get_agent_dev_knowledge,
    get_dev_knowledge_design_version_4,
)
from src.agent.tool_set.edit_tool import str_replace_based_edit_tool, str_replace_editor
from src.agent.tool_set.sepl_tools import (
    run_shell_cmd,
    search_files_by_keywords,
    think,
    view_directory,
    view_file_content,
)
from src.agent.tool_set.utils import get_runtime_config
from src.agent.utils import (
    compress_agent_thinking_observation_action,
    replay_agent_action,
)
from langchain_core.load import dumpd, dumps, load, loads
import json
from src.utils.format_utils import format_section_header

str_output_parser = StrOutputParser()

ANTHROPIC_MODEL_VERSION = "4"

logger = get_logger(__name__)
configure_logging(
    level=logging.INFO, log_file="knowledge_tts_workflow_for_swebench_v1.log"
)


@lru_cache(maxsize=8)
def get_llm(
    temperature: float = 0.0, thinking_budget: int = 1024, max_tokens: int = 3072
):
    """Get an LLM with the specified temperature."""
    return ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=1,
        max_tokens_to_sample=max_tokens + thinking_budget,
        thinking={"type": "enabled", "budget_tokens": thinking_budget},
        betas=["interleaved-thinking-2025-05-14"],
        timeout=360,
        stop=None,
        # streaming=True,
        max_retries=3,
    )


def input_handler_node(
    state: State, config: RunnableConfig
) -> Command[Literal["coordinator"]]:
    """
    Handle input and setup the environment.
    """
    logger.info("Starting input handler node")
    input_value = state.get("preset")
    if not input_value:
        raise ValueError("No input value provided")
    rc = runtime_config.RuntimeConfig()
    # Load configuration based on input type
    if "/issues/" in input_value:
        logger.info(f"Loading from GitHub issue URL: {input_value}")
        rc.load_from_github_issue_url(input_value)
        runtime_info = rc.dump_config()
        logger.info(
            f"Initialized runtime config with type={rc.runtime_type}, project={rc.proj_name}"
        )
        rc.pretty_print_runtime()
    else:
        logger.info(f"Loading swe-bench instance: {input_value}")
        swe_instances = load_dataset(
            "princeton-nlp/SWE-bench_Verified", split="test", cache_dir=RUNTIME_DIR
        )
        found = False
        for entry in swe_instances:
            entry_dict = dict(entry)
            if entry_dict["instance_id"] == input_value:
                found = True
                issue_desc = entry_dict["problem_statement"]
                break
        if not found:
            raise ValueError(f"Invalid SWE instance id: {input_value}")
        runtime_info = None

    # Format issue description
    issue_desc = f"<issue_description>\n{issue_desc}\n</issue_description>\n"
    logger.info(format_section_header("Issue Description", issue_desc))

    return Command(
        update={
            "issue_description": issue_desc,
            "runtime_info": runtime_info,
            "last_node": "input_handler",
        },
        goto="coordinator",
    )


def knowledge_extractor_node(
    state: State, config: RunnableConfig
) -> Command[Literal["coordinator"]]:
    """
    Knowledge extractor will extract the knowledge from the history similar issues.
    """
    KNOWLEDGE_COUNT = 3

    dev_knowledges = get_dev_knowledge_design_version_4.invoke(
        {"instance_id": state.get("preset"), "top_n": KNOWLEDGE_COUNT}
    )

    return Command(
        goto="coordinator",
        update={"last_node": "knowledge_extractor", "dev_knowledges": dev_knowledges},
    )


def coordinator_node(state: State, config: RunnableConfig) -> Command[
    Literal[
        "knowledge_extractor",
        "problem_decoder",
        "solution_mapper",
        "problem_solver",
        "post_processing",
    ]
]:
    last_node = state.get("last_node")

    if last_node == "input_handler":
        return Command(goto="knowledge_extractor")

    if last_node == "knowledge_extractor":
        return Command(goto="problem_decoder")

    if state.get("problem_decoder_final_output") and last_node == "problem_decoder":
        return Command(goto="solution_mapper")

    if state.get("solution_mapper_final_output") and last_node == "solution_mapper":
        return Command(goto="problem_solver")

    if state.get("problem_solver_final_output") and last_node == "problem_solver":
        return Command(goto="post_processing")

    return Command(goto="post_processing")


def problem_decoder_node(
    state: State, config: RunnableConfig
) -> Command[
    Literal["problem_decoder_sampler", "problem_decoder_aggregator", "coordinator"]
]:
    """
    Problem decoder will decode the problem into a list of problems.
    Act as a router to problem_decoder_sampler
    """
    logger.info("arrive at problem_decoder")
    # count how many dev_knowledges are there
    dev_knowledges = state.get("dev_knowledges", [])
    dev_knowledge_count = len(dev_knowledges)
    # Get target iterations from state
    target_iterations = state.get("decoder_iterations", 3)
    logger.info(
        f"dev_knowledge_count: {dev_knowledge_count}, target_iterations: {target_iterations}"
    )

    problem_decoder_outputs = state.get("problem_decoder_outputs", [])
    if problem_decoder_outputs is None:
        decoder_output_count = 0
    else:
        decoder_output_count = len(problem_decoder_outputs)
    logger.info(f"decoder_output_count: {decoder_output_count}")

    if decoder_output_count < target_iterations:
        # Get current knowledge if available, otherwise use None
        if decoder_output_count < dev_knowledge_count:
            current_knowledge = dev_knowledges[decoder_output_count]
        else:
            current_knowledge = None
            logger.info(
                f"No dev_knowledge available for index {decoder_output_count}, using None"
            )

        return Command(
            goto="problem_decoder_sampler",
            update={
                "current_knowledge": current_knowledge,
                "current_sample_index": decoder_output_count,
            },
        )
    elif (
        decoder_output_count == target_iterations
        and state.get("problem_decoder_final_output") is None
    ):

        return Command(goto="problem_decoder_aggregator")
    else:
        return Command(goto="coordinator", update={"last_node": "problem_decoder"})


def problem_decoder_sampler_node(
    state: State, config: RunnableConfig
) -> Command[Literal["problem_decoder"]]:
    """
    Problem decoder sampler will sample the problem from the list of problems based on the knowledge.
    """
    current_knowledge = state.get("current_knowledge")
    current_sample_index = state.get("current_sample_index")
    if current_knowledge:
        decoder_knowledge = get_agent_dev_knowledge(
            "decoder", current_knowledge.dev_knowledge
        )
    else:
        decoder_knowledge = ""

    problem_decoder_user_prompt = SWE_PROBLEM_DECODER_USER_PROMPT_V11.format(
        ISSUE_DESCRIPTION=state.get("issue_description"),
        DEV_KNOWLEDGE=decoder_knowledge,
        GUIDING_QUESTIONS="",
        ANSWER_OF_GUIDING_QUESTIONS="",
    )

    problem_decoder_state = {**state, "messages": []}
    human_msg_content = [
        {
            "text": problem_decoder_user_prompt,
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    problem_decoder_state["messages"] = messages_reducer(
        problem_decoder_state["messages"],
        [HumanMessage(content=cast(Any, human_msg_content))],
    )

    # Setup cache paths
    cache_path = None
    if state["cache_dir"] is not None:
        cache_path = os.path.join(
            state["cache_dir"],
            f"problem_decoder_{current_sample_index}.json",
        )

    result = None
    cache_loaded = False

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as fp:
                result = loads(json.load(fp), ignore_unserializable_fields=True)
            logger.info(f"Loading problem_decoder_{current_sample_index} output from cache: {cache_path}")
            if result is not None:
                cache_loaded = True
                logger.info(f"problem_decoder_{current_sample_index} loaded from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            result = None

    if not cache_loaded:

        tools = [
            view_directory,
            search_files_by_keywords,
            view_file_content,
            ask_repository_agent,
            think,
        ]
        tool_node = ToolNode(tools, handle_tool_errors=False)

        agent = create_react_agent(
            get_llm().bind_tools(tools),
            tools=tool_node,
            prompt=SWE_PROBLEM_DECODER_SYSTEM_PROMPT_V11,
            version="v2",
            state_schema=State,
        )

        # reset and init runtime
        runtime_obj = get_runtime_config(config)
        runtime_obj.reset_instance()
        preset = state.get("preset")
        if not preset:
            raise ValueError("No preset instance_id provided")
        runtime_obj.load_from_swe_rex_docker_instance(preset)
        runtime_obj.pretty_print_runtime()
        config.setdefault("configurable", {})["runtime_object"] = runtime_obj
        logger.info(
            f"Runtime initialized for problem decoder #{current_sample_index} with SWEREX deployment"
        )

        config.setdefault("configurable", {})[
            "agent_name"
        ] = f"problem_decoder_{current_sample_index}"
        logger.info(
            f"problem_decoder_sampler #{current_sample_index} starting execution with runtime_type={runtime_obj.runtime_type}"
        )

        config.setdefault("recursion_limit", 200)
        result = agent.invoke(problem_decoder_state, config=config)

        # stop the swe_rex_deployment
        if runtime_obj.runtime_type == runtime_config.RuntimeType.SWEREX:
            deployment = runtime_obj.swe_rex_deployment
            logger.info(
                f"Problem decoder {current_sample_index}: Stopping SWEREX deployment"
            )
            asyncio.run(deployment.stop())

        logger.info(
            f"problem_decoder_sampler #{current_sample_index} completed execution successfully"
        )

        new_messages = result["messages"][len(problem_decoder_state["messages"]) :]
        if len(new_messages) == 0 or (
            new_messages[-1].content is None or new_messages[-1].content == []
        ):
            error_msg = (
                f"No problem_decoder_sampler #{current_sample_index} output generated"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        if cache_path:
            try:
                result = {"messages": result["messages"]}
                string_representation = dumps(result, pretty=True)
                with open(cache_path, "w") as fp:
                    json.dump(string_representation, fp)
                logger.info(f"Saving problem_decoder_sampler #{current_sample_index} output to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_path}: {e}")

    if result is None:
        raise Exception("No result generated in problem_decoder_sampler")
    new_messages = result["messages"][len(problem_decoder_state["messages"]) :]
    
    
    # Name the agent messages
    for msg in new_messages:
        msg.name = f"problem_decoder {current_sample_index}"

    # Compress the messages
    compressed_message = compress_agent_thinking_observation_action(
        new_messages.copy(), f"problem_decoder {current_sample_index}"
    )
    if compressed_message is None:
        raise Exception("No compressed message generated in problem_decoder_sampler")
    logger.info(
        format_section_header(
            "problem_decoder",
            compressed_message,
            current_sample_index,
        )
    )
    problem_decoder_outputs = state.get("problem_decoder_outputs", [])
    return Command(
        goto="problem_decoder",
        update={
            "messages": new_messages,
            "problem_decoder_outputs": problem_decoder_outputs + [compressed_message],
        },
    )


def problem_decoder_aggregator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["problem_decoder"]]:
    """
    Problem decoder aggregator will aggregate the problem from the list of problems based on the knowledge.
    """
    problem_decoder_outputs = state.get("problem_decoder_outputs", [])

    if len(problem_decoder_outputs) == 1:
        logger.info(
            "problem_decoder_aggregator: only one output, skipping aggregation, using single output as final output"
        )
        problem_decoder_final_output = problem_decoder_outputs[0]
        logger.info(
            format_section_header(
                "problem_decoder_aggregator",
                problem_decoder_final_output,
            )
        )

        return Command(
            goto="problem_decoder",
            update={"problem_decoder_final_output": problem_decoder_final_output},
        )

    samples_content = []
    for i, problem_decoder_output in enumerate(problem_decoder_outputs):
        samples_content.append(
            f"<problem_decoder_trajectory_and_output_sample_#{i+1}>\n"
            f"{problem_decoder_output}\n"
            f"</problem_decoder_trajectory_and_output_sample_#{i+1}>"
        )
    samples_content = "\n\n".join(samples_content)

    problem_decoder_aggregator_prompt = (
        SWE_PROBLEM_DECODER_AGGREGATOR_PROMPT_V11.format(
            ISSUE_DESCRIPTION=state.get("issue_description"),
            SAMPLES_CONTENT=samples_content,
        )
    )

    # Setup cache paths
    cache_path = None
    if state["cache_dir"] is not None:
        cache_path = os.path.join(
            state["cache_dir"],
            "problem_decoder_aggregator.json",
        )

    critic_result = None
    cache_loaded = False

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as fp:
                critic_result = loads(json.load(fp), ignore_unserializable_fields=True)
            logger.info(f"Loading problem_decoder_aggregator output from cache: {cache_path}")
            if critic_result is not None:
                cache_loaded = True
                logger.info("problem_decoder_aggregator loaded from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            critic_result = None

    if not cache_loaded:
        critic_llm = get_llm(thinking_budget=4096, max_tokens=8192)
        critic_result = critic_llm.invoke(
            [
                {
                    "role": "user",
                    "content": (problem_decoder_aggregator_prompt),
                }
            ]
        )

        if cache_path:
            try:
                string_representation = dumps(critic_result, pretty=True)
                with open(cache_path, "w") as fp:
                    json.dump(string_representation, fp)
                logger.info(f"Saving problem_decoder_aggregator output to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_path}: {e}")

    if critic_result is None:
        raise Exception("No result generated in problem_decoder_aggregator")

    # Use the critic result content as the final output
    response = critic_result.content
    if isinstance(response, str):
        problem_decoder_final_output = response
    elif isinstance(response, list):
        problem_decoder_final_output = cast(Dict[str, Any], response[-1])["text"]
    else:
        raise Exception("critic_result should be a string or list")

    logger.info(
        format_section_header(
            "problem_decoder_aggregator",
            problem_decoder_final_output,
        )
    )

    return Command(
        goto="problem_decoder",
        update={"problem_decoder_final_output": problem_decoder_final_output},
    )


def solution_mapper_node(
    state: State, config: RunnableConfig
) -> Command[
    Literal["solution_mapper_sampler", "solution_mapper_aggregator", "coordinator"]
]:
    """
    Solution mapper will map the problem to potential solutions.
    Act as a router to solution_mapper_sampler
    """
    logger.info("arrive at solution_mapper")
    # count how many dev_knowledges are there
    dev_knowledges = state.get("dev_knowledges", [])
    dev_knowledge_count = len(dev_knowledges)

    target_iterations = state.get("mapper_iterations", 1)
    logger.info(
        f"dev_knowledge_count: {dev_knowledge_count}, target_iterations: {target_iterations}"
    )

    solution_mapper_outputs = state.get("solution_mapper_outputs", [])
    if solution_mapper_outputs is None:
        mapper_output_count = 0
    else:
        mapper_output_count = len(solution_mapper_outputs)
    logger.info(f"mapper_output_count: {mapper_output_count}")

    if mapper_output_count < target_iterations:
        # Get current knowledge if available, otherwise use None
        if mapper_output_count < dev_knowledge_count:
            current_knowledge = dev_knowledges[mapper_output_count]
        else:
            current_knowledge = None
            logger.info(
                f"No dev_knowledge available for index {mapper_output_count}, using None"
            )

        return Command(
            goto="solution_mapper_sampler",
            update={
                "current_knowledge": current_knowledge,
                "current_sample_index": mapper_output_count,
            },
        )
    elif (
        mapper_output_count == target_iterations
        and state.get("solution_mapper_final_output") is None
    ):
        return Command(goto="solution_mapper_aggregator")
    else:
        return Command(goto="coordinator", update={"last_node": "solution_mapper"})


def solution_mapper_sampler_node(
    state: State, config: RunnableConfig
) -> Command[Literal["solution_mapper"]]:
    """
    Solution mapper sampler will sample the solution mapping from the list of solutions based on the knowledge.
    """
    current_knowledge = state.get("current_knowledge")
    current_sample_index = state.get("current_sample_index")
    problem_decoder_output = state.get("problem_decoder_final_output")

    if current_knowledge:
        decoder_knowledge = get_agent_dev_knowledge(
            "mapper", current_knowledge.dev_knowledge
        )
    else:
        decoder_knowledge = ""

    solution_mapper_user_prompt = SWE_SOLUTION_MAPPER_USER_PROMPT_V11.format(
        ISSUE_DESCRIPTION=state.get("issue_description"),
        PROBLEM_DECODER_OUTPUT=problem_decoder_output,
        DEV_KNOWLEDGE=decoder_knowledge,
        GUIDING_QUESTIONS="",
        ANSWER_OF_GUIDING_QUESTIONS="",
    )

    solution_mapper_state = {**state, "messages": []}
    human_msg_content = [
        {
            "text": solution_mapper_user_prompt,
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    solution_mapper_state["messages"] = messages_reducer(
        solution_mapper_state["messages"],
        [HumanMessage(content=cast(Any, human_msg_content))],
    )

    # Setup cache paths
    cache_path = None
    if state["cache_dir"] is not None:
        cache_path = os.path.join(
            state["cache_dir"],
            f"solution_mapper_{current_sample_index}.json",
        )

    result = None
    cache_loaded = False

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as fp:
                result = loads(json.load(fp), ignore_unserializable_fields=True)
            logger.info(f"Loading solution_mapper_{current_sample_index} output from cache: {cache_path}")
            if result is not None:
                cache_loaded = True
                logger.info(f"solution_mapper_{current_sample_index} loaded from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            result = None

    if not cache_loaded:
        if ANTHROPIC_MODEL_VERSION == "4":
            anthropic_str_tool = {
                "type": "text_editor_20250429",
                "name": "str_replace_based_edit_tool",
            }
            tools = [
                search_files_by_keywords,
                str_replace_based_edit_tool,
                ask_repository_agent,
                run_shell_cmd,
                think,
            ]
        else:
            anthropic_str_tool = {
                "type": "text_editor_20250124",
                "name": "str_replace_editor",
            }
            tools = [
                search_files_by_keywords,
                str_replace_editor,
                ask_repository_agent,
                run_shell_cmd,
                think,
            ]
        tool_node = ToolNode(tools, handle_tool_errors=False)

        agent = create_react_agent(
            get_llm().bind_tools(
                [
                    search_files_by_keywords,
                    anthropic_str_tool,
                    ask_repository_agent,
                    run_shell_cmd,
                    think,
                ],
            ),
            tools=tool_node,
            prompt=SWE_SOLUTION_MAPPER_SYSTEM_PROMPT_V11,
            version="v2",
            state_schema=State,
        )

        # reset and
        runtime_obj = get_runtime_config(config)
        runtime_obj.reset_instance()
        preset = state.get("preset")
        if not preset:
            raise ValueError("No preset instance_id provided")
        runtime_obj.load_from_swe_rex_docker_instance(preset)
        runtime_obj.pretty_print_runtime()
        config.setdefault("configurable", {})["runtime_object"] = runtime_obj
        logger.info(
            f"Runtime initialized for solution mapper #{current_sample_index} with SWEREX deployment"
        )

        config.setdefault("configurable", {})[
            "agent_name"
        ] = f"solution_mapper_{current_sample_index}"
        logger.info(
            f"solution_mapper_sampler #{current_sample_index} starting execution with runtime_type={runtime_obj.runtime_type}"
        )

        config.setdefault("recursion_limit", 200)

        result = agent.invoke(solution_mapper_state, config=config)

        # stop the swe_rex_deployment
        if runtime_obj.runtime_type == runtime_config.RuntimeType.SWEREX:
            deployment = runtime_obj.swe_rex_deployment
            logger.info(
                f"Solution mapper {current_sample_index}: Stopping SWEREX deployment"
            )
            asyncio.run(deployment.stop())

        logger.info(
            f"solution_mapper_sampler #{current_sample_index} completed execution successfully"
        )
        new_messages = result["messages"][len(solution_mapper_state["messages"]) :]

        if len(new_messages) == 0 or (
            new_messages[-1].content is None or new_messages[-1].content == []
        ):
            error_msg = (
                f"No solution_mapper_sampler #{current_sample_index} output generated"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        if cache_path:
            try:
                result = {"messages": result["messages"]}
                string_representation = dumps(result, pretty=True)
                with open(cache_path, "w") as fp:
                    json.dump(string_representation, fp)
                logger.info(f"Saving solution_mapper_sampler #{current_sample_index} output to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_path}: {e}")

    if result is None:
        raise Exception("No result generated in solution_mapper_sampler")
    new_messages = result["messages"][len(solution_mapper_state["messages"]) :]

    # Name the agent messages
    for msg in new_messages:
        msg.name = f"solution_mapper {current_sample_index}"

    compressed_message = compress_agent_thinking_observation_action(
        new_messages.copy(), f"solution_mapper {current_sample_index}"
    )
    if compressed_message is None:
        raise Exception("No compressed message generated in solution_mapper_sampler")
    logger.info(
        format_section_header(
            "solution_mapper",
            compressed_message,
            current_sample_index,
        )
    )

    solution_mapper_outputs = state.get("solution_mapper_outputs", [])
    solution_mapper_messages = state.get("solution_mapper_messages", [])
    return Command(
        goto="solution_mapper",
        update={
            "messages": new_messages,
            "solution_mapper_messages": solution_mapper_messages + [new_messages],
            "solution_mapper_outputs": solution_mapper_outputs + [compressed_message],
        },
    )


def solution_mapper_aggregator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["solution_mapper"]]:
    """
    Solution mapper aggregator will aggregate the solution mapping from the list of solutions based on the knowledge.
    """
    solution_mapper_final_output = "dummy_solution_mapper_output"
    return Command(
        goto="solution_mapper",
        update={"solution_mapper_final_output": solution_mapper_final_output},
    )


def problem_solver_node(
    state: State, config: RunnableConfig
) -> Command[
    Literal["problem_solver_sampler", "problem_solver_aggregator", "coordinator"]
]:
    """
    Problem solver will implement the solution.
    Act as a router to problem_solver_sampler
    """
    logger.info("arrive at problem_solver")
    # count how many dev_knowledges are there
    dev_knowledges = state.get("dev_knowledges", [])
    dev_knowledge_count = len(dev_knowledges)

    target_iterations = state.get("solver_iterations", 1)
    logger.info(
        f"dev_knowledge_count: {dev_knowledge_count}, target_iterations: {target_iterations}"
    )

    problem_solver_outputs = state.get("problem_solver_outputs", [])
    if problem_solver_outputs is None:
        solver_output_count = 0
    else:
        solver_output_count = len(problem_solver_outputs)
    logger.info(f"solver_output_count: {solver_output_count}")

    if solver_output_count < target_iterations:
        # Get current knowledge if available, otherwise use None
        if solver_output_count < dev_knowledge_count:
            current_knowledge = dev_knowledges[solver_output_count]
        else:
            current_knowledge = None
            logger.info(
                f"No dev_knowledge available for index {solver_output_count}, using None"
            )
        return Command(
            goto="problem_solver_sampler",
            update={
                "current_knowledge": current_knowledge,
                "current_sample_index": solver_output_count,
            },
        )
    elif (
        solver_output_count == target_iterations
        and state.get("problem_solver_final_output") is None
    ):
        return Command(goto="problem_solver_aggregator")
    else:
        return Command(goto="coordinator", update={"last_node": "problem_solver"})


def problem_solver_sampler_node(
    state: State, config: RunnableConfig
) -> Command[Literal["problem_solver"]]:
    """
    Problem solver sampler will sample the solution implementation from the list of solutions based on the knowledge.
    """
    current_knowledge = state.get("current_knowledge")
    current_sample_index = state.get("current_sample_index")
    if current_sample_index is None:
        raise Exception("current_sample_index is None")
    else:
        current_sample_index = int(current_sample_index)
    problem_decoder_output = state.get("problem_decoder_final_output")
    solution_mapper_outputs = state.get("solution_mapper_outputs", [])
    solution_mapper_messages = state.get("solution_mapper_messages", [])
    if not solution_mapper_outputs or not isinstance(solution_mapper_outputs, list):
        raise Exception("No solution_mapper_outputs found")
    if len(solution_mapper_outputs) <= current_sample_index:
        raise Exception(
            f"current_sample_index {current_sample_index} is out of range for solution_mapper_outputs"
        )
    current_solution_mapper_output = solution_mapper_outputs[current_sample_index]
    current_solution_mapper_message = (
        solution_mapper_messages[current_sample_index]
        if current_sample_index < len(solution_mapper_messages)
        else []
    )

    if current_knowledge:
        decoder_knowledge = get_agent_dev_knowledge(
            "solver", current_knowledge.dev_knowledge
        )
    else:
        decoder_knowledge = ""

    problem_solver_user_prompt = SWE_PROBLEM_SOLVER_USER_PROMPT_V11.format(
        ISSUE_DESCRIPTION=state.get("issue_description"),
        PROBLEM_DECODER_OUTPUT=problem_decoder_output,
        CODE_CHANGE_PLAN=current_solution_mapper_output,
        DEV_KNOWLEDGE=decoder_knowledge,
    )

    problem_solver_state = {**state, "messages": []}
    human_msg_content = [
        {
            "text": problem_solver_user_prompt,
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    problem_solver_state["messages"] = messages_reducer(
        problem_solver_state["messages"],
        [HumanMessage(content=cast(Any, human_msg_content))],
    )

    # Setup cache paths
    cache_path = None
    patch_path = None
    if state["cache_dir"] is not None:
        cache_path = os.path.join(
            state["cache_dir"],
            f"problem_solver_{current_sample_index}.json",
        )
        patch_path = os.path.join(
            state["cache_dir"],
            f"problem_solver_{current_sample_index}.patch",
        )

    result = None
    generated_patch = None
    cache_loaded = False

    if cache_path and os.path.exists(cache_path):
        if patch_path and os.path.exists(patch_path):
            with open(patch_path, "r") as f:
                generated_patch = f.read()
                logger.info(
                    f"Loading problem_solver_{current_sample_index} result patch from cache: {patch_path}"
                )
        try:
            with open(cache_path, "r") as fp:
                result = loads(json.load(fp), ignore_unserializable_fields=True)
            logger.info(f"Loading problem_solver_{current_sample_index} output from cache: {cache_path}")
            if result is not None:
                cache_loaded = True
                logger.info(f"problem_solver_{current_sample_index} loaded from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            result = None

    if not cache_loaded or not generated_patch:
        if ANTHROPIC_MODEL_VERSION == "4":
            anthropic_str_tool = {
                "type": "text_editor_20250429",
                "name": "str_replace_based_edit_tool",
            }
            tools = [
                search_files_by_keywords,
                str_replace_based_edit_tool,
                ask_repository_agent,
                run_shell_cmd,
                think,
            ]
        else:
            anthropic_str_tool = {
                "type": "text_editor_20250124",
                "name": "str_replace_editor",
            }
            tools = [
                search_files_by_keywords,
                str_replace_editor,
                ask_repository_agent,
                run_shell_cmd,
                think,
            ]
        tool_node = ToolNode(tools, handle_tool_errors=False)

        agent = create_react_agent(
            get_llm().bind_tools(
                [
                    search_files_by_keywords,
                    anthropic_str_tool,
                    ask_repository_agent,
                    run_shell_cmd,
                    think,
                ],
            ),
            tools=tool_node,
            prompt=SWE_PROBLEM_SOLVER_SYSTEM_PROMPT_V11,
            version="v2",
            state_schema=State,
        )

        runtime_obj = get_runtime_config(config)
        runtime_obj.reset_instance()
        runtime_obj.load_from_swe_rex_docker_instance(state.get("preset"))
        runtime_obj.pretty_print_runtime()
        config.setdefault("configurable", {})["runtime_object"] = runtime_obj
        logger.info(
            f"Runtime initialized for problem solver {current_sample_index} with SWEREX deployment"
        )

        config.setdefault("configurable", {})[
            "agent_name"
        ] = f"problem_solver_{current_sample_index}"
        logger.info(
            f"problem_solver_sampler #{current_sample_index} starting execution with runtime_type={runtime_obj.runtime_type}"
        )

        if current_solution_mapper_message:
            if not isinstance(current_solution_mapper_message, list):
                raise Exception("current_solution_mapper_message is not a list")
            logger.info(
                f"problem_solver_sampler #{current_sample_index} replaying {len(current_solution_mapper_message)} messages"
            )
            problem_solver_state["messages"] = messages_reducer(
                problem_solver_state["messages"],
                current_solution_mapper_message,
            )
            replay_result = replay_agent_action(current_solution_mapper_message, config)
            if replay_result:
                logger.info(
                    f"problem_solver_sampler #{current_sample_index} successfully replayed {len(replay_result)} tool calls"
                )
            else:
                logger.warning(
                    f"problem_solver_sampler #{current_sample_index} no tool calls to replay"
                )

        logger.info(
            f"problem_solver_sampler #{current_sample_index} starting execution"
        )
        config.setdefault("recursion_limit", 200)
        result = agent.invoke(problem_solver_state, config=config)

        logger.info(
            f"problem_solver_sampler #{current_sample_index} completed execution successfully"
        )

        new_messages = result["messages"][len(problem_solver_state["messages"]) :]
        if len(new_messages) == 0 or (
            new_messages[-1].content is None or new_messages[-1].content == []
        ):
            error_msg = (
                f"No problem_solver_sampler #{current_sample_index} output generated"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        generated_patch = None
        # stop the swe_rex_deployment
        if runtime_obj.runtime_type == runtime_config.RuntimeType.SWEREX:
            generated_patch = run_shell_cmd.invoke(
                {"command": "git -c core.fileMode=false diff --exit-code --no-color"},
                config=config,
            )
            generated_patch = generated_patch + "\n"

            patch_size = len(generated_patch) if generated_patch else 0
            logger.info(
                f"Problem solver {current_sample_index} generated patch of size {patch_size} bytes"
            )
            deployment = runtime_obj.swe_rex_deployment
            logger.info(
                f"Problem solver {current_sample_index}: Stopping SWEREX deployment"
            )
            asyncio.run(deployment.stop())

        if generated_patch and patch_path:
            if generated_patch != "":
                os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                with open(patch_path, "w", encoding="utf-8") as f:
                    f.write(generated_patch)
                    logger.debug(
                        f"Saving problem_solver_{current_sample_index} result patch to cache: {patch_path}"
                    )
            else:
                logger.warning(
                    f"Problem solver {current_sample_index} generated EMPTY patch!"
                )

        if cache_path:
            try:
                result = {"messages": result["messages"]}
                string_representation = dumps(result, pretty=True)
                with open(cache_path, "w") as fp:
                    json.dump(string_representation, fp)
                logger.info(f"Saving problem_solver_sampler #{current_sample_index} output to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_path}: {e}")

    if result is None:
        raise Exception("No result generated in problem_solver_sampler")
    new_messages = result["messages"][len(problem_solver_state["messages"]) :]

    # Name the agent messages
    for msg in new_messages:
        msg.name = f"problem_solver {current_sample_index}"

    compressed_message = compress_agent_thinking_observation_action(
        new_messages.copy(), f"problem_solver {current_sample_index}"
    )
    if compressed_message is None:
        raise Exception("No compressed message generated in problem_solver_sampler")

    if generated_patch:
        patch_content = f"\nThe final patch is:\n{generated_patch}"
        compressed_message += patch_content

        # Print the compressed output
        logger.info(
            format_section_header(
                "problem_solver", compressed_message, current_sample_index
            )
        )

    problem_solver_outputs = state.get("problem_solver_outputs", [])
    generated_patches = state.get("generated_patches", [])

    return Command(
        goto="problem_solver",
        update={
            "messages": new_messages,
            "problem_solver_outputs": problem_solver_outputs + [compressed_message],
            "generated_patches": generated_patches + [generated_patch],
        },
    )


def problem_solver_aggregator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["problem_solver"]]:
    """
    Problem solver aggregator will aggregate the solution implementation from the list of solutions based on the knowledge.
    """
    problem_solver_outputs = state.get("problem_solver_outputs", [])
    problem_solver_final_output = "dummy_problem_solver_output"
    return Command(
        goto="problem_solver",
        update={"problem_solver_final_output": problem_solver_final_output},
    )


def post_processing_node(state: State, config: RunnableConfig):
    """
    Post processing will post process the output.
    """
    logger.info("arrive at post_processing")
    return {}


# Graph construction
def _build_graph():
    """Build and return the StateGraph."""
    builder = StateGraph(State)
    builder.add_edge(START, "input_handler")
    builder.add_node("input_handler", input_handler_node)
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("knowledge_extractor", knowledge_extractor_node)
    builder.add_node("problem_decoder", problem_decoder_node)
    builder.add_node("problem_decoder_sampler", problem_decoder_sampler_node, retry_policy=RetryPolicy(max_attempts=2, initial_interval=10.0, backoff_factor=2.0))
    builder.add_node("problem_decoder_aggregator", problem_decoder_aggregator_node)
    builder.add_node("solution_mapper", solution_mapper_node)
    builder.add_node("solution_mapper_sampler", solution_mapper_sampler_node, retry_policy=RetryPolicy(max_attempts=2, initial_interval=10.0, backoff_factor=2.0))
    builder.add_node("solution_mapper_aggregator", solution_mapper_aggregator_node)
    builder.add_node("problem_solver", problem_solver_node)
    builder.add_node("problem_solver_sampler", problem_solver_sampler_node, retry_policy=RetryPolicy(max_attempts=2, initial_interval=10.0, backoff_factor=2.0))
    builder.add_node("problem_solver_aggregator", problem_solver_aggregator_node)

    builder.add_node("post_processing", post_processing_node)
    builder.add_edge("post_processing", END)

    # Compile the graph
    return builder.compile()


# Create the graph
graph = _build_graph()


async def run_knowledge_workflow_async(
    instance_id: str,
    model_name: str = "claude-4-sonnet",
    cache_dir: Optional[str] = None,
    decoder_iterations: int = 3,
    mapper_iterations: int = 1,
    solver_iterations: int = 1,
):
    """Run the knowledge workflow asynchronously for a single SWE-bench instance.

    Args:
        instance_id: The SWE-bench instance ID or GitHub issue URL
        model_name: The model name to use
        cache_dir: The directory to cache the workflow
        decoder_iterations: Number of target iterations for problem decoder
        mapper_iterations: Number of target iterations for solution mapper
        solver_iterations: Number of target iterations for problem solver
    Returns:
        The final state after the workflow completes
    """
    if not instance_id:
        raise ValueError("Instance ID could not be empty")

    logger.info(f"Starting async knowledge workflow with instance: {instance_id}")

    # Initialize state with the instance - properly typed
    initial_state = {
        "preset": instance_id,
        "cache_dir": cache_dir,
        "decoder_iterations": decoder_iterations,
        "mapper_iterations": mapper_iterations,
        "solver_iterations": solver_iterations,
    }

    date_month_day = datetime.datetime.now().strftime("%Y%m%d")
    # Configuration for the workflow - properly typed
    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"thread_{instance_id}_{model_name}",
            "instance_id": instance_id,
        },
        "run_id": uuid.uuid4(),
        "tags": ["knowledge_tts_v1"],
        "run_name": f"swe-{model_name}-knowl_tts_v1-{date_month_day}",
        "recursion_limit": 200,
    }

    last_message_cnt = 0
    final_state = None

    try:
        async for chunk in graph.astream(
            input=cast(State, initial_state), config=config, stream_mode="values"
        ):
            try:
                final_state = chunk

                # Process messages if they exist
                if isinstance(chunk, dict) and "messages" in chunk:
                    if len(chunk["messages"]) > last_message_cnt:
                        last_message_cnt = len(chunk["messages"])
                        message = chunk["messages"][-1]
                        if hasattr(message, "content"):
                            print(f"Message: {message.content}")
                        elif hasattr(message, "pretty_print"):
                            message.pretty_print()
                        else:
                            print(f"Message: {message}")

            except Exception as e:
                logger.error(f"Error processing stream output: {e}")
                print(f"Error processing output: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error running async workflow: {e}")
        raise

    logger.info("Async knowledge workflow completed successfully")
    return final_state


def run_knowledge_workflow(
    instance_id: str,
    model_name: str = "claude-4-sonnet",
    cache_dir: Optional[str] = None,
    decoder_iterations: int = 3,
    mapper_iterations: int = 1,
    solver_iterations: int = 1,
):
    """Run the knowledge workflow for a single SWE-bench instance (synchronous wrapper).

    Args:
        instance_id: The SWE-bench instance ID or GitHub issue URL
        model_name: The model name to use
        cache_dir: The directory to cache the workflow
        decoder_iterations: Number of target iterations for problem decoder
        mapper_iterations: Number of target iterations for solution mapper
        solver_iterations: Number of target iterations for problem solver

    Returns:
        The final state after the workflow completes
    """
    return asyncio.run(run_knowledge_workflow_async(instance_id, model_name, cache_dir, decoder_iterations, mapper_iterations, solver_iterations))


async def main(instance_id: str, model_name: str, cache_dir: Optional[str] = None, decoder_iterations: int = 3, mapper_iterations: int = 1, solver_iterations: int = 1):
    """Main async function to run the workflow."""
    # Example usage with a single instance

    try:
        final_state = await run_knowledge_workflow_async(
            instance_id=instance_id, model_name=model_name, cache_dir=cache_dir,
            decoder_iterations=decoder_iterations, mapper_iterations=mapper_iterations, solver_iterations=solver_iterations
        )

        # Print final results
        if final_state:
            print("\n" + "=" * 50)
            print("WORKFLOW COMPLETED")
            print("=" * 50)

    except Exception as e:
        logger.error(f"Failed to run workflow: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    # Debug: Print sys.argv to see what arguments are being passed
    import sys

    print(f"sys.argv: {sys.argv}")

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run the knowledge TTS workflow for SWE-bench instances"
    )
    parser.add_argument(
        "--instance-id",
        default="django__django-13821",
        help="The SWE-bench instance ID or GitHub issue URL (default: django__django-13821)",
    )
    parser.add_argument(
        "--model-name",
        default="claude-4-sonnet",
        help="The model name to use (default: claude-4-sonnet)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="The directory to cache the workflow (default: None)",
    )
    parser.add_argument(
        "--clean-log",
        default=True,
        help="Whether to clean the history log file (default: True)",
    )
    parser.add_argument(
        "--decoder-iterations",
        type=int,
        default=3,
        help="Number of target iterations for problem decoder (default: 3)",
    )
    parser.add_argument(
        "--mapper-iterations",
        type=int,
        default=1,
        help="Number of target iterations for solution mapper (default: 1)",
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=1,
        help="Number of target iterations for problem solver (default: 1)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"Argument parsing failed with exit code: {e.code}")
        print("Available arguments:")
        parser.print_help()
        sys.exit(e.code)

    # Create cache directory if provided
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        log_dir = os.path.join(args.cache_dir, "log") if args.cache_dir else "log"
        os.makedirs(log_dir, exist_ok=True)
        # clean up the log file
        if os.path.exists(f"{log_dir}/{args.instance_id}.log") and args.clean_log:
            os.remove(f"{log_dir}/{args.instance_id}.log")

        configure_logging(
            level=logging.DEBUG if args.debug else logging.INFO,
            log_dir=log_dir,
            log_file=f"{args.instance_id}.log",
        )
    else:
        configure_logging(
            level=logging.DEBUG if args.debug else logging.INFO,
            log_file="knowledge_tts_workflow_for_swebench_v1.log",
        )

    # Run the async main function
    asyncio.run(main(args.instance_id, args.model_name, args.cache_dir, args.decoder_iterations, args.mapper_iterations, args.solver_iterations))
