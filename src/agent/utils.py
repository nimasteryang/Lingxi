
import asyncio
from dataclasses import dataclass
import json
import logging
import os
from pprint import pprint
import re
import time
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
import yaml

from src.agent.logging_config import get_logger
from src.agent.runtime_config import RuntimeConfig, RuntimeType
from src.agent.tool_set.utils import get_runtime_config
from src.utils.format_utils import format_analysis_for_llm

# Setup logging
logger = get_logger(__name__)


"""
Defines various util functions for the prototype."""


class UndefinedValueError(ValueError):
    """
    A custom exception raised when a variable is not defined.

    Args:
        variable_name (str): The name of the undefined variable
        message (str, optional): Custom error message
    """

    def __init__(self, variable_name, message=None):
        if message is None:
            message = f"`{variable_name}` is required and not defined in `.env` environment variables."

        self.variable_name = variable_name

        super().__init__(message)


def stage_message_processor(messages):
    """Based on XY's `message_processor` with few additions. It is used for newest version of summarizer/summary."""
    message_json = {}
    step = 0

    index = 0
    found_human_feedback = False
    messages = [
        vars(message) if not isinstance(message, dict) else message
        for message in messages
    ]
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

def compress_agent_thinking_observation_action(messages: List, agent_name: str) -> str|None:
    """
    Compress the agent thinking, observations and actions into a single, structured message with YAML-like formatting.

    Args:
        messages (List): List of messages including AI messages.
        agent_name (str): The name of the agent performing actions.

    Returns:
        Optional[List]: A compressed AIMessage with YAML-like structure,
                        or None if an error occurs.
    """
    if len(messages) == 1:
        logger.warning(f"No action found for {agent_name}")
        return messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
    elif len(messages) == 0:
        logger.warning(f"No action found for {agent_name}")
        return None

    if "problem_solver" in agent_name.lower():
        agent_type = "problem solver"
        default_action = f"{agent_type} starts to solve the problem:"
    elif "problem_decoder" in agent_name.lower():
        agent_type = "problem decoder"
        default_action = f"{agent_type} starts to decode the problem:"
    elif "solution_mapper" in agent_name.lower():
        agent_type = "solution mapper"
        default_action = f"{agent_type} starts to map the solution:"
    elif "reviewer" in agent_name.lower():
        agent_type = "reviewer"
        default_action = f"{agent_type} starts to review the solution:"
    else:
        logger.warning(f"Unknown agent name: {agent_name}")
        default_action = f"{agent_name} starts to work on the problem:"
    
    # Initialize YAML structure and action tracking
    yaml_structure = f"{default_action}\nactions:"
    action_step = 1
    formatted_steps = []
    total_actions = 0
    action_frequency = {}
    for i, msg in enumerate(messages):
        try:
            # Process AI messages that are not the last message
            if isinstance(msg, AIMessage) and i != len(messages) - 1:
                step_entry = {"step": action_step}
                thinking = None
                observation = None
                action = None
                for part in msg.content:
                    if part.get("type") == "thinking" and thinking is None:
                        thinking = part.get("thinking")
                    elif part.get("type") == "text":
                        observation = part.get("text")
                    elif part.get("type") == "tool_use":
                        action_name = part.get("name")
                        if not part.get("input"):
                            action_input = part.get("partial_json")
                        else:
                            action_input = part.get("input")
                        action = f"{action_name}({action_input})"
                        # Track action frequency
                        if action_name:
                            total_actions += 1
                            action_frequency[action_name] = action_frequency.get(action_name, 0) + 1
                    
                # fallback to `tool_calls` if no tool_use found in content
                if not action and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        action_name = tool_call['name']
                        action = f"{action_name}({tool_call['args']})"
                        # Track action frequency
                        if action_name:
                            total_actions += 1
                            action_frequency[action_name] = action_frequency.get(action_name, 0) + 1
                if thinking:
                    step_entry["thinking"] = thinking.strip()
                if observation:
                    step_entry["observation"] = observation.strip()
                if action:
                    step_entry["action"] = action
                    
                action_step += 1
                formatted_steps.append(step_entry)

            # If last message, add the conclusion
            elif i == len(messages) - 1:
                step_entry = {"step": action_step}
                thinking = None
                conclusion = None
                
                if isinstance(msg.content, str):
                    conclusion = msg.content
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if part.get("type") == "thinking":
                            thinking = part.get("thinking")
                        elif part.get("type") == "text":
                            conclusion = part.get("text")
                if thinking:
                    step_entry["thinking"] = thinking.strip()
                if conclusion:
                    step_entry["conclusion"] = conclusion
                formatted_steps.append(step_entry)

        except Exception as e:
            logger.error(f"Error processing message at index {i}: {e}")
            logger.error(f"Message content: {msg}")

    # Log action statistics
    logger.info(f"Agent '{agent_name}' - Total actions: {total_actions}")
    if action_frequency:
        logger.info(f"Agent '{agent_name}' - Action frequency: {action_frequency}")
    else:
        logger.info(f"Agent '{agent_name}' - No actions found")

    compressed_message = format_analysis_for_llm(formatted_steps, 'markdown')

    return compressed_message


def replay_agent_action(messages: List, config: RunnableConfig) -> Optional[List]:
    """
    Replay agent actions from LangGraph messages by extracting and executing tool calls.
    
    Args:
        messages (List): List of LangGraph messages containing tool calls
        config (RunnableConfig): Runtime configuration for tool execution
        
    Returns:
        Optional[List]: List of ToolMessage results from replayed actions, or None if error
    """
    rc = get_runtime_config(config)
    assert rc.initialized
    
    logger.info(f"Starting replay of agent actions with runtime_type: {rc.runtime_type}")
    
    # Extract tool calls from messages
    tool_calls_to_replay = []
    
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            # Check for tool_calls attribute (standard LangGraph format)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls_to_replay.append({
                        'message_index': i,
                        'tool_name': tool_call.get('name', 'unknown_tool'),
                        'tool_args': tool_call.get('args', {}),
                        'tool_call_id': tool_call.get('id', f'replay_call_{len(tool_calls_to_replay)}'),
                        'source': 'tool_calls'
                    })
            
            # Also check content for tool calls (alternative format)
            elif isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict) and content_item.get('type') == 'tool_use':
                        tool_calls_to_replay.append({
                            'message_index': i,
                            'tool_name': content_item.get('name', 'unknown_tool'),
                            'tool_args': content_item.get('input', {}),
                            'tool_call_id': content_item.get('id', f'replay_call_{len(tool_calls_to_replay)}'),
                            'source': 'content'
                        })
    
    if not tool_calls_to_replay:
        logger.warning("No tool calls found in messages to replay")
        return []
    
    logger.info(f"Found {len(tool_calls_to_replay)} tool calls to replay")
    
    # Replay tool calls based on runtime type
    replayed_results = []
    
    try:
        if rc.runtime_type == RuntimeType.LOCAL:
            replayed_results = _replay_tools_local(tool_calls_to_replay, rc, config)
        elif rc.runtime_type == RuntimeType.SWEREX:
            replayed_results = _replay_tools_swerex(tool_calls_to_replay, rc, config)
        else:
            logger.error(f"Unsupported runtime type: {rc.runtime_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error during tool replay: {e}", exc_info=True)
        return None
    
    logger.info(f"Successfully replayed {len(replayed_results)} tool calls")
    return replayed_results


def _replay_tools_local(tool_calls: List, rc: RuntimeConfig, config: RunnableConfig) -> List:
    """Replay tools in local environment"""
    results = []
    
    for tool_call in tool_calls:
        try:
            result = _execute_tool_call(tool_call, rc, config)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error replaying tool {tool_call['tool_name']}: {e}")
            # Create error result
            error_result = ToolMessage(
                content=f"Error executing {tool_call['tool_name']}: {str(e)}",
                tool_call_id=tool_call['tool_call_id'],
                name=tool_call['tool_name']
            )
            results.append(error_result)
    
    return results


def _replay_tools_docker(tool_calls: List, rc: RuntimeConfig, config: RunnableConfig) -> List:
    """Replay tools in Docker environment"""
    results = []
    
    for tool_call in tool_calls:
        try:
            result = _execute_tool_call(tool_call, rc, config)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error replaying tool {tool_call['tool_name']} in Docker: {e}")
            # Create error result
            error_result = ToolMessage(
                content=f"Error executing {tool_call['tool_name']} in Docker: {str(e)}",
                tool_call_id=tool_call['tool_call_id'],
                name=tool_call['tool_name']
            )
            results.append(error_result)
    
    return results


def _replay_tools_swerex(tool_calls: List, rc: RuntimeConfig, config: RunnableConfig) -> List:
    """Replay tools in SWEREx environment"""
    results = []
    
    for tool_call in tool_calls:
        try:
            result = _execute_tool_call(tool_call, rc, config)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error replaying tool {tool_call['tool_name']} in SWEREx: {e}")
            # Create error result
            error_result = ToolMessage(
                content=f"Error executing {tool_call['tool_name']} in SWEREx: {str(e)}",
                tool_call_id=tool_call['tool_call_id'],
                name=tool_call['tool_name']
            )
            results.append(error_result)
    
    return results


def _execute_tool_call(tool_call: dict, rc: RuntimeConfig, config: RunnableConfig) -> Optional[ToolMessage]:
    """Execute a single tool call and return the result as ToolMessage"""
    tool_name = tool_call['tool_name']
    tool_args = tool_call['tool_args']
    tool_call_id = tool_call['tool_call_id']
    
    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
    
    try:
        # Import and execute common tools based on tool name
        if tool_name == "view_directory":
            # pass view directory
            # from src.agent.tool_set.sepl_tools import view_directory
            # result = view_directory.invoke(tool_args, config=config)
            result = "skip view directory"
            
        elif tool_name == "view_file_content":
            # from src.agent.tool_set.sepl_tools import view_file_content
            # result = view_file_content.invoke(tool_args, config=config)
            result = "skip view file content"
            
        elif tool_name == "search_files_by_keywords":
            # from src.agent.tool_set.sepl_tools import search_files_by_keywords
            # result = search_files_by_keywords.invoke(tool_args, config=config)
            result = "skip search files by keywords"
            
        elif tool_name == "run_shell_cmd":
            from src.agent.tool_set.sepl_tools import run_shell_cmd
            result = run_shell_cmd.invoke(tool_args, config=config)
            
        elif tool_name == "str_replace_editor" or tool_name == "str_replace_based_edit_tool":
            from src.agent.tool_set.edit_tool import str_replace_editor
            result = str_replace_editor.invoke(tool_args, config=config)
            
        elif tool_name == "ask_repository_agent":
            # from src.agent.tool_set.deepwiki_tool import ask_repository
            # result = ask_repository.invoke(tool_args, config=config)
            result = "skip ask repository agent"
            
        else:
            logger.warning(f"Unknown tool: {tool_name}, skipping")
            result = f"Unknown tool: {tool_name}"
        
        # Create ToolMessage from result
        tool_message = ToolMessage(
            content=str(result) if result is not None else "No output",
            tool_call_id=tool_call_id,
            name=tool_name
        )
        
        return tool_message
        
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        # Return error as ToolMessage
        return ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id=tool_call_id,
            name=tool_name
        )