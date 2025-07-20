"""Defines the custom state structures for the prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Annotated, TypedDict
from dataclasses import dataclass, replace
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import AnyMessage
from operator import add
from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel

@dataclass
class CustomState(MessagesState):
    last_agent: Optional[str] = None
    next_agent: Optional[str] = None
    summary: Optional[str] = None
    human_in_the_loop: Optional[bool] = True  # Default to True to skip HIL
    preset: Optional[str] = None
    issue_description: Optional[str] = None


def messages_reducer(left: list, right: list) -> list:
    """Custom messages reducer with incremental caching support.
    
    This reducer automatically marks the last content block in the latest message
    with cache_control to enable incremental caching in multi-turn conversations.
    Only the most recent message will have cache_control to avoid exceeding the
    4-block limit imposed by Anthropic's API.
    """
    # First apply the standard add_messages logic
    result = add_messages(left, right)
    
    # Remove cache_control from all existing messages
    for message in result:
        if isinstance(message.content, list):
            for content_block in message.content:
                if isinstance(content_block, dict) and "cache_control" in content_block:
                    del content_block["cache_control"]
    
    # Then update the last message to enable caching
    if len(result) > 0:
        last_message = result[-1]
        if isinstance(last_message.content, list) and len(last_message.content) > 0:
            # Mark the last content block with cache_control
            last_message.content[-1]["cache_control"] = {"type": "ephemeral"}
        elif isinstance(last_message.content, str):
            # Convert string to list format with cache_control
            last_message.content = [
                {
                    "text": last_message.content,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
    
    return result

class DevKnowledge(BaseModel):
    repo_name: str
    issue_id: str
    issue_description: str
    patch: str
    patch_commit: str
    relevance_score: float
    dev_knowledge: str


class State(TypedDict):
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps = 200
    messages: Annotated[list[AnyMessage], messages_reducer]
    solution_mapper_messages: list[list[AnyMessage]]
    preset: Optional[str]
    issue_description: Optional[str]
    cache_dir: Optional[str]
    last_node: Optional[str]
    current_sample_index: Optional[int]
    current_knowledge: Optional[DevKnowledge]
    dev_knowledges: list[DevKnowledge]
    problem_decoder_outputs: list[str]
    problem_decoder_final_output: str
    solution_mapper_outputs: list[str]
    solution_mapper_final_output: str
    problem_solver_outputs: list[str]
    problem_solver_final_output: str
    generated_patches: list[str]
    decoder_iterations: Optional[int]
    mapper_iterations: Optional[int]
    solver_iterations: Optional[int]