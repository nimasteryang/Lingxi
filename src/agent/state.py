"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from langgraph.graph import MessagesState, add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langchain_core.messages import AnyMessage
from typing import Annotated, List, Optional


@dataclass
class CustomState(MessagesState):
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    last_agent: Optional[str] = None
    next_agent: Optional[str] = None
    summary: Optional[str] = None
    relevant_convo_history: Optional[List[str]] = None
    human_in_the_loop: Optional[bool] = True  # Default to True to skip HIL
    preset: Optional[str] = None
    # issue_link: Optional[str] = None
    previous_processed_messages_len: int = 0  # Used for new version of summary
    previous_messages_len: int = 0  # Used for new version of summary
    context: dict = None
    runtime_info: dict = None
    shorten_messages: Annotated[list[AnyMessage], add_messages] = None
    cache_dir: Optional[str] = None
    issue_description: Optional[str] = None
