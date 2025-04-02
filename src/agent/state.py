"""Defines the custom state structures for the prototype."""

from __future__ import annotations

from dataclasses import dataclass
from langgraph.graph import MessagesState, add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langchain_core.messages import AnyMessage
from typing import Annotated, List, Optional


@dataclass
class CustomState(MessagesState):
    last_agent: Optional[str] = None
    next_agent: Optional[str] = None
    summary: Optional[str] = None
    human_in_the_loop: Optional[bool] = True  # Default to True to skip HIL
    preset: Optional[str] = None
    issue_description: Optional[str] = None
