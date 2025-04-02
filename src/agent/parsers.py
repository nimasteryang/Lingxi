"""
Defines parsers used to extract information from LLM responses
"""

from typing import List

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class RelevantFileExplanation(BaseModel):
    """Parse LLM output to construct a file path string and explanation string."""

    file_path: str = Field(description="The filepath of the relevant file.")
    explanation: str = Field(
        description="The explanation of how the file is relevant to the query."
    )


class RelevantFileExplanations(BaseModel):
    """Parse LLM output to construct a list of RelevantFileExplanation"""

    relevant_file_explanations: List[RelevantFileExplanation]


relevant_file_explanations_parser = JsonOutputParser(
    pydantic_object=RelevantFileExplanations
)
