from typing import List

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class RelevantFileExplanation(BaseModel):
    file_path: str = Field(description="The filepath of the relevant file.")
    explanation: str = Field(description="The explanation of how the file is relevant to the query.")


class RelevantFileExplanations(BaseModel):
    relevant_file_explanations: List[RelevantFileExplanation]


relevant_file_explanations_parser = JsonOutputParser(pydantic_object=RelevantFileExplanations)


class RelevantMessageIds(BaseModel):
    ids: List[str] = Field(description="The list of relevant message IDs.")


relevant_message_ids_parser = JsonOutputParser(pydantic_object=RelevantMessageIds)
