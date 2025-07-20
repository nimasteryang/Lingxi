import pandas as pd
import re
import os
import dotenv
try:
    from lxml import etree  # type: ignore
except ImportError:
    print("Warning: lxml not available, XML formatting will be limited")
    etree = None
from typing import Annotated, Dict, Any, Optional, Union
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.agent.tool_set.sepl_tools import view_directory, search_files_by_keywords, view_file_content
from src.agent import runtime_config
from swerex.runtime.abstract import CreateBashSessionRequest, BashAction, Command, WriteFileRequest, CloseBashSessionRequest
import asyncio
import uuid
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.prompt.dev_knowledge_prompts import DEV_KNOWLEDGE_ANALYSIS_SYSTEM_PROMPT, DEV_KNOWLEDGE_ANALYSIS_USER_PROMPT, DEV_KNOWLEDGE_SUMMARY_SYSTEM_PROMPT, DEV_KNOWLEDGE_SUMMARY_USER_PROMPT, GUIDING_QUESTIONS_PROMPT
from src.agent.logging_config import get_logger
from src.agent.state import DevKnowledge, State

# Setup logging
logger = get_logger(__name__)

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR = os.path.join(PROJECT_ROOT_DIR, "data","history_issue_knowledge","cache_20250710")
# create the directory if it does not exist
if not os.path.exists(HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR):
    os.makedirs(HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR)

SUMMARY_CACHE_DIR = os.path.join(PROJECT_ROOT_DIR,"dev_knowl", "cache", "summaries")

print(PROJECT_ROOT_DIR)
print(HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR)
assert(os.path.exists(HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR))

def pretty_lxml_from_str(xml_str: str, encoding: str = "unicode") -> str:
    """Format XML string with proper indentation and encoding.
    
    Args:
        xml_str: The XML string to format
        encoding: Output encoding (default: "unicode")
    
    Returns:
        Formatted XML string
    """
    if etree is None:
        return xml_str
        
    parser = etree.XMLParser(remove_blank_text=True)
    try:
        root = etree.fromstring(xml_str.encode(), parser)
        return etree.tostring(root, pretty_print=True, encoding=encoding)
    except etree.XMLSyntaxError as e:
        logger.warning(f"XML syntax error in pretty_lxml_from_str: {e}")
        return xml_str

def format_xml_content(tag: str, content: str) -> str:
    """Format content into a properly structured XML tag.
    
    Args:
        tag: The XML tag name
        content: The content to wrap in XML tags
        
    Returns:
        Formatted XML string with proper newlines and indentation
    """
    if content is None:
        return ""
    
    # Basic XML escaping for raw text content (only if not already XML)
    if not content.strip().startswith('<'):
        content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Clean up the content - remove extra whitespace and normalize line breaks
    content = content.strip()
    
    # Format with proper newlines: opening tag, content (indented), closing tag
    formatted_xml = f"<{tag}>\n{content}\n</{tag}>\n"
    
    return formatted_xml

def extract_xml_content(xml_string: str, extract_str: str) -> Optional[str]:
    """Extract content from XML-like tags in a string."""
    match = re.search(rf"<{extract_str}>(.*?)</{extract_str}>", xml_string, re.DOTALL)
    return match.group(1).strip() if match else None

def generate_content_hash(content: str) -> str:
    """
    Generate a hash for content to use as cache identifier.
    
    Args:
        content: Content to hash
        
    Returns:
        Hash string
    """
    import hashlib
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]

def load_summary_from_cache(repo: str, issue_identifier: str) -> Optional[str]:
    """
    Load summary from cache if available.
    
    Args:
        repo: Repository name
        issue_identifier: Issue number or identifier
        
    Returns:
        Cached summary or None if not found
    """
    try:
        cache_path = os.path.join(SUMMARY_CACHE_DIR, f"{repo}-{issue_identifier}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                summary = f.read().strip()
                if len(summary) >= 50:  # Minimum summary length
                    return summary
                else:
                    logger.warning(f"Cache entry too short for {repo}-{issue_identifier}, ignoring")
                    return None
        return None
    except Exception as e:
        logger.error(f"Error loading summary from cache for {repo}-{issue_identifier}: {e}")
        return None

def create_llm() -> ChatAnthropic:
    """Create a ChatAnthropic instance with proper parameters."""
    # Use the same pattern as other files in the codebase
    return ChatAnthropic(
            model_name="claude-3-5-sonnet-latest",
            temperature=1,
            max_tokens_to_sample=8096,
            timeout=360,
            stop=None,
        )
    # return ChatAnthropic(
    #         model_name="claude-sonnet-4-20250514",
    #         temperature=1,
    #         max_tokens_to_sample=9216,
    #         thinking={"type": "enabled", "budget_tokens": 1024},
    #         timeout=360,
    #         stop=None,
    #     )

def contruct_history_knowledge(project_name: str, issue_id: str, issue_description: str, 
                              patch: str, patch_commit: str, instance_id: str) -> str:
    """Construct history knowledge from cached analysis or generate new analysis."""
    
    analysis_cache_id = f"{project_name}_{issue_id}_step_analysis.txt"
    analysis_cache_path = os.path.join(HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR, analysis_cache_id)
    
    if os.path.exists(analysis_cache_path):
        logger.info(f"Loading cached analysis: {analysis_cache_id}")
        with open(analysis_cache_path, "r") as f:
            step_analysis_knowledge = f.read()
    else:
        logger.info(f"Generating new analysis for: {analysis_cache_id}")
        # Generate new analysis if cache is not found
        step_analysis_knowledge = _generate_step_analysis(
            project_name, issue_id, issue_description, patch, patch_commit, instance_id
        )
        with open(analysis_cache_path, "w") as f:
            f.write(step_analysis_knowledge)

    # Get or generate summary
    summary_cache_id = f"{project_name}_{issue_id}_step_summary.txt"
    summary_cache_path = os.path.join(HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR, summary_cache_id)
    
    if os.path.exists(summary_cache_path):
        logger.info(f"Loading cached summary: {summary_cache_id}")
        with open(summary_cache_path, "r") as f:
            step_summary_knowledge = f.read()
    else:
        logger.info(f"Generating new summary for: {summary_cache_id}")
        step_summary_knowledge = _generate_step_summary(step_analysis_knowledge, patch)
        with open(summary_cache_path, "w") as f:
            f.write(step_summary_knowledge)
    
    return step_summary_knowledge

def _generate_step_analysis(project_name: str, issue_id: str, issue_description: str, 
                           patch: str, patch_commit: str, instance_id: str) -> str:
    """Generate step analysis using the agent."""
    try:
        # Environment setup
        analysis_runtime = runtime_config.RuntimeConfig()
        analysis_runtime.load_from_swe_rex_docker_instance(instance_id, checkout_commit=patch_commit)
        
        # Check if runtime is available
        if analysis_runtime.swe_rex_deployment is None or analysis_runtime.swe_rex_deployment.runtime is None:
            raise ValueError("Runtime deployment not available")
            
        
        # Create analysis agent
        analysis_agent_tool_set = [view_directory, search_files_by_keywords, view_file_content]
        prompt = DEV_KNOWLEDGE_ANALYSIS_SYSTEM_PROMPT
        user_prompt = DEV_KNOWLEDGE_ANALYSIS_USER_PROMPT.format(
            ISSUE_DESCRIPTION=issue_description, PATCH=patch
        )
        
        analysis_agent = create_react_agent(
            create_llm(),
            tools=analysis_agent_tool_set,
            prompt=prompt,
            name = "history_issue_knowledge_analysis_agent",
            state_schema=State,
        )

        # Create proper config dict - this is what the agent expects
        config_dict = {
            "recursion_limit": 100,
            "configurable": {
                "thread_id": f"thread_{project_name}_{issue_id}_step_analysis",
                "runtime_object": analysis_runtime,
            }
        }
        
        step_analysis_messages = analysis_agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]}, 
            config=config_dict  # type: ignore
        )

        new_messages = step_analysis_messages["messages"]

        # stop the swerex deployment
        if analysis_runtime.runtime_type == runtime_config.RuntimeType.SWEREX:
            deployment = analysis_runtime.swe_rex_deployment
            logger.info("History issue knowledge analysis agent: Stopping SWEREX deployment")
            asyncio.run(deployment.stop())
        
        if isinstance(new_messages[-1].content, list):
            analysis_messages = new_messages[-1].content[-1]['text']
        else:
            analysis_messages = new_messages[-1].content

        return analysis_messages
        
    except Exception as e:
        logger.error(f"Error generating step analysis: {e}")
        return f"Error generating analysis: {str(e)}"

def _generate_step_summary(step_analysis_knowledge: str, patch: str) -> str:
    """Generate step summary from analysis."""
    try:
        llm = create_llm()
        prompt = DEV_KNOWLEDGE_SUMMARY_SYSTEM_PROMPT
        user_prompt = DEV_KNOWLEDGE_SUMMARY_USER_PROMPT.format(
            HISTORIC_ISSUE_ANALYSIS=step_analysis_knowledge, PATCH=patch
        )
        messages = [SystemMessage(content=prompt), HumanMessage(content=user_prompt)]
        step_summary_messages = llm.invoke(messages)
        
        # Extract content properly
        content = step_summary_messages.content
        if isinstance(content, list) and len(content) > 0:
            # Handle case where content is a list
            last_item = content[-1]
            if isinstance(last_item, dict):
                content = last_item.get('text', '')
            else:
                content = str(last_item)
        else:
            content = str(content)
            
        logger.info(f"Step summary knowledge generated: {len(content)} chars")
        return content
        
    except Exception as e:
        logger.error(f"Error generating step summary: {e}")
        return f"Error generating summary: {str(e)}"

def construct_dev_knowledge_prompt_version_1(knowledge_content: str) -> str:
    """Construct the dev knowledge prompt with the given knowledge content."""
    return (
        "Consider the steps and hints in the following knowledge wrap in <dev_knowledge> tag provided by a human expert in solving a similar history issue. You should adapt the steps and hints to the current issue.\n"
        + "<dev_knowledge>"
        + knowledge_content
        + "</dev_knowledge>\n"
    )

DECODER_KNOWLEDGE_PROMPT_VERSION_2 = """
You have privileged access to the following project level development knowledge:    
<dev_knowledge>{knowledge_content}</dev_knowledge>
Above developement knowledge are extracted from the history issues that are similar to the current issue.

How to use it:
1. You should first perform the READING and EXPLORATION phases of the current issue.
2. Then, consult these sections and make sure you have covered all the aspects in the knowledge. 
    2.1 If you have not covered the <general_root_cause_analysis_steps> section, you should continue to explore the repository if necessary.
    2.2 If you have not considered the <involved_components> or <relevant_architecture> section, you should reconsider if the current issue is related to the components/architecture in the knowledge and continue to explore the repository if necessary.
3. Consider other sections in the knowledge when you perform the issue analysis.
"""
MAPPER_KNOWLEDGE_PROMPT_VERSION_2 = """
You have privileged access to the following project level development knowledge:    
<dev_knowledge>{knowledge_content}</dev_knowledge>
Above developement knowledge are extracted from the history issues that are similar to the current issue.

How to use it:
1. When you perform the FIX ANALYSIS phase, you should consider the <general_fix_pattern> and <design_patterns_and_coding_practices> section.
2. After you have generated the fix solution, make sure the <summary_of_fix_checklist> section is covered.
3. Consider the <additional_concepts> section knowledge if applicable.
"""

SOLVER_KNOWLEDGE_PROMPT_VERSION_2 = """
You have privileged access to the following project level development knowledge:    
<dev_knowledge>{knowledge_content}</dev_knowledge>
Above developement knowledge are extracted from the history issues that are similar to the current issue.
When you edit the code, you should consider the <design_patterns_and_coding_practices> section.
"""

def construct_dev_knowledge_prompt_version_2(knowledge_content: str, role: str) -> str:
    """Construct the dev knowledge prompt with the given knowledge content."""
    if "decoder" in role.lower():
        return DECODER_KNOWLEDGE_PROMPT_VERSION_2.format(knowledge_content=knowledge_content)
    elif "mapper" in role.lower():
        return MAPPER_KNOWLEDGE_PROMPT_VERSION_2.format(knowledge_content=knowledge_content)
    elif "solver" in role.lower():
        return SOLVER_KNOWLEDGE_PROMPT_VERSION_2.format(knowledge_content=knowledge_content)
    else:
        raise ValueError(f"Invalid role: {role}")

def get_dev_knowledge_design_version_1(issue_description: str, instance_id: str, role: str) -> str:
    """Get development knowledge design version 1 (legacy function)."""
    
    issue_number = int(instance_id.split("-")[-1])
    
    history_issue_file = os.path.join(PROJECT_ROOT_DIR, "dev_knowl", "data", "20250619-SwebenchCustom-WithHistoricIssue-Reranking.jsonl")
    print(f"History issue file: {history_issue_file}")
    assert os.path.exists(history_issue_file), f"History issue file does not exist: {history_issue_file}"
    df = pd.read_json(history_issue_file, lines=True)
    
    retrieved_issues = df[df["instance_id"] == instance_id]
    retrieved_issues = retrieved_issues[retrieved_issues["retrieved_issue_number"] < issue_number]

    if len(retrieved_issues) == 0:
        print(f"No retrieved issues found for instance_id: {instance_id}")
        return ""
    
    retrieved_issues = retrieved_issues.sort_values(by="reranked_position")
    assert len(retrieved_issues) > 0, f"No retrieved issues found for instance_id: {instance_id}"
    most_similar_issue = retrieved_issues.iloc[0]

    project_name = most_similar_issue["repo"].replace("/", "+")
    issue_id = most_similar_issue["retrieved_issue_number"]
    issue_description = most_similar_issue["retrieved_issue_desc"]
    patch = most_similar_issue["retrieved_patch"]
    patch_commit = most_similar_issue["retrieved_commit_id"]

    knowledge = contruct_history_knowledge(
        project_name=project_name,
        issue_id=issue_id,
        issue_description=issue_description,
        patch=patch,
        patch_commit=patch_commit,
        instance_id=instance_id,
    )

    # Define role-specific XML tags
    role_tags = _get_role_specific_tags(role)
    
    # Word count of the knowledge
    word_count = len(knowledge.split())
    print(f"Word count of the knowledge: {word_count}")

    if knowledge is None:
        raise ValueError(f"No knowledge found for instance_id: {instance_id}")
    
    retrieved_knowledge = {}
    for tag in role_tags:
        tag_knowledge = extract_xml_content(knowledge, tag)
        if tag_knowledge is not None:
            retrieved_knowledge[tag] = tag_knowledge

    # Format as XML
    retrieved_knowledge_formatted = _format_knowledge_as_xml(retrieved_knowledge)
    return construct_dev_knowledge_prompt_version_2(retrieved_knowledge_formatted, role)

def _get_role_specific_tags(role: str) -> list[str]:
    """Get XML tags specific to the given role."""
    decoder_use_xml_tags = [
        "general_root_cause_analysis_steps",
        "bug_categorization",
        "relevant_architecture",
        "involved_components",
        "specific_involved_classes_functions_methods",
        "feature_or_functionality_of_issue",
        "additional_concepts",
    ]
    mapper_use_xml_tags = [
        "general_fix_pattern",
        "summary_of_fix_checklist",
        "design_patterns_and_coding_practices",
        "additional_concepts",
    ]
    solver_use_xml_tags = [
        "design_patterns_and_coding_practices",
    ]

    if "decoder" in role.lower():
        return decoder_use_xml_tags
    elif "mapper" in role.lower():
        return mapper_use_xml_tags
    elif "solver" in role.lower():
        return solver_use_xml_tags
    else:
        raise ValueError(f"Invalid role: {role}")

def _format_knowledge_as_xml(retrieved_knowledge: Dict[str, str]) -> str:
    """Format retrieved knowledge as XML."""
    try:
        formatted_parts = [
            format_xml_content(tag, content)
            for tag, content in retrieved_knowledge.items()
            if content is not None
        ]
        return "\n".join(formatted_parts).rstrip()
    except Exception as e:
        logger.error(f"Error in restructure to xml tags: {e}")
        # Fallback with proper newlines and spacing
        fallback_parts = [
            f"<{tag}>{content}</{tag}>"
            for tag, content in retrieved_knowledge.items()
            if content is not None
        ]
        return "\n\n".join(fallback_parts)

@tool
def get_dev_knowledge_design_version_2(
    issue_description: Annotated[str, "The description of the current issue to be solved"],
    instance_id: Annotated[str, "The instance ID of the current issue"],
    role: Annotated[str, "The role of the agent: 'decoder', 'mapper', or 'solver'"],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Get development knowledge from similar historical issues to help solve the current issue.
    
    This tool retrieves and processes knowledge from similar historical issues that have been 
    previously solved. It returns a dictionary with the current dev knowledge prompt, with room
    for additional knowledge aspects in the future.
    
    Args:
        issue_description: Description of the current issue
        instance_id: Instance ID in format 'repo__repo-number' (e.g., 'scikit-learn__scikit-learn-10908')
        role: The role requesting knowledge ('decoder', 'mapper', or 'solver')
        config: Runtime configuration
        
    Returns:
        Dictionary containing:
        - dev_knowledge_prompt: Ready-to-use prompt text with dev knowledge
        - Additional knowledge aspects can be added here in the future
    """

    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None

    issue_number = int(instance_id.split("-")[-1])
    log_output.append(f"--get_dev_knowledge instance_id: {instance_id} role: {role}")
    
    history_issue_file = os.path.join(PROJECT_ROOT_DIR, "dev_knowl", "data", "20250707-SwebenchCustom-WithHistoricIssue-Reranking-Summary-Filtered.jsonl")
    log_output.append(f"----loading history file: {os.path.basename(history_issue_file)}")
    assert os.path.exists(history_issue_file), f"History issue file does not exist: {history_issue_file}"
    df = pd.read_json(history_issue_file, lines=True)
    
    retrieved_issues = df[df["instance_id"] == instance_id]
    
    if len(retrieved_issues) == 0:
        log_output.append(f"----no retrieved issues found")
        logger.info("\n".join(log_output))
        return {"dev_knowledge": "", "guiding_questions": ""}

    assert len(retrieved_issues) > 0, f"No retrieved issues found for instance_id: {instance_id}"
    most_similar_issue = retrieved_issues.iloc[0]

    project_name = most_similar_issue["repo"].replace("/", "+")
    issue_id = most_similar_issue["retrieved_issue_number"]
    retrieved_issue_description = most_similar_issue["retrieved_issue_desc"]
    patch = most_similar_issue["retrieved_patch"]
    patch_commit = most_similar_issue["retrieved_commit_id"]

    log_output.append(f"----most similar issue: {project_name} #{issue_id}")
    
    knowledge = contruct_history_knowledge(
        project_name=project_name,
        issue_id=issue_id,
        issue_description=retrieved_issue_description,
        patch=patch,
        patch_commit=patch_commit,
        instance_id=instance_id,
    )
    
    # Get role-specific tags
    role_tags = _get_role_specific_tags(role)
    log_output.append(f"----role tags: {role_tags}")

    # word count of the knowledge
    word_count = len(knowledge.split())
    log_output.append(f"----knowledge word count: {word_count}")

    if knowledge is None:
        raise ValueError(f"No knowledge found for instance_id: {instance_id}")
    
    retrieved_knowledge = {}
    for tag in role_tags:
        tag_knowledge = extract_xml_content(knowledge, tag)
        if tag_knowledge is not None:
            retrieved_knowledge[tag] = tag_knowledge

    log_output.append(f"----extracted knowledge tags: {list(retrieved_knowledge.keys())}")

    # Generate guiding questions for decoder role
    guiding_questions = ""
    if "decoder" in role.lower():
        guiding_questions = _generate_guiding_questions(issue_description, knowledge)
        log_output.append(f"----generated guiding questions: {len(guiding_questions)} chars")
    else:
        log_output.append(f"----skipped guiding questions for role: {role}")

    # Format knowledge as XML
    retrieved_knowledge_formatted = _format_knowledge_as_xml(retrieved_knowledge)
    
    logger.info("\n".join(log_output))

    return {
        "dev_knowledge": construct_dev_knowledge_prompt_version_1(retrieved_knowledge_formatted),
        "guiding_questions": guiding_questions
    }

@tool
def get_dev_knowledge_design_version_3(
    issue_description: Annotated[str, "The description of the current issue to be solved"],
    instance_id: Annotated[str, "The instance ID of the current issue"],
    role: Annotated[str, "The role of the agent: 'decoder', 'mapper', or 'solver'"],
    guiding_questions: Annotated[bool, "Whether to generate guiding questions"],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Get development knowledge from similar historical issues to help solve the current issue.
    
    This tool retrieves and processes knowledge from similar historical issues that have been 
    previously solved. It returns a dictionary with the current dev knowledge prompt, with room
    for additional knowledge aspects in the future.
    
    Args:
        issue_description: Description of the current issue
        instance_id: Instance ID in format 'repo__repo-number' (e.g., 'scikit-learn__scikit-learn-10908')
        role: The role requesting knowledge ('decoder', 'mapper', or 'solver')
        config: Runtime configuration
        
    Returns:
        Dictionary containing:
        - dev_knowledge_prompt: Ready-to-use prompt text with dev knowledge
        - Additional knowledge aspects can be added here in the future
    """

    log_output = []
    if config:
        agent_name = config.get("configurable", {}).get("agent_name")
        log_output.append(f"{agent_name}:")
    else:
        agent_name = None

    issue_number = int(instance_id.split("-")[-1])
    log_output.append(f"--get_dev_knowledge instance_id: {instance_id} role: {role}")
    
    history_issue_file = os.path.join(PROJECT_ROOT_DIR, "dev_knowl", "data", "20250707-SwebenchCustom-WithHistoricIssue-Reranking-Summary-Filtered.jsonl")
    log_output.append(f"----loading history file: {os.path.basename(history_issue_file)}")
    assert os.path.exists(history_issue_file), f"History issue file does not exist: {history_issue_file}"
    df = pd.read_json(history_issue_file, lines=True)
    
    retrieved_issues = df[df["instance_id"] == instance_id]
    
    if len(retrieved_issues) == 0:
        log_output.append(f"----no retrieved issues found")
        logger.info("\n".join(log_output))
        return {"dev_knowledge": "", "guiding_questions": ""}

    assert len(retrieved_issues) > 0, f"No retrieved issues found for instance_id: {instance_id}"

    retrieved_issues = retrieved_issues.sort_values(by="reranked_position")
    most_similar_issue = retrieved_issues.iloc[0]

    project_name = most_similar_issue["repo"].replace("/", "+")
    issue_id = most_similar_issue["retrieved_issue_number"]
    retrieved_issue_description = most_similar_issue["retrieved_issue_desc"]
    patch = most_similar_issue["retrieved_patch"]
    patch_commit = most_similar_issue["retrieved_commit_id"]
    relevance_score = most_similar_issue["relevance_score"]

    log_output.append(f"----most similar issue: {project_name} #{issue_id} (relevance_score: {relevance_score})")
    
    knowledge = contruct_history_knowledge(
        project_name=project_name,
        issue_id=issue_id,
        issue_description=retrieved_issue_description,
        patch=patch,
        patch_commit=patch_commit,
        instance_id=instance_id,
    )
    
    # Get role-specific tags
    role_tags = _get_role_specific_tags(role)
    log_output.append(f"----role tags: {role_tags}")

    # word count of the knowledge
    word_count = len(knowledge.split())
    log_output.append(f"----knowledge word count: {word_count}")

    if knowledge is None:
        raise ValueError(f"No knowledge found for instance_id: {instance_id}")
    
    retrieved_knowledge = {}
    for tag in role_tags:
        tag_knowledge = extract_xml_content(knowledge, tag)
        if tag_knowledge is not None:
            retrieved_knowledge[tag] = tag_knowledge

    log_output.append(f"----extracted knowledge tags: {list(retrieved_knowledge.keys())}")

    # Generate guiding questions for decoder role
    guiding_questions_result = ""
    if guiding_questions and "decoder" in role.lower():
        guiding_questions_result = _generate_guiding_questions(issue_description, knowledge)
        log_output.append(f"----generated guiding questions: {len(guiding_questions_result)} chars")
    else:
        log_output.append(f"----skipped guiding questions for role: {role}")

    # Format knowledge as XML
    retrieved_knowledge_formatted = _format_knowledge_as_xml(retrieved_knowledge)
    
    logger.info("\n".join(log_output))

    return {
        "dev_knowledge": construct_dev_knowledge_prompt_version_2(retrieved_knowledge_formatted, role),
        "guiding_questions": guiding_questions_result
    }

def _generate_guiding_questions(issue_description: str, knowledge: str) -> str:
    """Generate guiding questions for the decoder role."""
    general_root_cause_analysis_steps = extract_xml_content(knowledge, "general_root_cause_analysis_steps")
    
    if not general_root_cause_analysis_steps:
        logger.info("No general_root_cause_analysis_steps found in knowledge, skipping guiding questions")
        return ""
    
    try:
        logger.info("Generating guiding questions with LLM")
        llm = create_llm()
        guiding_questions_prompt = GUIDING_QUESTIONS_PROMPT.format(
            issue_description=issue_description, 
            high_level_extraction=general_root_cause_analysis_steps
        )
        messages = [HumanMessage(content=guiding_questions_prompt)]
        guided_questions_messages = llm.invoke(messages)
        
        # Extract content properly
        guided_questions = guided_questions_messages.content
        if isinstance(guided_questions, list):
            guided_questions = " ".join(str(item) for item in guided_questions)
        elif not isinstance(guided_questions, str):
            guided_questions = str(guided_questions)
        
        guide_prefix = "Make sure you tried to answer the following questions:\n"
        result = guide_prefix + guided_questions
        logger.info(f"Successfully generated guiding questions ({len(result)} chars)")
        return result
        
    except Exception as e:
        logger.error(f"Error generating guiding questions: {e}")
        return ""

@tool
def get_dev_knowledge_design_version_4(
    instance_id: Annotated[str, "The instance ID of the current issue"],
    top_n: Annotated[int, "The number of top similar issues to retrieve"],
    config: Optional[RunnableConfig] = None
) -> list[DevKnowledge]:
    """
    Get dev knowledge from similar historical issues to help solve the current issue.
    """
    logger.info(f"--get_dev_knowledge instance_id: {instance_id}")
    
    history_issue_file = os.path.join(PROJECT_ROOT_DIR, "dev_knowl", "data", "20250707-SwebenchCustom-WithHistoricIssue-Reranking-Summary-Filtered.jsonl")
    logger.info(f"----loading history file: {os.path.basename(history_issue_file)}")
    assert os.path.exists(history_issue_file), f"History issue file does not exist: {history_issue_file}"
    df = pd.read_json(history_issue_file, lines=True)
    
    retrieved_issues = df[df["instance_id"] == instance_id]
    
    if len(retrieved_issues) == 0:
        logger.info("----no retrieved issues found")
        return []

    retrieved_issues = retrieved_issues.sort_values(by="reranked_position")

    actual_count = min(top_n, len(retrieved_issues))
    logger.info(f"----retrieving {actual_count} out of {top_n} requested issues ({len(retrieved_issues)} available)")

    dev_knowledges = []
    for i in range(actual_count):
        most_similar_issue = retrieved_issues.iloc[i]
        project_name = most_similar_issue["repo"].replace("/", "+")
        issue_id = most_similar_issue["retrieved_issue_number"]
        retrieved_issue_description = most_similar_issue["retrieved_issue_desc"]
        patch = most_similar_issue["retrieved_patch"]
        patch_commit = most_similar_issue["retrieved_commit_id"]
        relevance_score = most_similar_issue["relevance_score"]

        logger.info(f"Constructing dev knowledge from no.{i+1} similar issue: {project_name} #{issue_id} (relevance_score: {relevance_score})")

        history_knowledge = contruct_history_knowledge(
            project_name=project_name,
            issue_id=issue_id,
            issue_description=retrieved_issue_description,
            patch=patch,
            patch_commit=patch_commit,
            instance_id=instance_id,
        )
        dev_knowledge = DevKnowledge(
            repo_name=project_name,
            issue_id=str(issue_id),
            issue_description=retrieved_issue_description,
            patch=patch,
            patch_commit=patch_commit,
            relevance_score=relevance_score,
            dev_knowledge=history_knowledge,
        )
        dev_knowledges.append(dev_knowledge)

    logger.info(f"----constructed {len(dev_knowledges)} dev knowledge")
    return dev_knowledges
    
def get_agent_dev_knowledge(role: str, knowledge: str) -> str:
    """
    Get dev knowledge for the agent.
    """
    role_tags = _get_role_specific_tags(role)
    
    retrieved_knowledge = {}
    for tag in role_tags:
        tag_knowledge = extract_xml_content(knowledge, tag)
        if tag_knowledge is not None:
            retrieved_knowledge[tag] = tag_knowledge
    
    retrieved_knowledge_formatted = _format_knowledge_as_xml(retrieved_knowledge)
    return construct_dev_knowledge_prompt_version_2(retrieved_knowledge_formatted, role)

if __name__ == "__main__":
    # Test the new tool functionality
    print("Testing get_dev_knowledge_design_version_2 tool...")

    os.environ["LANGSMITH_TRACING"] = "true"

    jsonl_file = "/home/xuyang/codexray/dev_knowl/data/20250707-SwebenchCustom-WithHistoricIssue-Reranking-Summary.jsonl"
    df = pd.read_json(jsonl_file, lines=True)
    print(df.columns)

    EVAL_INSTANCES = [
        "astropy__astropy-14508",
        "django__django-11333",
        "django__django-14434",
        "pydata__xarray-3095",
        "sphinx-doc__sphinx-8035",
        "django__django-16642",
        "scikit-learn__scikit-learn-10908",
        "matplotlib__matplotlib-14623",
        "psf__requests-2317",
        "psf__requests-1724",
        "pylint-dev__pylint-7277",
        "django__django-13568",
        "pytest-dev__pytest-10051",
        "pylint-dev__pylint-6528",
        "django__django-17084",
    ]

    for instance_id in EVAL_INSTANCES:
        issue_number = int(instance_id.split("-")[-1])
        
        print(f"Instance id: {instance_id}")
        retrieved_issues = df[df["instance_id"] == instance_id]
        
        print(retrieved_issues['retrieved_issue_number'].tolist())
        print(f"Number of retrieved issues: {len(retrieved_issues)}")

        # Filter retrieved issues
        if "django" not in instance_id:
            retrieved_issues = retrieved_issues[retrieved_issues["retrieved_issue_number"] < issue_number]
        
        # filter out relevance score > 0.99
        retrieved_issues = retrieved_issues[retrieved_issues["relevance_score"] <= 0.99]

        print(f"Number of retrieved issues after filtering: {len(retrieved_issues)}")

        retrieved_issues = retrieved_issues.sort_values(by="reranked_position")

        if len(retrieved_issues) == 0:
            print(f"No retrieved issues found for instance_id: {instance_id}")
            continue

        most_similar_issue = retrieved_issues.iloc[0]

        # print retrieved issues id
        print(f"Retrieved issues id: {retrieved_issues['retrieved_issue_number'].tolist()}")
        # print relevancy score
        print(f"Relevancy score: {retrieved_issues['relevance_score'].tolist()}")
        
        # Read cached summary for the current instance's problem statement
        current_problem_statement = retrieved_issues.iloc[0]['problem_statement']
        content_hash = generate_content_hash(current_problem_statement)
        current_instance_summary = load_summary_from_cache("query", content_hash)
        
        print(f"\nCurrent instance ({instance_id}) problem statement summary:")
        if current_instance_summary:
            print("-------summary of current instance-------")
            print(f"Summary: {current_instance_summary}")
        else:
            print("No cached summary found for current instance")
        
        # Read cached summary for the most similar historical issue
        repo = most_similar_issue['repo'].split("/")[-1]
        issue_number_str = str(most_similar_issue['retrieved_issue_number'])
        cached_summary = load_summary_from_cache(repo, issue_number_str)
        
        print(f"\nMost similar historical issue ({repo}-{issue_number_str}) summary:")
        if cached_summary:
            print(f"--------summary of most similar historical issue {round(most_similar_issue['relevance_score'], 2)}--------")
            print(f"Summary: {cached_summary}")
        else:
            print(f"No cached summary found for historical issue")

        project_name = most_similar_issue["repo"].replace("/", "+")
        issue_id = most_similar_issue["retrieved_issue_number"]
        issue_description = most_similar_issue["retrieved_issue_desc"]
        patch = most_similar_issue["retrieved_patch"]
        patch_commit = most_similar_issue["retrieved_commit_id"]

        history_knowledge = contruct_history_knowledge(
            project_name=project_name,
            issue_id=issue_id,
            issue_description=issue_description,
            patch=patch,
            patch_commit=patch_commit,
            instance_id=instance_id,
        )
        print(f"History knowledge: {history_knowledge}")