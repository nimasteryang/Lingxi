# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lingxi is a multi-agent software engineering framework that automates issue resolution using LangGraph workflows. It achieved 74.6% success on SWE-Bench Verified. The system uses specialized agents (Problem Decoder, Solution Mapper, Problem Solver) that coordinate through a supervisor to analyze GitHub issues and generate code fixes.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -e .

# Or using uv (recommended)
uv run --env-file .env [command]
```

### Required Environment Variables
Create a `.env` file in the project root with:
```bash
LLM_PROVIDER=anthropic  # or "openai", "deepseek"
LLM_MODEL=claude-3-5-haiku-latest
ANTHROPIC_API_KEY=your_key_here  # or OPENAI_API_KEY, DEEPSEEK_API_KEY
OPENAI_API_KEY=your_key_here     # Required for embeddings
GITHUB_TOKEN=your_token_here     # For GitHub API access
```

### Running the Application
```bash
# Start LangGraph Studio
langgraph dev --no-reload

# Or with uv
uv run --env-file .env langgraph dev --no-reload
```

### Code Quality
```bash
# Linting with ruff
ruff check src/
ruff format src/

# Type checking
mypy src/
```

### Testing
```bash
pytest
```

## Architecture Overview

### Core Components

**LangGraph Workflows**: Two main graph implementations in `src/agent/`:
- `supervisor_graph_demo.py`: Primary issue resolution workflow with supervisor coordination
- `hierarchy_graph_demo.py`: Extended workflow with multi-agent manager and reviewer

**Agent System**: Five specialized agents with distinct roles:
- **Supervisor**: Routes workflow between agents based on progress
- **Problem Decoder**: Analyzes issues and performs bug localization using `[view_directory, search_relevant_files, view_file_content]`
- **Solution Mapper**: Creates detailed code change plans using `[view_directory, search_relevant_files, view_file_content]`
- **Problem Solver**: Implements code changes using `[view_directory, search_relevant_files, str_replace_editor]`
- **Multi-Agent Manager**: Coordinates between issue resolver and reviewer (hierarchy graph only)
- **Reviewer**: Validates fixes and runs tests using `[view_directory, search_relevant_files, view_file_content, run_shell_cmd]`

**State Management**: Custom LangGraph state in `src/agent/state.py`:
- `CustomState`: Basic workflow state with agent routing
- `State`: Enhanced state for SWE-Bench workflow with iteration tracking and caching support

### Key Tools (`src/agent/tool_set/`)

**Core Tools**:
- `view_directory`: File system exploration with depth limiting
- `view_file_content`: File reading with optional line range selection  
- `search_relevant_files`: Vector similarity search using ChromaDB and OpenAI embeddings
- `str_replace_editor`: OHEditor-based file editing (adapted from OpenHands)
- `run_shell_cmd`: Sequential shell command execution

**Context Management** (`context_tools.py`):
- Creates project knowledge vector database using tree-sitter parsing
- Differentiates between functions/methods and general code
- Builds ChromaDB at `RUNTIME_DIR` location

### Configuration and Runtime

**Runtime Setup** (`runtime_config.py`):
- `RuntimeConfig`: Singleton for environment configuration
- `load_from_github_issue_url()`: Automatically clones repositories and sets up workspace

**Constants** (`constant.py`):
- `RUNTIME_DIR`: Local storage for cloned repos and vector databases
- `PATCH_RESULT_DIR`: Output location for generated patches
- Language parsers for Python and Java file indexing

## Development Workflow

### Adding New Agents
1. Create system prompt in `src/agent/prompt/`
2. Define tools list for the agent
3. Use `create_react_agent(llm, tools=tools, prompt=system_prompt)`
4. Create node function to invoke agent and process results
5. Add node to StateGraph and define edges

### Creating New Graphs
1. Define StateGraph with appropriate State class
2. Add nodes using `workflow.add_node(name, function)`
3. Define routing with `workflow.add_edge()` or conditional edges
4. Compile graph with `workflow.compile()`
5. Register in `langgraph.json` as `"graph_name": "file.py:variable_name"`

### Tool Development
Use `@tool` decorator with comprehensive docstrings:
```python
@tool
def my_tool(param: str) -> str:
    """Tool description with parameter and return info."""
    return result
```

## Important Implementation Details

**Vector Database Management**: Each GitHub issue creates a separate ChromaDB instance. Clean `RUNTIME_DIR` periodically to manage storage.

**Repository Cloning**: Issues are processed by cloning repos at specific commits (latest or pre-fix commit). This creates multiple local copies requiring cleanup.

**Message State**: ReACT agents require structured `MessagesState` with message list. Tool usage creates `AIMessage` followed by `ToolMessage` - both must be preserved.

**Human-in-the-Loop**: Available in both graphs but disabled by default. Enable in LangGraph Studio UI before submission.

**Caching**: Custom message reducer in `State` enables incremental caching by marking the last content block with `cache_control`.

## File Organization

```
src/agent/
├── supervisor_graph_demo.py    # Main workflow graph
├── hierarchy_graph_demo.py     # Extended workflow with reviewer
├── state.py                    # LangGraph state definitions
├── llm.py                     # LLM provider configuration
├── runtime_config.py          # Environment setup
├── prompt/                    # Agent system prompts
├── tool_set/                  # Agent tools and utilities
│   ├── sepl_tools.py         # Core development tools
│   ├── context_tools.py      # Vector search and knowledge management
│   ├── edit_tool.py          # File editing capabilities
│   └── linter/               # Code quality tools
└── utils.py                   # Helper functions
```