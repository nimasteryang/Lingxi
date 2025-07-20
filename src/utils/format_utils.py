# Utility functions
import ast
from datetime import datetime
import json
import re
from typing import Optional
from typing import Any, Dict, List
from typing import Any, Dict, List, Optional

import yaml

def format_section_header(title: str, content: str, index: Optional[int] = None) -> str:
    """Format a section with consistent header styling."""
    section_title = f"{title} {index}" if index is not None else title
    header = f"================================ {section_title} ================================="

    if index is not None:
        return (
            f"\n\n{header}\n\n"
            f"<{section_title}>\n"
            f"{content}\n"
            f"</{section_title}>\n"
        )
    else:
        return f"\n\n{header}\n\n{content}"


class LLMInputFormatter:
    """
    Formats structured analysis data for optimal LLM consumption.
    Supports multiple output formats with Markdown as the recommended default.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def to_markdown(self, include_metadata: bool = False) -> str:
        """
        Convert data to Markdown format - recommended for LLM input.
        
        Args:
            include_metadata: Whether to include metadata section
            
        Returns:
            Formatted markdown string
        """
        md_lines = []
        
        # Add metadata section if requested
        if include_metadata:
            md_lines.append("# Analysis Session")
            md_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_lines.append(f"**Total Steps**: {len(self.data)}")
            md_lines.append("\n---\n")
        
        # Process each step
        for item in self.data:
            step_num = item.get('step', 'N/A')
            md_lines.append(f"## Step {step_num}")
            
            # Add thinking section if present
            if 'thinking' in item:
                md_lines.append("\n### 🤔 Thinking")
                md_lines.append(self._format_text(item['thinking']))
            
            # Add observation section if present
            if 'observation' in item:
                md_lines.append("\n### 👁️ Observation")
                md_lines.append(self._format_text(item['observation']))
            
            # Add action section if present
            if 'action' in item:
                md_lines.append("\n### 🎯 Action")
                md_lines.append(self._format_text(item['action']))
            
            # Add conclusion section if present
            if 'conclusion' in item:
                md_lines.append("\n### 📋 Conclusion")
                md_lines.append(self._format_text(item['conclusion']))
            md_lines.append("\n")
        return '\n'.join(md_lines)
    
    def to_structured_markdown(self) -> str:
        """
        Convert to a more structured Markdown format with clear sections.
        Best for complex analysis with multiple types of content.
        """
        md_lines = ["# Analysis Report\n"]
        
        # Create summary section
        md_lines.append("## Summary\n")
        md_lines.append(f"- **Total Steps**: {len(self.data)}")
        md_lines.append(f"- **Steps with Thinking**: {sum(1 for item in self.data if 'thinking' in item)}")
        md_lines.append(f"- **Steps with Actions**: {sum(1 for item in self.data if 'action' in item)}")
        md_lines.append(f"- **Steps with Conclusions**: {sum(1 for item in self.data if 'conclusion' in item)}\n")
        
        # Main content
        md_lines.append("## Detailed Analysis\n")
        
        for item in self.data:
            step = item.get('step', 'N/A')
            
            # Create collapsible section for each step
            md_lines.append(f"<details>")
            md_lines.append(f"<summary><strong>Step {step}</strong></summary>\n")
            
            if 'thinking' in item:
                md_lines.append("**💭 Thinking Process:**")
                md_lines.append(f"\n{self._format_text(item['thinking'])}\n")
            
            if 'observation' in item:
                md_lines.append("**🔍 Observation:**")
                md_lines.append(f"\n{self._format_text(item['observation'])}\n")
            
            if 'action' in item:
                md_lines.append("**⚡ Action Taken:**")
                md_lines.append(f"\n```python\n{item['action']}\n```\n")
            
            if 'conclusion' in item:
                md_lines.append("**✅ Conclusion:**")
                md_lines.append(f"\n{self._format_text(item['conclusion'])}\n")
            
            md_lines.append("</details>\n")
        
        return '\n'.join(md_lines)
    
    def to_conversation_format(self) -> str:
        """
        Convert to a conversation-style format, ideal for chat-based LLMs.
        """
        lines = []
        
        for item in self.data:
            step = item.get('step', 'N/A')
            
            if 'thinking' in item:
                lines.append(f"**Assistant (Step {step} - Thinking):**")
                lines.append(self._format_text(item['thinking']))
                # lines.append("")
            
            if 'observation' in item:
                lines.append(f"**Assistant (Step {step} - Observation):**")
                lines.append(self._format_text(item['observation']))
                # lines.append("")
            
            if 'action' in item:
                lines.append(f"**System (Step {step} - Action):**")
                lines.append(self._format_text(item['action']))
                # lines.append("")
            
            if 'conclusion' in item:
                lines.append(f"**Assistant (Step {step} - Conclusion):**")
                lines.append(self._format_text(item['conclusion']))
                # lines.append("")
        
        return '\n'.join(lines)
    
    def to_yaml(self) -> str:
        """
        Convert to YAML format - more structured but less readable for long text.
        """
        # Clean up the data for YAML serialization
        cleaned_data = []
        for item in self.data:
            cleaned_item = {}
            for key, value in item.items():
                if isinstance(value, str):
                    # Clean up long strings for better YAML formatting
                    cleaned_item[key] = value.strip()
                else:
                    cleaned_item[key] = value
            cleaned_data.append(cleaned_item)
        
        return yaml.dump(
            cleaned_data,
            default_flow_style=False,
            width=80,
            sort_keys=False
        )
    
    def to_json_formatted(self) -> str:
        """
        Convert to formatted JSON - good for APIs but not ideal for LLM reading.
        """
        return json.dumps(self.data, indent=2, ensure_ascii=False)
    
    def _format_text(self, text) -> str:
        """
        Format text for better readability in Markdown.
        Handles both strings and lists.
        """
        # Handle different input types
        if isinstance(text, list):
            # Join list elements with newlines
            text = '\n'.join(str(item) for item in text)
        elif text is None:
            text = ""
        else:
            # Convert to string if it's not already
            text = str(text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text.strip())
        
        # Ensure proper spacing around code blocks
        text = re.sub(r'```(\w+)?\n', r'\n```\1\n', text)
        text = re.sub(r'\n```', r'\n\n```', text)
        
        # Format numbered lists properly
        text = re.sub(r'(\d+)\.\s+', r'\n\1. ', text)
        
        return text
    
    def extract_key_insights(self) -> str:
        """
        Extract and format only the key insights (conclusions) for a summary.
        """
        insights = []
        
        for item in self.data:
            if 'conclusion' in item:
                step = item.get('step', 'N/A')
                insights.append(f"### Step {step} Insight")
                insights.append(self._format_text(item['conclusion']))
                insights.append("")
        
        if insights:
            return "# Key Insights Summary\n\n" + '\n'.join(insights)
        else:
            return "# No conclusions found in the analysis"


# Example usage function
def format_analysis_for_llm(data: List[Dict[str, Any]], format_type: str = 'markdown') -> str:
    """
    Main function to format analysis data for LLM consumption.
    
    Args:
        data: List of analysis steps
        format_type: One of 'markdown', 'structured_markdown', 'conversation', 'yaml', 'json'
    
    Returns:
        Formatted string ready for LLM input
    """
    formatter = LLMInputFormatter(data)
    
    format_methods = {
        'markdown': formatter.to_markdown,
        'structured_markdown': formatter.to_structured_markdown,
        'conversation': formatter.to_conversation_format,
        'yaml': formatter.to_yaml,
        'json': formatter.to_json_formatted,
        'insights': formatter.extract_key_insights
    }
    
    if format_type not in format_methods:
        raise ValueError(f"Unknown format type: {format_type}. Choose from {list(format_methods.keys())}")
    
    return format_methods[format_type]()


