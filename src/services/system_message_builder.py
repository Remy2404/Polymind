"""
Shared utility for building system messages with consistent tool instructions.
This module provides reusable components for creating system messages across different AI providers.
"""
from typing import Optional, List, Dict, Any
from src.services.model_handlers.model_configs import ModelConfigurations


class SystemMessageBuilder:
    """Builder for creating consistent system messages across AI providers."""
    
    DEFAULT_BASE_MESSAGE = "You are an advanced AI assistant that helps users with various tasks."
    
    @staticmethod
    def get_base_message(model_id: str) -> str:
        """Get base message from model configuration or use default.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Base system message string
        """
        model_config = ModelConfigurations.get_all_models().get(model_id)
        if model_config and model_config.system_message:
            return model_config.system_message
        return SystemMessageBuilder.DEFAULT_BASE_MESSAGE
    
    @staticmethod
    def get_context_hint(context: Optional[List[Dict]] = None) -> str:
        """Get context hint text if context is provided.
        
        Args:
            context: Optional conversation context
            
        Returns:
            Context hint string or empty string
        """
        if context:
            return " Use conversation history/context when relevant."
        return ""
    
    @staticmethod
    def build_basic_message(
        model_id: str,
        context: Optional[List[Dict]] = None,
        add_concise_hint: bool = False
    ) -> str:
        """Build a basic system message without tool instructions.
        
        Args:
            model_id: The model identifier
            context: Optional conversation context
            add_concise_hint: Whether to add "Be concise, helpful, and accurate" hint
            
        Returns:
            System message string
        """
        base_message = SystemMessageBuilder.get_base_message(model_id)
        context_hint = SystemMessageBuilder.get_context_hint(context)
        
        if add_concise_hint and not context:
            model_config = ModelConfigurations.get_all_models().get(model_id)
            if not model_config:
                return base_message + " Be concise, helpful, and accurate."
        
        return base_message + context_hint
    
    @staticmethod
    def categorize_tools_generic(
        tools: List[Dict[str, Any]],
        name_extractor: callable = lambda t: t["function"]["name"],
        description_extractor: callable = lambda t: t["function"].get("description", "")
    ) -> Dict[str, List[str]]:
        """Categorize tools by their functionality.
        
        This is a generic tool categorization that works with different tool formats.
        Subclasses can provide custom extractors for different tool structures.
        
        Args:
            tools: List of tool definitions
            name_extractor: Function to extract tool name from tool object
            description_extractor: Function to extract description from tool object
            
        Returns:
            Dictionary mapping categories to tool names
        """
        categories = {
            "Content Fetching": [],
            "Documentation": [],
            "Search & Research": [],
            "Development": [],
            "Analysis": [],
            "Communication": [],
            "Document Processing": [],
            "Other": [],
        }
        
        category_keywords = {
            "Content Fetching": ["fetch", "html", "markdown", "txt", "json", "url", "webpage", "content", "crawl"],
            "Documentation": ["doc", "docs", "documentation", "library", "api", "guide", "tutorial", "reference"],
            "Search & Research": ["search", "find", "query", "lookup", "research", "web", "browse"],
            "Development": ["code", "dev", "build", "compile", "test", "debug", "git"],
            "Analysis": ["analyze", "process", "calculate", "compute", "evaluate"],
            "Communication": ["send", "notify", "message", "email", "chat"],
            "Document Processing": ["pdf", "docx", "export", "generate", "format", "convert"],
        }
        
        for tool in tools:
            try:
                tool_name_full = name_extractor(tool)
                tool_name = tool_name_full.lower()
                description = description_extractor(tool).lower()
                
                categorized = False
                for category, keywords in category_keywords.items():
                    if any(keyword in tool_name or keyword in description for keyword in keywords):
                        categories[category].append(tool_name_full)
                        categorized = True
                        break
                
                if not categorized:
                    categories["Other"].append(tool_name_full)
            except (KeyError, AttributeError, TypeError):
                # Skip tools that don't match expected format
                continue
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    @staticmethod
    def build_tool_instructions(
        tool_names: List[str],
        tool_categories: Dict[str, List[str]],
        provider_specific_instructions: str = ""
    ) -> str:
        """Build tool usage instructions.
        
        Args:
            tool_names: List of available tool names
            tool_categories: Dictionary of categorized tools
            provider_specific_instructions: Any provider-specific instructions to prepend
            
        Returns:
            Tool instructions string
        """
        base_instructions = f"""
You have access to the following tools: {', '.join(tool_names)}
{provider_specific_instructions}
1. **Identify the Right Tool**: Choose the most appropriate tool based on the user's request
2. **Provide Complete Arguments**: Ensure all required parameters are included in your tool calls
3. **Handle Results**: Use the tool results to provide comprehensive, accurate responses
4. **Combine Tools**: Use multiple tools in parallel when possible to provide comprehensive answers. Call all relevant tools in one response to gather complete information.

**Available Tool Categories:**
{chr(10).join([f"- **{category}**: {', '.join(category_tools)}" for category, category_tools in tool_categories.items()])}

**Best Practices:**
- Always use tools when they can provide more accurate or current information
- When multiple tools are relevant, use them together in one response for thorough analysis
- Provide detailed, helpful responses based on tool results
- If a tool fails, try alternative approaches or inform the user
- Do not mention tool internal details or <think> tags in your final response

Focus on providing the most helpful and accurate response possible using the available tools."""
        return base_instructions
