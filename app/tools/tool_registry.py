from typing import Dict, List, Optional, Set
from app.tools.base_tool import BaseTool


class ToolRegistry:
    """
    Registry for managing tools.
    Allows tools to be registered, retrieved, and categorized.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, Set[str]] = {}
    
    def register(self, tool: BaseTool, categories: Optional[List[str]] = None) -> 'ToolRegistry':
        """
        Register a tool with the registry.
        Optionally assign it to categories.
        """
        self._tools[tool.name] = tool
        
        if categories:
            for category in categories:
                if category not in self._categories:
                    self._categories[category] = set()
                self._categories[category].add(tool.name)
        
        return self
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a specific category."""
        if category not in self._categories:
            return []
        
        return [self._tools[name] for name in self._categories[category] 
                if name in self._tools]
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self._categories.keys())
    
    def to_function_tools_list(self) -> List[Dict]:
        """Convert all tools to FunctionTool compatible format."""
        return [tool.to_function_tool() for tool in self._tools.values()] 