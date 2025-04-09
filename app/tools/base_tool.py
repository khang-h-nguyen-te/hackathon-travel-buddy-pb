from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type
from pydantic import BaseModel, Field


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    Tools are callable objects that execute a specific function.
    """
    name: str
    description: str
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool functionality"""
        pass
    
    def to_function_tool(self) -> Dict[str, Any]:
        """
        Convert this tool to a format compatible with llama_index FunctionTool
        or other similar tool interfaces
        """
        return {
            "name": self.name,
            "description": self.description,
            "fn": self.__call__
        } 