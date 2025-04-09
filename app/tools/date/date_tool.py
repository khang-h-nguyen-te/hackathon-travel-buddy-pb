from datetime import datetime
from typing import Dict, Any

from app.tools.base_tool import BaseTool


class DateExtractionTool(BaseTool):
    """Tool for extracting the current date/time."""
    
    def __init__(self):
        super().__init__(
            name="ExtractCurrentDate",
            description="Returns the current date/time, allowing you to interpret requests like 'last week' or 'two months ago.'"
        )
    
    def __call__(self) -> str:
        """
        Returns the current date/time in ISO format.
        This allows the agent to understand time-based queries.
        """
        return datetime.now().isoformat() 