from pydantic import BaseModel
from typing import Dict, List
from app.config.env_config import config


class ChatMessage(BaseModel):
    """Model for a chat message."""
    role: str
    content: str


class HistoryModule:
    """Module for managing conversation history."""
    
    def __init__(self, token_limit: int = None):
        # Use the ChatMemoryBuffer from llama_index
        from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
        
        # Use provided token limit or get from config
        self.token_limit = token_limit or config.history_token_limit
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=self.token_limit)
        
        # Initialize an internal list to store conversation history
        self.history: List[ChatMessage] = []

    def add_user_message(self, content: str):
        """Add a user message to the history."""
        msg = ChatMessage(role="user", content=content)
        self.history.append(msg)
        # Also update the underlying memory store
        self.memory.chat_store.set_messages(
            self.memory.chat_store_key, 
            [m.dict() for m in self.history]
        )

    def add_agent_message(self, content: str):
        """Add an agent message to the history."""
        msg = ChatMessage(role="assistant", content=content)
        self.history.append(msg)
        self.memory.chat_store.set_messages(
            self.memory.chat_store_key, 
            [m.dict() for m in self.history]
        )

    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries.
        """
        # Return the history as a list of dicts so that it passes validation
        return [msg.dict() for msg in self.history] 