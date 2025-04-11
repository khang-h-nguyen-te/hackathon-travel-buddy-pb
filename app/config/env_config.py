import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EnvConfig:
    """Configuration manager for environment variables."""
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get an environment variable.
        
        Args:
            key: The name of the environment variable.
            default: The default value to return if the variable is not set.
            
        Returns:
            The value of the environment variable, or the default value.
        """
        return os.environ.get(key, default)
    
    @staticmethod
    def get_required(key: str) -> str:
        """
        Get a required environment variable.
        
        Args:
            key: The name of the environment variable.
            
        Returns:
            The value of the environment variable.
            
        Raises:
            ValueError: If the environment variable is not set.
        """
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    @staticmethod
    def is_debug() -> bool:
        """Check if debug mode is enabled."""
        # Ensure default is '0' if DEBUG is not set
        return bool(int(os.environ.get("DEBUG", "0")))

    @staticmethod
    def get_int(key: str, default: int) -> int:
        """Get an environment variable as an integer."""
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

# Application-specific config properties
class AppConfig:
    """Application-specific configuration."""
    
    @property
    def openai_api_key(self) -> str:
        """Get the OpenAI API key."""
        return EnvConfig.get_required("OPENAI_API_KEY")
    
    @property
    def supabase_url(self) -> str:
        """Get the Supabase URL."""
        # Assuming the key in .env might be VITE_PUBLIC_BASE_URL based on earlier code
        # If it's SUPABASE_URL, change "VITE_PUBLIC_BASE_URL" to "SUPABASE_URL"
        return EnvConfig.get_required("VITE_PUBLIC_BASE_URL") 
    
    @property
    def supabase_anon_key(self) -> str:
        """Get the Supabase anonymous key."""
        # Assuming the key in .env might be VITE_VITE_APP_SUPABASE_ANON_KEY
        # If it's SUPABASE_ANON_KEY, change this line
        return EnvConfig.get_required("VITE_VITE_APP_SUPABASE_ANON_KEY")
    
    @property
    def supabase_service_key(self) -> str:
        """Get the Supabase service key."""
        return EnvConfig.get_required("SUPABASE_SERVICE_KEY")

    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled."""
        return EnvConfig.is_debug()
    
    @property
    def llm_model(self) -> str:
        """Get the LLM model name (default: gpt-4o)."""
        return EnvConfig.get("LLM_MODEL", "gpt-4o")

    @property
    def jwt_private_key(self) -> str:
        """Get the JWT private key for password encryption."""
        # Using JWT_SECRET based on previous logic
        return EnvConfig.get_required("JWT_SECRET") 

    @property
    def openrouter_api_key(self) -> str:
        """Get the OpenRouter API key."""
        return EnvConfig.get("OPENROUTER_API_KEY", "") # Use get with default empty string

    @property
    def history_token_limit(self) -> int:
        """Get the token limit for chat history."""
        return EnvConfig.get_int("HISTORY_TOKEN_LIMIT", 1000)


# Create a singleton instance of the config
config = AppConfig() 