from openai import OpenAI
from app.config.env_config import config


class EmbeddingService:
    """Service for generating embeddings from text using OpenAI API."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or config.openai_api_key
        self.client = OpenAI(api_key=self.api_key)
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """
        Generate an embedding for the provided text.
        
        Args:
            text (str): The text to generate an embedding for.
            model (str): The embedding model to use.
            
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding 