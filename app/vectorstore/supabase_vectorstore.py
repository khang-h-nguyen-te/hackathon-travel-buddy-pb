import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from app.config.supabase_config import get_supabase_client


class SupabaseVectorStore:
    """Interface to Supabase vector store for meeting data."""
    
    def __init__(self, url: str, key: str, auth: str = None):
        self.url = url
        self.key = key
        self.auth = auth
        self.client = self.create_supabase_client()
        self.logger = logging.getLogger(__name__)

    def create_supabase_client(self):
        """
        Create a Supabase client instance.
        If an auth header is provided, add it to the global headers so that row-level security (RLS)
        policies are applied using the user's context.
        """
        return get_supabase_client(self.auth)

    def get_user(self):
        """
        Retrieve the user from Supabase Auth.
        """
        token = self.auth.replace("Bearer ", "")
        user_response = self.client.auth.get_user(token)        
        return user_response.user


    def search_travel_packages(self, 
                             location_vector: list,
                             duration_vector: list,
                             budget_vector: list,
                             transportation_vector: list,
                             accommodation_vector: list,
                             food_vector: list,
                             activities_vector: list,
                             notes_vector: list,
                             match_count: int = 10):
        """
        Search for travel packages based on multiple vector criteria.
        
        Args:
            location_vector: Vector embedding for location preferences
            duration_vector: Vector embedding for duration preferences
            budget_vector: Vector embedding for budget preferences
            transportation_vector: Vector embedding for transportation preferences
            accommodation_vector: Vector embedding for accommodation preferences
            food_vector: Vector embedding for food preferences
            activities_vector: Vector embedding for activities preferences
            notes_vector: Vector embedding for additional notes/preferences
            match_count: Maximum number of results to return
            
        Returns:
            List of matching travel packages
        """
        response = self.client.rpc("search_travel_packages", {
            "location_vector_input": location_vector,
            "duration_vector_input": duration_vector,
            "budget_vector_input": budget_vector,
            "transportation_vector_input": transportation_vector,
            "accommodation_vector_input": accommodation_vector,
            "food_vector_input": food_vector,
            "activities_vector_input": activities_vector,
            "notes_vector_input": notes_vector,
            "match_count": match_count
        }).execute()
        return response.data 