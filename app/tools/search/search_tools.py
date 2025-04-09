from typing import Dict, Any, Optional, List
from pydantic import Field

from app.tools.base_tool import BaseTool
from app.services.embeddings import EmbeddingService
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore


class SearchMeetingsTool(BaseTool):
    """Tool for searching meetings in the database."""
    
    def __init__(self, vector_store: SupabaseVectorStore, embedding_service: EmbeddingService):
        super().__init__(
            name="SearchMeetings",
            description="Search for relevant meetings. Returns documents formatted from a list of dictionaries."
        )
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def __call__(self, user_input: str = Field(
        description="Input from the user, please include the organization name in the user_input as well. "
                    "If user mentions date range, please include the date range in the user_input as well."
    )) -> str:
        """
        Search for meetings matching the user input.
        
        Args:
            user_input: The user's query string.
            
        Returns:
            Formatted search results as a string.
        """
        query_embedding = self.embedding_service.get_embedding(user_input)
        user = self.vector_store.get_user()
        user_id = user.id
        
        # Call the Supabase RPC method for hybrid search
        results = self.vector_store.search_meetings(user_input, query_embedding, user_id)
        
        # Process the returned list of documents
        if not results:
            return "No documents found."
        
        formatted_results = []
        for idx, doc in enumerate(results):
            doc_lines = [f"Document {idx+1}:"]
            for key, value in doc.items():
                doc_lines.append(f"  {key}: {value}")
            formatted_results.append("\n".join(doc_lines))
        
        return "\n\n".join(formatted_results)


class SearchMeetingsByOrganizationTool(BaseTool):
    """Tool for searching meetings filtered by organization."""
    
    def __init__(self, vector_store: SupabaseVectorStore, embedding_service: EmbeddingService):
        super().__init__(
            name="SearchMeetingsByOrganization",
            description="Search for relevant meetings by organization. Returns documents formatted from a list of dictionaries."
        )
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def __call__(self, 
                user_input: str = Field(
                    description="Input from the user, please include the organization name in the user_input as well. "
                                "If user mentions date range, please include the date range in the user_input as well."
                ),
                organization_input: str = Field(
                    description="The exact organization name to filter meetings by (should be validated first with ValidateOrganization)"
                )) -> str:
        """
        Search for meetings by organization.
        
        Args:
            user_input: The user's query string.
            organization_input: The organization name to filter by.
            
        Returns:
            Formatted search results as a string.
        """
        query_embedding = self.embedding_service.get_embedding(user_input)
        user = self.vector_store.get_user()
        user_id = user.id
        
        # Call the Supabase RPC method for organization-specific search
        results = self.vector_store.search_meetings_by_organization(
            user_input, query_embedding, user_id, organization_input
        )
        
        # Process the returned list of documents
        if not results:
            return "No documents found."
        
        formatted_results = []
        for idx, doc in enumerate(results):
            doc_lines = [f"Document {idx+1}:"]
            for key, value in doc.items():
                doc_lines.append(f"  {key}: {value}")
            formatted_results.append("\n".join(doc_lines))
        
        return "\n\n".join(formatted_results)


class SearchTravelPackagesTool(BaseTool):
    """Tool for searching travel packages in the database."""
    
    def __init__(self, vector_store: SupabaseVectorStore, embedding_service: EmbeddingService):
        super().__init__(
            name="SearchTravelPackages",
            description="Search for relevant travel packages based on multiple criteria. Returns documents formatted from a list of dictionaries."
        )
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def __call__(self, 
                location_input: str = Field(description="Location preferences or destination"),
                duration_input: str = Field(description="Duration preferences"),
                budget_input: str = Field(description="Budget preferences"),
                transportation_input: str = Field(description="Transportation preferences"),
                accommodation_input: str = Field(description="Accommodation preferences"),
                food_input: str = Field(description="Food preferences"),
                activities_input: str = Field(description="Activities preferences"),
                notes_input: str = Field(description="Additional notes or preferences"),
                match_count: int = Field(default=10, description="Number of results to return")
                ) -> List[Dict]:
        """
        Search for travel packages matching the user's preferences.
        
        Args:
            location_input: Location preferences or destination
            duration_input: Duration preferences
            budget_input: Budget preferences
            transportation_input: Transportation preferences
            accommodation_input: Accommodation preferences
            food_input: Food preferences
            activities_input: Activities preferences
            notes_input: Additional notes or preferences
            match_count: Number of results to return
            
        Returns:
            List of travel package dictionaries matching the search criteria
        """
        # Helper function to check if input is valid
        def is_valid_input(input_str: str) -> bool:
            return input_str is not None and len(input_str.strip()) > 1

        # Get embeddings for each input, using "empty string" embedding if input is invalid
        location_embedding = self.embedding_service.get_embedding(location_input) if is_valid_input(location_input) else self.embedding_service.get_embedding("empty string")
        duration_embedding = self.embedding_service.get_embedding(duration_input) if is_valid_input(duration_input) else self.embedding_service.get_embedding("empty string")
        budget_embedding = self.embedding_service.get_embedding(budget_input) if is_valid_input(budget_input) else self.embedding_service.get_embedding("empty string")
        transportation_embedding = self.embedding_service.get_embedding(transportation_input) if is_valid_input(transportation_input) else self.embedding_service.get_embedding("empty string")
        accommodation_embedding = self.embedding_service.get_embedding(accommodation_input) if is_valid_input(accommodation_input) else self.embedding_service.get_embedding("empty string")
        food_embedding = self.embedding_service.get_embedding(food_input) if is_valid_input(food_input) else self.embedding_service.get_embedding("empty string")
        activities_embedding = self.embedding_service.get_embedding(activities_input) if is_valid_input(activities_input) else self.embedding_service.get_embedding("empty string")
        notes_embedding = self.embedding_service.get_embedding(notes_input) if is_valid_input(notes_input) else self.embedding_service.get_embedding("empty string")
        
        # Call the Supabase RPC method for travel package search
        results = self.vector_store.search_travel_packages(
            location_vector=location_embedding,
            duration_vector=duration_embedding,
            budget_vector=budget_embedding,
            transportation_vector=transportation_embedding,
            accommodation_vector=accommodation_embedding,
            food_vector=food_embedding,
            activities_vector=activities_embedding,
            notes_vector=notes_embedding,
            match_count=match_count
        )
        
        # Return the list of dictionaries directly
        if not results:
            return [] # Return empty list if no results
        
        return results # Return the raw list of dictionaries 