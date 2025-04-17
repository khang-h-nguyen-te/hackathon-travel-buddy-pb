import json
import logging
from typing import List, Dict, Any, Optional

import openai
from supabase import Client

from app.config.env_config import config
from app.models.suggest_tour_models import TourSuggestionResponse, BaseTour
from app.services.embeddings import EmbeddingService
from app.templates.prompt_templates import SUGGEST_TOUR_TEMPLATE
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore

logger = logging.getLogger(__name__)

class TourSuggestionService:
    """Service for generating combined tour suggestions."""
    
    def __init__(
        self, 
        vector_store: SupabaseVectorStore,
        embedding_service: EmbeddingService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        
    async def search_travel_packages_v2(
        self,
        location_input: str = "",
        budget_input: str = "",
        accommodation_input: str = "",
        activities_input: str = "",
        notes_input: str = "",
        match_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for travel packages using the search_travel_packages_v2 function in Supabase.
        
        Args:
            location_input: Location preferences or destination
            budget_input: Budget preferences
            accommodation_input: Accommodation preferences
            activities_input: Activities preferences
            notes_input: Additional notes
            match_count: Number of results to return
            
        Returns:
            List of travel packages
        """
        try:
            print("[INFO] Generating embeddings for search_travel_packages_v2")
            
            # Generate embeddings for each input (if provided)
            budget_vector = None
            accommodation_vector = None
            activities_vector = None
            notes_vector = None
            
            try:
                # Create an empty vector of the right dimension (1536) to use as default
                empty_vector = [0.0] * 1536
                
                if budget_input:
                    budget_vector = self.embedding_service.get_embedding(budget_input)
                    print(f"[DEBUG] Generated budget vector with length: {len(budget_vector) if budget_vector else 0}")
                else:
                    budget_vector = empty_vector
                
                if accommodation_input:
                    accommodation_vector = self.embedding_service.get_embedding(accommodation_input)
                    print(f"[DEBUG] Generated accommodation vector with length: {len(accommodation_vector) if accommodation_vector else 0}")
                else:
                    accommodation_vector = empty_vector
                
                if activities_input:
                    activities_vector = self.embedding_service.get_embedding(activities_input)
                    print(f"[DEBUG] Generated activities vector with length: {len(activities_vector) if activities_vector else 0}")
                else:
                    activities_vector = empty_vector
                
                if notes_input:
                    notes_vector = self.embedding_service.get_embedding(notes_input)
                    print(f"[DEBUG] Generated notes vector with length: {len(notes_vector) if notes_vector else 0}")
                else:
                    notes_vector = empty_vector
            except Exception as embed_err:
                print(f"[ERROR] Error generating embeddings: {str(embed_err)}")
                raise Exception(f"Failed to generate embeddings: {str(embed_err)}")
            
            # Call the Supabase function
            print("[INFO] Calling Supabase search_travel_packages_v2 function")
            
            # Using the vector_store's supabase client
            supabase_client = self.vector_store.client
            
            # Prepare payload for logging
            payload = {
                "location_input": location_input,
                "budget_vector_input": f"<vector with length {len(budget_vector) if budget_vector else 0}>",
                "accommodation_vector_input": f"<vector with length {len(accommodation_vector) if accommodation_vector else 0}>",
                "activities_vector_input": f"<vector with length {len(activities_vector) if activities_vector else 0}>",
                "notes_vector_input": f"<vector with length {len(notes_vector) if notes_vector else 0}>",
                "match_count": match_count
            }
            print(f"[DEBUG] RPC payload: {payload}")
            
            # Prepare the actual payload for the RPC call
            rpc_payload = {
                "location_input": location_input,
                "budget_vector_input": budget_vector,
                "accommodation_vector_input": accommodation_vector,
                "activities_vector_input": activities_vector,
                "notes_vector_input": notes_vector,
                "match_count": match_count
            }
            
            try:
                print(f"[DEBUG] Calling RPC function 'search_travel_packages_v3' with payload keys: {rpc_payload.keys()}")
                response = supabase_client.rpc(
                    "search_travel_packages_v3",
                    rpc_payload
                ).execute()
                
                if hasattr(response, "error") and response.error is not None:
                    print(f"[ERROR] Error from Supabase search_travel_packages_v3: {response.error}")
                    raise Exception(f"Supabase search error: {response.error}")
                
                # The response should contain a data field with the results
                packages = response.data if response.data else []
                print(f"[INFO] Found {len(packages)} packages from search_travel_packages_v2")
                
                # Log a sample of the packages found (just first one)
                if packages and len(packages) > 0:
                    print(f"[DEBUG] Sample package: {packages[0]}")
                
                return packages
                
            except Exception as rpc_err:
                print(f"[ERROR] Error during Supabase RPC call: {str(rpc_err)}")
                raise Exception(f"Supabase RPC error: {str(rpc_err)}")
                
        except Exception as e:
            print(f"[ERROR] Error in search_travel_packages_v2: {str(e)}")
            raise
    
    async def generate_tour_suggestion(
        self,
        base_tours: List[Dict[str, Any]],
        num_participants: int,
        preferred_activities: str,
        accommodation_preference: str,
        budget_range: str,
        duration_adjustment: str = "",
        location_input: str = "",
        num_suggestions: int = 3
    ) -> TourSuggestionResponse:
        """
        Generate multiple tour suggestions by combining base tours and customizing them.
        
        Args:
            base_tours: List of base tour packages to combine
            num_participants: Number of participants
            preferred_activities: Comma-separated list of preferred activity types
            accommodation_preference: Accommodation preference
            budget_range: Budget range
            duration_adjustment: Duration adjustment as a string (e.g., "2 days longer", "shorter trip")
            location_input: The main location that should be the focus of the tour
            num_suggestions: Number of different tour suggestions to generate
            
        Returns:
            A structured tour suggestion response with multiple tour options
        """
        print(f"[INFO] Generating {num_suggestions} tour suggestions for {len(base_tours)} base tours in/near {location_input}")
        
        # Format the base tours for the prompt
        formatted_tours = []
        for i, tour in enumerate(base_tours, 1):
            formatted_tour = (
                f"**Base Tour {i}**: {tour.get('title', f'Tour {i}')} "
                f"({tour.get('duration_days', 'unknown')} days), "
                f"location: {tour.get('location', 'Unknown location')}, "
                f"highlights: {', '.join(tour.get('highlights', ['unknown']))}, "
                f"price: ${tour.get('price', 0)}, "
                f"description: {tour.get('description', 'No description available')}"
            )
            formatted_tours.append(formatted_tour)
        
        # Format the customization parameters
        customization = {
            "Number of participants": num_participants,
            "Preferred activities": preferred_activities,
            "Accommodation preference": accommodation_preference,
            "Budget range": budget_range,
            "Location": location_input
        }
        
        if duration_adjustment:
            customization["Duration adjustment"] = duration_adjustment
        
        # Create a system message with the SUGGEST_TOUR_TEMPLATE
        system_prompt = SUGGEST_TOUR_TEMPLATE
        
        # Build the prompt components separately to avoid f-string backslash issues
        base_tours_text = "## Base Tours:\n"
        for tour in formatted_tours:
            base_tours_text += f"{tour}\n"
        
        # Build the customization parameters section
        params_text = "## Customization Parameters:\n"
        params_text += f"- Number of participants: {num_participants}\n"
        params_text += f"- Preferred activities: {preferred_activities}\n"
        params_text += f"- Accommodation preference: {accommodation_preference}\n"
        params_text += f"- Budget range: {budget_range}\n"
        params_text += f"- Main location: {location_input}\n"
        params_text += f"- Number of tour suggestions to generate: {num_suggestions}\n"
        
        if duration_adjustment:
            params_text += f"- Duration adjustment: {duration_adjustment}\n"
        
        # Build the complete prompt
        user_prompt = (
            f"Please create {num_suggestions} different tour suggestions by combining these base tour packages based on the parameters for a tour in/near {location_input}:\n\n" +
            base_tours_text + "\n" +
            params_text + "\n" +
            f"Please provide {num_suggestions} distinct suggested tour packages that offer different experiences " +
            "while still using the base tours. Make each suggestion unique in terms of itinerary, duration, or focus. " +
            f"All tours MUST be focused on {location_input} and only include nearby locations (within 1-2 hours travel maximum). " +
            "Include a daily itinerary, pricing breakdown, accommodation options, transportation details, and highlights for each suggestion. " +
            f"DO NOT suggest activities or destinations that are far from {location_input} (more than 2 hours travel). " +
            "IMPORTANT: Format the description field for each suggested tour in markdown format, using the same styling as shown in the base tours."
        )
        
        # Call the OpenAI API to generate the tour suggestion with structured output
        try:
            print("[INFO] Calling OpenAI API for tour suggestions")
            
            # Use the parse method for structured output
            from app.models.suggest_tour_models import TourSuggestionResponse, TourSuggestion
            
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o", # Using a capable model for complex tour planning
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=TourSuggestionResponse,
                temperature=0.9, # Increased temperature for more diverse suggestions
            )
            
            # Get the parsed structured output
            tour_response = completion.choices[0].message.parsed
            
            # Ensure we have the expected number of suggestions, or at least one
            if not tour_response.suggested_tours or len(tour_response.suggested_tours) == 0:
                print("[WARNING] No tour suggestions were generated, creating a fallback suggestion")
                # Create at least one suggestion as fallback
                fallback_suggestion = TourSuggestion(
                    title=f"Tour Package for {location_input}",
                    description=f"A customized tour package for {location_input} and nearby areas.",
                    total_duration=5,
                    pricing={
                        "total_price_per_person": 500.0,
                        "total_price_group": float(500.0 * num_participants),
                        "includes": ["Accommodation", "Local transportation"],
                        "excludes": ["International flights", "Personal expenses"]
                    },
                    accommodation=[{
                        "name": "Standard Hotel",
                        "type": accommodation_preference,
                        "description": "Standard accommodation based on your preferences.",
                        "price_range": budget_range
                    }],
                    transportation={
                        "methods": ["Local transportation"],
                        "description": "Standard local transportation options."
                    },
                    itinerary=[{
                        "day": 1,
                        "activities": ["Arrival and check-in", "Welcome dinner"],
                        "description": "Arrival day with welcome activities."
                    }],
                    highlights=["Customized tour based on your preferences"]
                )
                tour_response.suggested_tours = [fallback_suggestion]
            
            # Set the base_tours_used field with the complete tour data
            base_tours_list = []
            for tour in base_tours:
                try:
                    # Create a BaseTour object from each tour dictionary
                    base_tour = BaseTour(
                        id=tour.get('id', ''),
                        title=tour.get('title', 'Unnamed Tour'),
                        description=tour.get('description', 'No description available'),
                        duration_days=tour.get('duration_days', 0),
                        price=float(tour.get('price', 0)),
                        location=tour.get('location', ''),
                        highlights=tour.get('highlights', []),
                        accommodation_type=tour.get('accommodation_type', None),
                        activities=tour.get('activities', []),
                        transportation=tour.get('transportation', None)
                    )
                    base_tours_list.append(base_tour)
                except Exception as e:
                    print(f"[WARNING] Error converting tour to BaseTour model: {str(e)}")
                    # Continue with other tours if one fails
            
            tour_response.base_tours_used = base_tours_list
            
            # Log the number of suggestions generated
            print(f"[INFO] Generated {len(tour_response.suggested_tours)} tour suggestions")
            
            return tour_response
                
        except Exception as e:
            print(f"[ERROR] Error generating tour suggestions: {str(e)}")
            raise Exception(f"Tour suggestion generation failed: {str(e)}") 