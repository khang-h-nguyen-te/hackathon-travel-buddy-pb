import logging
import sys
from typing import Dict, List
import json
from pprint import pprint

# Add parent directory to path so we can import from app
sys.path.append("..")

from app.config.env_config import config
from app.services.embeddings import EmbeddingService
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore
from app.tools.search.search_tools import SearchTravelPackagesTool

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator(message: str = ""):
    print("\n" + "="*50)
    if message:
        print(message)
        print("="*50)

def debug_search_packages(
    location_input: str = "beach vacation in Vietnam",
    duration_input: str = "5 days",
    budget_input: str = "around $500",
    transportation_input: str = "comfortable travel",
    accommodation_input: str = "nice hotel",
    food_input: str = "local food",
    activities_input: str = "swimming, relaxing",
    notes_input: str = "peaceful location",
    match_count: int = 5
) -> None:
    """Debug the search packages functionality with detailed logging."""
    
    print_separator("Initializing Services")
    embedding_service = EmbeddingService()
    vector_store = SupabaseVectorStore(
        url=config.supabase_url,
        key=config.supabase_anon_key,
        auth=config.supabase_service_key  # Using service key for debugging
    )
    
    print_separator("Creating Search Tool")
    search_tool = SearchTravelPackagesTool(
        vector_store=vector_store,
        embedding_service=embedding_service
    )
    
    # Log input parameters
    print_separator("Search Parameters")
    params = {
        "location_input": location_input,
        "duration_input": duration_input,
        "budget_input": budget_input,
        "transportation_input": transportation_input,
        "accommodation_input": accommodation_input,
        "food_input": food_input,
        "activities_input": activities_input,
        "notes_input": notes_input,
        "match_count": match_count
    }
    print("Input parameters:")
    pprint(params)
    
    # Generate and log embeddings
    print_separator("Generating Embeddings")
    embeddings = {}
    for key, value in params.items():
        if key != "match_count" and value:
            try:
                embedding = embedding_service.get_embedding(value)
                embeddings[key] = {
                    "text": value,
                    "embedding_size": len(embedding),
                    "embedding_sample": embedding[:5]  # Show first 5 values
                }
            except Exception as e:
                logger.error(f"Error generating embedding for {key}: {str(e)}")
    
    print("Embeddings generated:")
    pprint(embeddings)
    
    # Perform search
    print_separator("Performing Search")
    try:
        results = search_tool(
            location_input=location_input,
            duration_input=duration_input,
            budget_input=budget_input,
            transportation_input=transportation_input,
            accommodation_input=accommodation_input,
            food_input=food_input,
            activities_input=activities_input,
            notes_input=notes_input,
            match_count=match_count
        )
        
        print("\nSearch Results:")
        if isinstance(results, list):
            print(f"Number of results: {len(results)}")
            for idx, result in enumerate(results, 1):
                print(f"\nResult {idx}:")
                pprint(result)
        else:
            print("Unexpected result format:")
            pprint(results)
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "name": "Beach Vacation Test",
            "params": {
                "location_input": "beach vacation in Vietnam",
                "duration_input": "5 days",
                "budget_input": "around $500",
                "activities_input": "swimming, relaxing",
                "notes_input": ""
            }
        },
        {
            "name": "Adventure Trip Test",
            "params": {
                "location_input": "mountain trekking in Vietnam",
                "duration_input": "3 days",
                "budget_input": "budget friendly",
                "activities_input": "hiking, camping",
                "notes_input": "adventurous experience"
            }
        }
    ]
    
    for test_case in test_cases:
        print_separator(f"Running Test Case: {test_case['name']}")
        debug_search_packages(**test_case["params"])
        print("\n" + "="*50 + "\n") 