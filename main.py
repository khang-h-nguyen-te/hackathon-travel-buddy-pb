from typing import Annotated, List, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import litellm

# Import from our application structure
from app.agent.agent_rag import AgentRag
from app.models.request_models import (
    QueryRequest, 
    SignInRequest, 
    TravelPackageSearchRequest,
    TravelPackageSearchResponse,
    TourSuggestionRequest,
    TravelPackage
)
from app.utils.response_utils import create_response, validate_params
from app.utils.crypto_utils import encrypt_password, decrypt_password
from app.history.history_module import HistoryModule
from app.config.supabase_config import get_supabase_client
from app.config.env_config import config
from app.services.embeddings import EmbeddingService
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore
from app.tools.search.search_tools import SearchTravelPackagesTool
from app.models.suggest_tour_models import CustomizationParameters, TourSuggestionResponse
from app.services.tour_suggestion_service import TourSuggestionService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title="Meeting Chatbot API",
    description="API for interacting with the Meeting Chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

executor = ThreadPoolExecutor(max_workers=4)

# Create Supabase client
supabase = get_supabase_client()

# Create a global HistoryModule instance
chat_history_module = HistoryModule()  # Now uses config for token limit
agent_initializer = AgentRag(history_module=chat_history_module)

# Create logger for the FastAPI app
logger = logging.getLogger(__name__)

# Create a tour suggestion service instance
tour_service = TourSuggestionService(
    vector_store=None,  # Will be initialized properly in the endpoint
    embedding_service=None  # Will be initialized properly in the endpoint
)

@app.post("/authenticate/{command}")
async def authenticate(command: str, payload: SignInRequest):
    """
    Authenticate a user and initialize the agent.
    
    Args:
        command: The authentication command to execute
        payload: The authentication request payload
    
    Returns:
        Authentication response with token
    """
    params = payload.dict()

    if command == 'signInWithPassword':
        if not validate_params(params, ['email', 'password']):
            raise HTTPException(status_code=400, detail="Missing required parameters")

        try:
            # Decrypt password if it's encrypted
            try:
                password = decrypt_password(params['password'])
            except Exception:
                # If decryption fails, use the password as is
                password = params['password']
                
            response = supabase.auth.sign_in_with_password({
                'email': params['email'],
                'password': password
            })
            agent_initializer.setup_agent(response.session.access_token)
            logger.info(f"Agent initialized for user: {params['email']}")
            return create_response(response, 200)
        
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    elif command == 'signUpWithPassword':
        if not validate_params(params, ['email', 'password', 'display_name']):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        try:
            # Decrypt password if it's encrypted
            try:
                password = decrypt_password(params['password'])
            except Exception:
                # If decryption fails, use the password as is
                password = params['password']
                
            response = supabase.auth.sign_up({
                'email': params['email'],
                'password': password,
            })

            ## Not yet implemented set profile data

            logger.info(f"User account created: {params['email']}")
            return create_response(response, 200)
        
        except Exception as e:
            logger.error(f"Sign up error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=404, detail=f"Authentication type '{command}' not recognized")

# A helper function to process the query synchronously
def process_query(query: str) -> str:
    """
    Process a user query using the agent.
    
    Args:
        query: The user's question
    
    Returns:
        The agent's response
    """
    response = agent_initializer.agent_query(query)
    return response
    
# Define a POST endpoint to receive user queries
@app.post("/ask")
async def ask_query(payload: QueryRequest, authorization: str):
    """
    Process a user question and return the agent's response.
    
    Args:
        payload: The query request containing the user question
        request: The FastAPI request object
    
    Returns:
        The agent's response
    """
    query = payload.query
    auth_header = authorization
    logger.debug(f"Processing query: {query}")
    
    # Offload the blocking agent call to a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, process_query, query)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to process query")
    
    return {"response": result}

# A helper function to process the travel package search synchronously
def process_travel_search(
    location_input: str,
    duration_input: str,
    budget_input: str,
    transportation_input: str,
    accommodation_input: str,
    food_input: str,
    activities_input: str,
    notes_input: str,
    match_count: int,
    vector_store: SupabaseVectorStore,
    embedding_service: EmbeddingService
) -> List[Dict]:
    """
    Process a travel package search using the search tool.
    
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
        vector_store: The Supabase vector store instance
        embedding_service: The embedding service instance
    
    Returns:
        List of travel package dictionaries
    """
    search_tool = SearchTravelPackagesTool(
        vector_store=vector_store,
        embedding_service=embedding_service
    )
    
    # Get the raw results (list of dictionaries) directly from the search tool
    packages = search_tool(
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

    logger.info(f"Raw packages from search tool: {packages}")
    
    # No need to parse string results anymore
    # Simply return the list of dictionaries
    return packages

# --- Helper function for LLM Reranking (New) ---
async def process_llm_rerank(
    search_request: TravelPackageSearchRequest,
    candidate_packages: List[Dict]
) -> List[TravelPackage]:
    """
    Uses an LLM to re-rank/select candidate packages based on user criteria.

    Args:
        search_request: The original search request payload.
        candidate_packages: List of package dictionaries from initial search.

    Returns:
        List of validated TravelPackage objects selected by the LLM.
    """
    logger.info(f"Starting LLM re-ranking for {len(candidate_packages)} candidates.")

    if not candidate_packages:
        logger.warning("No candidate packages provided for LLM reranking.")
        return []
        
    if not config.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY is not configured. Skipping LLM reranking.")
        # Fallback: Return original candidates as TravelPackage objects
        validated_packages = []
        for pkg_dict in candidate_packages:
            try:
                validated_packages.append(TravelPackage.model_validate(pkg_dict))
            except Exception as e:
                logger.error(f"Failed to validate candidate package: {pkg_dict.get('id', 'N/A')}. Error: {e}")
        return validated_packages


    # Prepare user criteria string
    criteria = f"""
    User Search Criteria:
    - Location: {search_request.location_input or 'Any'}
    - Duration: {search_request.duration_input or 'Any'}
    - Budget: {search_request.budget_input or 'Any'}
    - Transportation: {search_request.transportation_input or 'Not specified'}
    - Accommodation: {search_request.accommodation_input or 'Not specified'}
    - Food: {search_request.food_input or 'Not specified'}
    - Activities: {search_request.activities_input or 'Any'}
    - Notes: {search_request.notes_input or 'None'}
    """

    # Prepare candidate packages JSON string
    try:
        candidates_json_str = json.dumps(candidate_packages, indent=2)
    except Exception as e:
        logger.error(f"Failed to serialize candidate packages to JSON: {e}")
        return [] # Cannot proceed without candidates

    # Construct the prompt for the LLM
    prompt_messages = [
        {
            "role": "system",
            "content": f"""
You are an expert travel agent assistant. Your task is to rerank the travel packages for a user based on their preferences and a list of candidate packages found.

**Instructions:**
1.  Review the User Search Criteria provided.
2.  Examine the list of Candidate Travel Packages provided below. Each package is a JSON object.
3.  Rerank the packages from the provided list that match the user's criteria. Consider all aspects like location (most important), duration, budget hints (interpret budget strings like 'around $500', 'budget friendly'), activities, etc.
4.  **Do not invent new packages or modify details** of the existing packages (like price, duration, description, id etc.). Only select from the candidates.
5.  Return your selection as a **single JSON array** containing the complete JSON objects of the selected packages. The JSON array should be the *only* content in your response.
6.  If none of the candidate packages are a good match for the criteria, return an empty JSON array `[]`.

Example output format (should be a JSON array of package objects):
```json
[
  {{
    "id": "pkg_123",
    "title": "Amazing Beach Trip",
    "provider_id": "prov_abc",
    "location_id": "loc_xyz",
    "price": 499.99,
    "duration_days": 5,
    "highlights": ["Swimming", "Sunbathing"],
    "description": "Relax on the beautiful beaches...",
    "image_url": "http://example.com/image.jpg"
  }},
  {{
    "id": "pkg_456",
    // ... other fields ...
  }}
]
```
"""
        },
        {
            "role": "user",
            "content": f"""
{criteria}

Candidate Travel Packages:
```json
{candidates_json_str}
```

Based on the User Search Criteria, please select some high matching packages from the list above and return them as a JSON array. Remember to only include packages from the provided list and maintain their original details.
"""
        }
    ]

    llm_selected_packages = []
    try:
        logger.debug("Calling LLM for reranking...")
        # Set API key for litellm (it reads environment variables, but setting explicitly can be clearer)
        litellm.api_key = config.openrouter_api_key 
        # Optionally set base URL if needed, but often inferred for OpenRouter
        # litellm.api_base = "https://openrouter.ai/api/v1" 

        response = await litellm.acompletion( # Use async completion
            model="openrouter/meta-llama/llama-3.3-70b-instruct", # Using Llama 3 70b via OpenRouter
            messages=prompt_messages,
            response_format={"type": "json_object"}, # Request JSON output
            temperature=0.5 # Lower temperature for more deterministic selection
        )

        logger.debug("LLM response received.")
        llm_output_content = response.choices[0].message.content

        if not llm_output_content:
            logger.warning("LLM returned empty content.")
            return []

        # Parse the JSON output from the LLM
        try:
            parsed_llm_output = json.loads(llm_output_content)
            if not isinstance(parsed_llm_output, list):
                 # Sometimes the LLM might wrap the list in a top-level key, try to find it
                 if isinstance(parsed_llm_output, dict) and len(parsed_llm_output) == 1:
                     key = list(parsed_llm_output.keys())[0]
                     if isinstance(parsed_llm_output[key], list):
                         parsed_llm_output = parsed_llm_output[key]
                         logger.warning("LLM output was nested in a dict, extracted list.")
                     else:
                        raise ValueError("LLM JSON output was not a list or a dict containing a single list.")
                 else:
                    raise ValueError("LLM JSON output was not a list.")

            # Validate each object in the list against the TravelPackage model
            validated_packages = []
            original_ids = {pkg['id'] for pkg in candidate_packages}

            for item in parsed_llm_output:
                if not isinstance(item, dict):
                    logger.warning(f"LLM returned a non-dictionary item in the list: {item}")
                    continue
                
                item_id = item.get('id')
                if not item_id:
                     logger.warning(f"LLM returned a package without an ID: {item}")
                     continue
                
                # Crucially, check if the ID was in the original candidate list
                if item_id not in original_ids:
                    logger.warning(f"LLM returned a package ID ('{item_id}') that was not in the original candidates. Skipping.")
                    continue

                try:
                    # Validate the structure and types using the Pydantic model
                    validated_pkg = TravelPackage.model_validate(item)
                    validated_packages.append(validated_pkg)
                except Exception as pydantic_error:
                    logger.warning(f"LLM returned package (ID: {item_id}) that failed Pydantic validation: {pydantic_error}. Item: {item}")

            llm_selected_packages = validated_packages
            logger.info(f"LLM successfully selected {len(llm_selected_packages)} packages after validation.")

        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to decode JSON response from LLM: {json_error}")
            logger.error(f"LLM Raw Output: {llm_output_content}")
        except ValueError as val_error:
             logger.error(f"LLM JSON output validation error: {val_error}")
             logger.error(f"LLM Raw Output: {llm_output_content}")


    except Exception as e:
        logger.error(f"Error during LLM reranking call: {str(e)}", exc_info=True)
        # Optionally, could fallback to returning original candidates here too

    return llm_selected_packages

# --- Original Search Endpoint ---
@app.post("/search-travel-packages", response_model=TravelPackageSearchResponse)
async def search_travel_packages(
    authorization: str,
    payload: TravelPackageSearchRequest
):
    # ... (existing code for validation and setup) ...
    auth_header = authorization
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Valid Authorization header with Bearer token is required"
        )
    
    # Initialize services
    try:
        embedding_service = EmbeddingService()
        vector_store = SupabaseVectorStore(
        url=config.supabase_url,
        key=config.supabase_anon_key,
            auth=auth_header.replace("Bearer ", "") # Use service key for backend search
        )
    except Exception as e:
        logger.error(f"Failed to initialize search services: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize search services")

    logger.debug(f"Processing search request: {payload.dict()}")

    # Offload the blocking search call to a thread pool
    loop = asyncio.get_event_loop()
    try:
        package_dicts = await loop.run_in_executor(
        executor,
            process_travel_search, # Assuming process_travel_search returns List[Dict]
        payload.location_input,
        payload.duration_input,
        payload.budget_input,
        payload.transportation_input,
        payload.accommodation_input,
        payload.food_input,
        payload.activities_input,
        payload.notes_input,
        payload.match_count,
        vector_store,
        embedding_service
    )
    
        # Validate results into Pydantic models
        validated_packages = []
        for pkg_dict in package_dicts:
            try:
                validated_packages.append(TravelPackage.model_validate(pkg_dict))
            except Exception as e:
                logger.error(f"Failed to validate search result package: {pkg_dict.get('id', 'N/A')}. Error: {e}")
                # Decide whether to skip or raise error

    except Exception as e:
        logger.error(f"Error during travel package search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process travel package search")

    response = TravelPackageSearchResponse(
        packages=validated_packages,
        total_count=len(validated_packages)
    )
    return response

# --- New Search Endpoint with LLM Reranking ---
@app.post("/search-travel-packages-v2", response_model=TravelPackageSearchResponse)
async def search_travel_packages_v2(
    authorization: str,
    payload: TravelPackageSearchRequest
):
    """
    Search for travel packages, then use an LLM to re-rank/select results.
    """
    # --- Step 1: Initial Search (same as original endpoint) ---
    auth_header = authorization
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Valid Authorization header with Bearer token is required"
        )
        
    try:
        embedding_service = EmbeddingService()
        vector_store = SupabaseVectorStore(
            url=config.supabase_url,
            key=config.supabase_anon_key,
            auth=auth_header.replace("Bearer ", "")
        )
    except Exception as e:
        logger.error(f"Failed to initialize search services (v2): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize search services")

    logger.debug(f"Processing v2 search request: {payload.dict()}")
    loop = asyncio.get_event_loop()
    try:
        # Get initial candidates as dictionaries
        initial_package_dicts = await loop.run_in_executor(
            executor,
            process_travel_search,
            payload.location_input,
            payload.duration_input,
            payload.budget_input,
            payload.transportation_input,
            payload.accommodation_input,
            payload.food_input,
            payload.activities_input,
            payload.notes_input,
            payload.match_count, # Fetch initial candidates
            vector_store,
            embedding_service
        )
        logger.info(f"Initial search found {len(initial_package_dicts)} candidates.")

    except Exception as e:
        logger.error(f"Error during initial travel package search (v2): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process initial travel package search")

    # --- Step 2: LLM Re-ranking/Selection ---
    if not initial_package_dicts:
        # If no initial results, return empty
         return TravelPackageSearchResponse(packages=[], total_count=0)
         
    try:
        # Run the LLM processing asynchronously
        reranked_packages = await process_llm_rerank(payload, initial_package_dicts)
        logger.info(f"LLM reranking resulted in {len(reranked_packages)} packages.")
    except Exception as e:
        logger.error(f"Error during LLM reranking process: {str(e)}", exc_info=True)
        # Fallback strategy: Return the original results if LLM fails? Or empty?
        # For now, let's return empty if LLM stage fails critically.
        # Consider returning initial results converted to TravelPackage if preferred.
        reranked_packages = [] # Default to empty on critical LLM failure
        # Example fallback to initial results:
        # reranked_packages = [TravelPackage.model_validate(pkg) for pkg in initial_package_dicts]


    # --- Step 3: Final Response ---
    response = TravelPackageSearchResponse(
        packages=reranked_packages,
        total_count=len(reranked_packages)
    )
    return response

# Define a POST endpoint to search travel packages v2 and suggest combined tours
@app.post("/suggest-tour", response_model=TourSuggestionResponse)
async def suggest_tour(
    authorization: str,
    request: TourSuggestionRequest
):
    """
    Search for travel packages and generate a combined tour suggestion.
    
    First searches for travel packages using vector search, then uses LLM to create a personalized tour suggestion
    by combining and customizing the found packages.
    """
    # Validate authorization
    auth_header = authorization
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Valid Authorization header with Bearer token is required"
        )
    
    # Initialize services
    try:
        embedding_service = EmbeddingService()
        vector_store = SupabaseVectorStore(
            url=config.supabase_url,
            key=config.supabase_anon_key,
            auth=auth_header # Use service key for backend search
        )
        
        # Update the service with initialized components
        tour_service.vector_store = vector_store
        tour_service.embedding_service = embedding_service
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize services for suggest-tour: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize search services")

    # Extract values from the request model
    location_input = request.location_input
    budget_input = request.budget_input
    accommodation_input = request.accommodation_input
    activities_input = request.activities_input
    num_participants = request.num_participants
    preferred_activities = request.preferred_activities
    accommodation_preference = request.accommodation_preference
    budget_range = request.budget_range
    duration_adjustment = request.duration_adjustment
    match_count = request.match_count

    # Build notes_input by concatenating other fields
    notes_elements = []
    if location_input:
        notes_elements.append(f"Location: {location_input}")
    if budget_input:
        notes_elements.append(f"Budget: {budget_input}")
    if budget_range and budget_range != budget_input:
        notes_elements.append(f"Budget range: {budget_range}")
    if accommodation_input:
        notes_elements.append(f"Accommodation: {accommodation_input}")
    if accommodation_preference and accommodation_preference != "mid-range":
        notes_elements.append(f"Accommodation preference: {accommodation_preference}")
    if activities_input:
        notes_elements.append(f"Activities: {activities_input}")
    if preferred_activities and preferred_activities != activities_input:
        notes_elements.append(f"Preferred activities: {preferred_activities}")
    if num_participants != 2:
        notes_elements.append(f"Group size: {num_participants} participants")
    if duration_adjustment:
        notes_elements.append(f"Duration adjustment: {duration_adjustment}")
    
    notes_input = ". ".join(notes_elements)
    print(f"[DEBUG] Generated notes_input: {notes_input}")
    print(f"[DEBUG] Processing suggest-tour request with location: {location_input}, activities: {activities_input} and budget: {budget_input}")
    
    # Step 1: Search for base tour packages using v2 search
    try:
        # Use the budget_range as budget_input if budget_input is empty
        if not budget_input and budget_range:
            budget_input = budget_range
            
        # Use preferred_activities as activities_input if activities_input is empty
        if not activities_input and preferred_activities:
            activities_input = preferred_activities
            
        # Search for base packages
        try:
            print("[DEBUG] Calling search_travel_packages_v2 with parameters:")
            print(f"[DEBUG] location_input: {location_input}")
            print(f"[DEBUG] budget_input: {budget_input}")
            print(f"[DEBUG] accommodation_input: {accommodation_input}")
            print(f"[DEBUG] activities_input: {activities_input}")
            print(f"[DEBUG] notes_input: {notes_input}")
            print(f"[DEBUG] match_count: {match_count}")
            
            base_packages = await tour_service.search_travel_packages_v2(
                location_input=location_input,
                budget_input=budget_input,
                accommodation_input=accommodation_input,
                activities_input=activities_input,
                notes_input=notes_input,
                match_count=match_count
            )
        except Exception as e:
            print(f"[ERROR] Error details in search_travel_packages_v2: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to search for base tour packages")
        
        if not base_packages:
            print("[WARNING] No base packages found for the given search criteria")
            raise HTTPException(
                status_code=404, 
                detail="No tour packages found matching your criteria. Please try different search parameters."
            )
            
        print(f"[INFO] Found {len(base_packages)} base packages for tour suggestion")
            
    except Exception as e:
        print(f"[ERROR] Error during base package search: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search for base tour packages")
    
    # Step 2: Generate tour suggestion
    try:
        # If no preferred_activities were provided but activities_input exists, use that
        if not preferred_activities and activities_input:
            preferred_activities = activities_input
        
        # If preferred_activities is still empty, provide a default
        if not preferred_activities:
            preferred_activities = "Sightseeing"
            
        # If no budget_range was provided but budget_input exists, use that
        if not budget_range and budget_input:
            budget_range = budget_input
            
        # Generate tour suggestion
        suggestion_response = await tour_service.generate_tour_suggestion(
            base_tours=base_packages,
            num_participants=num_participants,
            preferred_activities=preferred_activities,
            accommodation_preference=accommodation_preference,
            budget_range=budget_range,
            duration_adjustment=duration_adjustment,
            location_input=location_input
        )
        
        print("[INFO] Successfully generated tour suggestion")
        return suggestion_response
        
    except Exception as e:
        print(f"[ERROR] Error during tour suggestion generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate tour suggestion")

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "ok"}

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 