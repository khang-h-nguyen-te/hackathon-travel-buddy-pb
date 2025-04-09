from typing import Annotated, List, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Import from our application structure
from app.agent.agent_rag import AgentRag
from app.models.request_models import (
    QueryRequest, 
    SignInRequest, 
    TravelPackageSearchRequest,
    TravelPackageSearchResponse,
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

# Define a POST endpoint to search travel packages
@app.post("/search-travel-packages", response_model=TravelPackageSearchResponse)
async def search_travel_packages(
    authorization: str,
    payload: TravelPackageSearchRequest
):
    """
    Search for travel packages based on user preferences.
    
    Args:
        payload: The travel package search request containing user preferences
        request: The FastAPI request object
    
    Returns:
        List of travel packages matching the search criteria
    """
    # Get the Authorization header from the request
    auth_header = authorization
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Valid Authorization header with Bearer token is required"
        )
    
    # Extract the token from the Authorization header
    # Remove 'Bearer ' prefix and any extra spaces
    token = auth_header.replace("Bearer ", "").strip()
    
    # Initialize vector store and embedding service
    vector_store = SupabaseVectorStore(
        url=config.supabase_url,
        key=config.supabase_anon_key,
        auth=token  # Pass the clean token without 'Bearer ' prefix
    )
    embedding_service = EmbeddingService()
    
    # Offload the blocking search to a thread pool
    loop = asyncio.get_event_loop()
    packages = await loop.run_in_executor(
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
        payload.match_count,
        vector_store,
        embedding_service
    )
    

    valid_packages = []
    # Get required field names (works for Pydantic v1 and v2)
    required_keys = {name for name, field in TravelPackage.__fields__.items() if field.is_required}
    # For Pydantic v2 specifically, you could use:
    # required_keys = {name for name, field_info in TravelPackage.model_fields.items() if field_info.is_required()}
    # We'll use the __fields__ approach for broader compatibility for now.

    for pkg in packages:
        # Remove the combined_score if it exists
        pkg.pop('combined_score', None)

        # Check if all required keys are present
        if all(key in pkg for key in required_keys):
            # Optional: Add further checks here if needed, e.g., ensure values are not None
            valid_packages.append(pkg)
        else:
            # Log a warning or handle the incomplete package data
            missing_keys = required_keys - pkg.keys()
            logger.warning(f"Skipping incomplete package data: {pkg}. Missing required keys: {missing_keys}")

    travel_packages = [TravelPackage(**pkg) for pkg in valid_packages]

    return TravelPackageSearchResponse(
        packages=travel_packages,
        total_count=len(travel_packages)
    )

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "ok"}

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
# if __name__ == "__main__":
#     # Run the FastAPI app using uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 