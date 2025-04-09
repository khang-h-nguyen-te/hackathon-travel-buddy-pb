from pydantic import BaseModel, validator
from typing import Any, Dict, Optional, Union, List
import ast


class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str


class SignInRequest(BaseModel):
    """Request model for the /authenticate endpoint."""
    email: str
    password: str
    display_name: Optional[str] = None


class TravelPackage(BaseModel):
    """Model for a travel package."""
    id: str
    title: str
    provider_id: str
    location_id: str
    price: float
    duration_days: int
    highlights: List[str]
    description: str
    image_url: Optional[str] = None
    # Note: Vector fields are not included in the response model
    # as they are used internally for search only

    @validator('highlights', pre=True)
    def parse_highlights(cls, v):
        if isinstance(v, str):
            try:
                # Handle string representation of list
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                # If the string is not a valid Python literal, split by comma
                return [x.strip() for x in v.strip('[]').split(',')]
        return v


class TravelPackageSearchRequest(BaseModel):
    """Request model for the /search-travel-packages endpoint."""
    location_input: str = ""
    duration_input: str = ""
    budget_input: str = ""
    transportation_input: str = ""
    accommodation_input: str = ""
    food_input: str = ""
    activities_input: str = ""
    notes_input: str = ""
    match_count: Optional[int] = 10

    @validator('*', pre=True)
    def empty_string_to_none(cls, v, field):
        if field.name == 'match_count':
            return v
        return v if v is not None else ""


class TravelPackageSearchResponse(BaseModel):
    """Response model for the /search-travel-packages endpoint."""
    packages: List[TravelPackage]
    total_count: int


class APIResponse(BaseModel):
    """Standard API response model."""
    message: Any
    status_code: int
    error: bool = False 