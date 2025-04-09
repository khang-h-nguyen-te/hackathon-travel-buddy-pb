from pydantic import BaseModel, field_validator, model_validator
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

    @field_validator('highlights', mode='before')
    @classmethod
    def parse_highlights(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                # Handle string representation of list
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
                # If literal_eval doesn't return a list, fallback to splitting
                return [x.strip() for x in v.strip('[]').split(',') if x.strip()]
            except (ValueError, SyntaxError):
                 # If the string is not a valid Python literal, split by comma
                 # Ensure splitting non-empty strings after stripping
                 return [x.strip() for x in v.strip('[]').split(',') if x.strip()]
        # If it's already a list or other type, pass it through
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

    @model_validator(mode='before')
    @classmethod
    def empty_string_to_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                # Apply logic only to string fields, excluding match_count
                if key != 'match_count' and key in cls.model_fields and value is None:
                     # Check if the field is expected to be a string (or Optional[str])
                     # This check might need refinement based on actual field types if they vary more
                    annotation = cls.model_fields[key].annotation
                    if annotation == str or annotation == Optional[str]:
                        data[key] = ""
        return data


class TravelPackageSearchResponse(BaseModel):
    """Response model for the /search-travel-packages endpoint."""
    packages: List[TravelPackage]
    total_count: int


class APIResponse(BaseModel):
    """Standard API response model."""
    message: Any
    status_code: int
    error: bool = False 