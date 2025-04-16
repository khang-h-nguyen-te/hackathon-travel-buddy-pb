from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CustomizationParameters(BaseModel):
    """Input parameters for customizing tour suggestions."""
    num_participants: int = Field(..., description="Number of participants in the tour")
    preferred_activities: str = Field(..., description="Comma-separated list of preferred activity types")
    accommodation_preference: str = Field(..., description="Accommodation preference (e.g., 'budget', 'mid-range', 'luxury')")
    budget_range: str = Field(..., description="Budget range for the entire trip")
    duration_adjustment: Optional[int] = Field(None, description="Optional adjustment to the tour duration in days")

class PricingDetails(BaseModel):
    """Structured pricing details for the tour"""
    total_price_per_person: float = Field(..., description="Total price per person")
    total_price_group: float = Field(..., description="Total price for the entire group")
    includes: Optional[List[str]] = Field(default_factory=list, description="What the price includes")
    excludes: Optional[List[str]] = Field(default_factory=list, description="What the price excludes")

class AccommodationOption(BaseModel):
    """Accommodation option model"""
    name: str = Field(..., description="Name of the accommodation")
    type: str = Field(..., description="Type of accommodation (hotel, hostel, etc.)")
    description: str = Field(..., description="Description of the accommodation")
    price_range: Optional[str] = Field(None, description="Price range for the accommodation")

class TransportationDetails(BaseModel):
    """Transportation details model"""
    methods: List[str] = Field(..., description="Transportation methods used")
    description: str = Field(..., description="Description of transportation arrangements")

class ItineraryDay(BaseModel):
    """Daily itinerary model"""
    day: int = Field(..., description="Day number")
    activities: List[str] = Field(..., description="Activities for the day")
    meals: Optional[List[str]] = Field(default_factory=list, description="Meals included")
    description: str = Field(..., description="Description of the day's activities")

class BaseTour(BaseModel):
    """Base tour package model"""
    id: str = Field(..., description="Unique identifier for the tour package")
    title: str = Field(..., description="Title of the tour package")
    description: str = Field(..., description="Description of the tour package")
    duration_days: int = Field(..., description="Duration of the tour in days")
    price: float = Field(..., description="Price of the tour")
    location: str = Field(..., description="Location of the tour")
    highlights: List[str] = Field(default_factory=list, description="Highlights of the tour")
    # Add any other fields that are in your base tour packages
    accommodation_type: Optional[str] = Field(None, description="Type of accommodation provided")
    activities: Optional[List[str]] = Field(default_factory=list, description="Activities included in the tour")
    transportation: Optional[str] = Field(None, description="Transportation details")

class TourSuggestion(BaseModel):
    """Output model for suggested combined tour."""
    title: str = Field(..., description="Catchy title for the combined tour package")
    description: str = Field(..., description="Overall description of the combined tour")
    total_duration: int = Field(..., description="Total duration of the combined tour in days")
    pricing: PricingDetails = Field(..., description="Pricing breakdown")
    accommodation: List[AccommodationOption] = Field(..., description="List of accommodation options")
    transportation: TransportationDetails = Field(..., description="Transportation details")
    itinerary: List[ItineraryDay] = Field(..., description="Day-by-day itinerary")
    highlights: List[str] = Field(..., description="Key highlights of the combined tour")

class TourSuggestionResponse(BaseModel):
    """Response model for the tour suggestion endpoint."""
    suggested_tour: TourSuggestion = Field(..., description="The suggested combined tour package")
    base_tours_used: List[BaseTour] = Field(default_factory=list, description="Base tours used in the suggestion") 