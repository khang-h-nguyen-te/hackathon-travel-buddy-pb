from typing import List, Optional
from pydantic import BaseModel, Field
import json

from app.tools.base_tool import BaseTool
from app.config.env_config import config
from openai import OpenAI


class OrganizationValidation(BaseModel):
    """Pydantic model for organization validation output."""
    organization_name: Optional[str] = Field(
        description="The exact organization name if found, None if not found"
    )


class OrganizationValidationTool(BaseTool):
    """Tool for validating organization names against a known list."""
    
    def __init__(self, organizations: List[str]):
        super().__init__(
            name="ValidateOrganization",
            description="Validates and extracts the correct organization name from the input. "
                       "Returns the exact organization name if found, or None if not found."
        )
        self.organizations = organizations
        self.openai_client = OpenAI(api_key=config.openai_api_key)
    
    def __call__(self, organization_input: str) -> OrganizationValidation:
        """
        Validate and extract the correct organization name from the input.
        Returns a Pydantic model with the organization name.
        """
        if not self.organizations:
            return OrganizationValidation(organization_name=None)

        try:
            completion = self.openai_client.chat.completions.create(
                model=config.llm_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an organization validator. Given a list of valid organizations, find the most similar match for the input.
                        Valid organizations: {', '.join(self.organizations)}
                        
                        Rules:
                        1. Return ONLY the most similar organization name from the list
                        2. If no good match is found, return null
                        3. Do not add any explanation or additional text
                        
                        RETURN AS JSON in the format:
                        {{"organization_name": "matched_org_name_or_null"}}
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Find the matching organization for: {organization_input}"
                    }
                ]
            )

            # Extract and parse the JSON response
            response_content = completion.choices[0].message.content
            try:
                response_json = json.loads(response_content)
                return OrganizationValidation(organization_name=response_json.get("organization_name"))
            except (json.JSONDecodeError, AttributeError):
                # Fallback in case of json parsing errors
                return OrganizationValidation(organization_name=None)

        except Exception as e:
            return OrganizationValidation(organization_name=None) 