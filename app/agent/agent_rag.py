from typing import Dict, List, Optional
import os
import pandas as pd

from llama_index.llms.openai import OpenAI as OpenAI_LLAMA
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import PromptTemplate
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool

from app.history.history_module import HistoryModule
from app.templates.prompt_templates import SYSTEM_TEMPLATE
from app.services.embeddings import EmbeddingService
from app.vectorstore.supabase_vectorstore import SupabaseVectorStore
from app.tools.date.date_tool import DateExtractionTool
from app.tools.organization.organization_tool import OrganizationValidationTool
from app.tools.search.search_tools import SearchMeetingsTool, SearchMeetingsByOrganizationTool, SearchTravelPackagesTool
from app.config.env_config import config


class AgentRag:
    """
    RAG (Retrieval-Augmented Generation) agent for meeting queries.
    This agent integrates multiple tools and uses a vector store for document retrieval.
    """
    
    def __init__(self, history_module: HistoryModule):
        self.qa_template = PromptTemplate(SYSTEM_TEMPLATE)
        self.gpt4_llm = OpenAI_LLAMA(model=config.llm_model)
        self.vector_store = None
        self.agent = None
        self.embedding_service = EmbeddingService()
        self._load_organizations()
    
    def _load_organizations(self):
        """Load organizations from CSV file."""
        try:
            # df = None
            # # Get unique organization names, excluding empty values
            # self.organizations = df['Account'].dropna().unique().tolist()
            pass
        except Exception as e:
            # print(f"Error loading organizations: {e}")
            self.organizations = [
                '5 Elements Brewery', 'Absher Construction', 'Accel Scaling',
                'AFG VIETNAM', 'AI-Assisted Coaching and Mentoring Tools', 'Aim up Vietnam', 
                'Alchemy', 'Alchemy Asia', 'Aquila', 'Avision Young', 'Avison Young', 
                'Brooks AI', 'Brooks.ai', 'Caram Gems', 
                'CASH Financial Services Group Limited', 'CGM', 'Chikita Restaurant', 
                'Common Metal', 'Compass Events Pte Ltd', 'Dao Nguyen Legal', 
                'Delight Labs PR', 'Design X', 'DFDL Lawfirm', 'Digital Trends Media Group', 
                'Doxa Talent', 'Edge8', 'Eric Enriquez', 'Fairview International School', 
                'GRADY GOLF', 'Grit Volleyball', 'Hermes Landscaping', 'Hit Lights LED', 
                'HITlights', 'IFP Partners Limited', 'Ikaria Group', 'Invest Migrate', 
                'IPPG Vietnam', 'Kation', 'Kyungbang Vietnam', 'MomentsWare', 
                'On Target by Abound Health', 'Oseran Hahn', 'Pho 24', 'Power of 3', 
                'Qualicious', 'Rock Hill Asia', 'Single Grain', 'Socket', 'Sound Acoustic', 
                'Studio 3', 'Studio3eight', 'Surrogate First', 'TAL Apparel', 'Tartine Saigon', 
                'The Icarus Institute', 'The Problem Solver', 'Unlock Venture Partners', 
                'Veracity ', 'Vespa Adventures', 'Vietrose Internatinal', 
                'Vietrose International Vietnam', 'Vulcan Lab', 'Wareease', 'West Coast Dental', 
                'Westcoast International Dental Clinic', 'Wink Hotel Group', 
                'Work Healthy Australia', 'YPO Gold Forum', 'Grady Golf'
            ]
    
    def setup_agent(self, auth: str):
        """
        Set up the agent with necessary tools and configurations.
        
        Args:
            auth: Authentication token for Supabase.
        """
        # Initialize vector store
        self.vector_store = SupabaseVectorStore(
            url=config.supabase_url,
            key=config.supabase_anon_key,
            auth=auth
        )
        
        search_travel_tool = SearchTravelPackagesTool(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service
        )
        
        # Define a wrapper function to return string output for the agent
        def _search_travel_packages_agent_wrapper(
                location_input: str,
                duration_input: str,
                budget_input: str,
                transportation_input: str,
                accommodation_input: str,
                food_input: str,
                activities_input: str,
                notes_input: str,
                match_count: int = 10
                ) -> str:
            """Internal wrapper to format SearchTravelPackagesTool output as a string for the agent.
            Searches for relevant travel packages based on multiple criteria.
            Args:
                location_input (str): Location preferences or destination.
                duration_input (str): Duration preferences.
                budget_input (str): Budget preferences.
                transportation_input (str): Transportation preferences.
                accommodation_input (str): Accommodation preferences.
                food_input (str): Food preferences.
                activities_input (str): Activities preferences.
                notes_input (str): Additional notes or preferences.
                match_count (int): Number of results to return (default 10).
            """
            
            # Call the original tool to get the list of dictionaries
            results_list = search_travel_tool(
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
            
            # Format the list of dictionaries into a string
            if not results_list:
                return "No travel packages found matching your preferences."
            
            formatted_results = []
            for idx, package in enumerate(results_list):
                package_lines = [f"Package {idx+1}:"]
                for key, value in package.items():
                    # Exclude combined_score from the string output
                    if key != 'combined_score':
                        package_lines.append(f"  {key}: {value}")
                formatted_results.append("\n".join(package_lines))
            
            return "\n\n".join(formatted_results)

        # Create the FunctionTool using the wrapper function
        search_travel_function_tool = FunctionTool.from_defaults(
            name=search_travel_tool.name,
            description=search_travel_tool.description,
            fn=_search_travel_packages_agent_wrapper # Use the wrapper function
        )
        
        # Set up memory with configurable token limit
        memory = ChatMemoryBuffer.from_defaults(token_limit=config.history_token_limit)
        
        # Initialize agent with tools
        self.agent = OpenAIAgent.from_tools(
            tools=[
                search_travel_function_tool
            ],
            llm=self.gpt4_llm,
            memory=memory,
            verbose=True,
            system_prompt=SYSTEM_TEMPLATE
        )
    
    def agent_query(self, query: str) -> str:
        """
        Query the agent with a user question.
        
        Args:
            query: The user's question.
            
        Returns:
            The agent's response.
        """
        response = self.agent.chat(query)
        print("Chat history:", self.agent.chat_history)
        return response 