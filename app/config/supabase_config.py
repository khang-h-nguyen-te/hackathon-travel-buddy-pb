import os
from supabase import create_client, Client
from supabase.client import ClientOptions

from app.config.env_config import config


def get_supabase_credentials():
    """Get Supabase credentials from environment variables."""
    return {
        "url": config.supabase_url,
        "key": config.supabase_anon_key
    }


def get_supabase_client(auth_header=None):
    """
    Create a Supabase client instance.
    
    Args:
        auth_header (str, optional): Auth header to include in requests. 
            If provided, it will be added to global headers.
    
    Returns:
        Client: A Supabase client instance.
    """
    credentials = get_supabase_credentials()
    
    if auth_header:
        return create_client(
            credentials["url"], 
            credentials["key"], 
            options=ClientOptions(
                headers={"Authorization": auth_header},
                # schema="dummy_schema",
            )
        )
    
    return create_client(
        credentials["url"], 
        credentials["key"],
        # options=ClientOptions(
        #     schema="dummy_schema",
        # )
    ) 