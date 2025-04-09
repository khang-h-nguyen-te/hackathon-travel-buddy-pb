# Meeting Chatbot Backend

A FastAPI application that serves as a chatbot backend for querying historical meeting data stored in Supabase.

## Project Structure

```
hackathon-travel-buddy/
├── main.py  # Main application entry point
├── app/
│   ├── agent/
│   │   └── agent_rag.py  # RAG agent implementation
│   ├── config/
│   │   └── supabase_config.py  # Supabase configuration
│   ├── history/
│   │   └── history_module.py  # Chat history management
│   ├── models/
│   │   └── request_models.py  # API request/response models
│   ├── services/
│   │   └── embeddings.py  # Embedding service
│   ├── templates/
│   │   └── prompt_templates.py  # System prompts
│   ├── tools/
│   │   ├── base_tool.py  # Abstract base class for tools
│   │   ├── tool_registry.py  # Tool registry/factory
│   │   ├── date/
│   │   │   └── date_tool.py  # Date extraction tools
│   │   ├── organization/
│   │   │   └── organization_tool.py  # Organization validation
│   │   └── search/
│   │       └── search_tools.py  # Meeting search tools
│   ├── utils/
│   │   └── response_utils.py  # API response utilities
│   └── vectorstore/
│       └── supabase_vectorstore.py  # Vector store for meetings
└── requirements.txt
```

## Key Components

- **Agent**: The main RAG (Retrieval-Augmented Generation) agent that orchestrates the tools and handles user queries.
- **Tools**: Modular, pluggable components that provide specific functionality like date extraction, organization validation, and search.
- **Tool Registry**: A manager for all available tools, allowing tools to be registered, categorized, and retrieved.
- **Vector Store**: Interface to Supabase for searching meeting data using vector embeddings.
- **History Module**: Manages conversation history for context-aware responses.

## Setup and Usage

1. Copy `.env.example` to `.env` and fill in the required variables
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

The API will be available at `http://localhost:8000`.

## API Endpoints

- **POST /authenticate/{command}**: Sign in and initialize the agent
- **POST /ask**: Send a question to the agent
- **GET /health**: Simple health check endpoint

## Extending the Tool System

To add a new tool:

1. Create a new tool class that extends `BaseTool`
2. Implement the `__call__` method to execute the tool's functionality
3. Register the tool in `agent_rag.py`

Example:

```python
from app.tools.base_tool import BaseTool

class MyNewTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="MyNewTool",
            description="Description of what my tool does"
        )
    
    def __call__(self, param1, param2) -> str:
        # Implement your tool logic here
        return "Tool result"
``` 