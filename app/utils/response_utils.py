from typing import Any, Dict, List


def create_response(message: Any, status_code: int, error: bool = False) -> Dict[str, Any]:
    """
    Create a standardized response object.
    
    Args:
        message: The response message or data.
        status_code: The HTTP status code.
        error: Whether this response represents an error.
        
    Returns:
        Dictionary with standard response format.
    """
    return {
        "message": message,
        "status_code": status_code,
        "error": error
    }


def validate_params(params: Dict[str, Any], required_params: List[str]) -> bool:
    """
    Validate that all required parameters are present in the params dictionary.
    
    Args:
        params: Dictionary of parameters to validate.
        required_params: List of parameter names that are required.
        
    Returns:
        True if all required parameters are present, False otherwise.
    """
    for param in required_params:
        if param not in params:
            return False
    return True 