"""
Frontend helper utilities for integrating with backend MongoDB telemetry features.
Use these functions in your Streamlit app to:
- Log telemetry events
"""

import streamlit as st
import requests
import uuid
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_or_create_device_id() -> str:
    """
    Get or create device ID for current session.
    Stored in Streamlit session state.
    """
    if "device_id" not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())
    
    return st.session_state.device_id


def get_user_id() -> Optional[str]:
    """
    Get user ID if authenticated.
    Returns None if user is not logged in.
    """
    return st.session_state.get("user_id")


def log_session_metadata(
    api_url: str,
    ip_address: str = "127.0.0.1",
    user_agent: Optional[str] = None
) -> bool:
    """
    Log session metadata (IP, location) with Redis gating.
    
    Args:
        api_url: Backend API base URL
        ip_address: Client IP address
        user_agent: User agent string
    
    Returns:
        True if logged, False if skipped or failed
    """
    try:
        device_id = get_or_create_device_id()
        user_id = get_user_id()
        
        payload = {
            "device_id": device_id,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        response = requests.post(
            f"{api_url}/log-session",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("logged", False)
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to log session metadata: {str(e)}")
        return False


def log_event(
    api_url: str,
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Log a telemetry event to MongoDB.
    
    Args:
        api_url: Backend API base URL
        event_type: Event type (DASHBOARD_VIEW, ANALYSIS_REQUEST, etc.)
        metadata: Optional event metadata
    
    Returns:
        True if logged successfully
    """
    try:
        device_id = get_or_create_device_id()
        user_id = get_user_id()
        
        payload = {
            "event_type": event_type,
            "device_id": device_id,
            "user_id": user_id,
            "metadata": metadata
        }
        
        response = requests.post(
            f"{api_url}/log-event",
            json=payload,
            timeout=5
        )
        
        return response.status_code == 200
        
    except Exception as e:
        logger.error(f"Failed to log event: {str(e)}")
        return False


def initialize_telemetry(api_url: str):
    """
    Initialize telemetry on app load.
    Call this once at the start of your Streamlit app.
    
    Args:
        api_url: Backend API base URL
    """
    # Log session metadata (gated by Redis)
    log_session_metadata(api_url)
    
    # Log dashboard view event
    log_event(api_url, "DASHBOARD_VIEW")
