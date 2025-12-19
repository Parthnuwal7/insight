"""
Frontend helper utilities for integrating with backend MongoDB/Redis features.
Use these functions in your Streamlit app to:
- Log telemetry events
- Handle rate limiting
- Submit async jobs
- Track job status
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


def submit_analysis_job(
    api_url: str,
    data: list,
    options: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Submit ABSA analysis job to async queue.
    
    Args:
        api_url: Backend API base URL
        data: Review data (list of dicts)
        options: Optional processing options
    
    Returns:
        Job ID if successful, None otherwise
    """
    try:
        device_id = get_or_create_device_id()
        user_id = get_user_id()
        
        if options is None:
            options = {}
        
        # Add device_id to options for tracking
        options["device_id"] = device_id
        
        payload = {
            "data": data,
            "options": options,
            "user_id": user_id or "anonymous"
        }
        
        response = requests.post(
            f"{api_url}/submit-job",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 429:
            # Rate limit hit
            st.error("⚠️ Rate limit exceeded. Please wait a minute before trying again.")
            return None
        
        response.raise_for_status()
        
        data = response.json()
        job_id = data.get("job_id")
        
        return job_id
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            error_data = e.response.json()
            st.error(f"⚠️ {error_data.get('detail', {}).get('message', 'Rate limit exceeded')}")
        else:
            st.error(f"Failed to submit job: {str(e)}")
        return None
    
    except Exception as e:
        st.error(f"Failed to submit job: {str(e)}")
        logger.error(f"Job submission error: {str(e)}")
        return None


def get_job_status(api_url: str, job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get status of submitted job.
    
    Args:
        api_url: Backend API base URL
        job_id: Job identifier
    
    Returns:
        Job status dict or None
    """
    try:
        response = requests.get(
            f"{api_url}/job-status/{job_id}",
            timeout=5
        )
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        return None


def poll_job_until_complete(
    api_url: str,
    job_id: str,
    progress_callback=None,
    max_wait_seconds: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Poll job status until complete.
    
    Args:
        api_url: Backend API base URL
        job_id: Job identifier
        progress_callback: Optional callback function for progress updates
        max_wait_seconds: Maximum time to wait
    
    Returns:
        Job result if completed, None otherwise
    """
    import time
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > max_wait_seconds:
            if progress_callback:
                progress_callback("Timeout waiting for job completion")
            return None
        
        status_data = get_job_status(api_url, job_id)
        
        if not status_data:
            if progress_callback:
                progress_callback("Job not found")
            return None
        
        status = status_data.get("status")
        
        if progress_callback:
            progress_callback(f"Status: {status}")
        
        if status == "DONE":
            return status_data.get("result")
        
        elif status == "FAILED":
            if progress_callback:
                progress_callback("Job failed")
            return None
        
        # Wait before polling again
        time.sleep(2)


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
