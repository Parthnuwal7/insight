"""
Frontend helper utilities for the Streamlit app.
"""

import streamlit as st
import uuid
from typing import Optional


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
