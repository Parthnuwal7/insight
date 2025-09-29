"""
Utility functions for data management, history saving, and test data generation.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import pickle
import streamlit as st
from faker import Faker
import random

fake = Faker()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle date objects and other non-serializable types."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class DataManager:
    """Manages data storage, loading, and history tracking."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self.uploads_path = os.path.join(base_path, "uploads")
        self.processed_path = os.path.join(base_path, "processed")
        self.history_path = os.path.join(base_path, "history")
        
        # Create directories if they don't exist
        for path in [self.uploads_path, self.processed_path, self.history_path]:
            os.makedirs(path, exist_ok=True)
    
    def save_processed_data(self, data: Dict[str, Any], session_id: str) -> str:
        """
        Save processed data and return file path.
        
        Args:
            data: Processed data dictionary
            session_id: Unique session identifier
            
        Returns:
            File path where data was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{session_id}_{timestamp}.pkl"
        filepath = os.path.join(self.processed_path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    def load_processed_data(self, filepath: str) -> Dict[str, Any]:
        """Load processed data from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_dashboard_state(self, session_id: str, dashboard_config: Dict[str, Any]) -> str:
        """
        Save dashboard configuration to history.
        
        Args:
            session_id: Unique session identifier
            dashboard_config: Dashboard configuration and metadata
            
        Returns:
            History entry ID
        """
        timestamp = datetime.now()
        history_entry = {
            'id': f"{session_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'session_id': session_id,
            'timestamp': timestamp.isoformat(),
            'config': dashboard_config,
            'metadata': {
                'total_reviews': dashboard_config.get('total_reviews', 0),
                'data_file': dashboard_config.get('data_file', ''),
                'filters_applied': dashboard_config.get('filters_applied', {}),
                'charts_generated': dashboard_config.get('charts_generated', [])
            }
        }
        
        # Save to history file
        history_file = os.path.join(self.history_path, f"history_{history_entry['id']}.json")
        with open(history_file, 'w') as f:
            json.dump(history_entry, f, indent=2, cls=CustomJSONEncoder)
        
        return history_entry['id']
    
    def get_history_list(self) -> List[Dict[str, Any]]:
        """Get list of all saved dashboard histories."""
        history_files = [f for f in os.listdir(self.history_path) if f.endswith('.json')]
        history_list = []
        
        for filename in history_files:
            filepath = os.path.join(self.history_path, filename)
            try:
                with open(filepath, 'r') as f:
                    history_entry = json.load(f)
                    history_list.append({
                        'id': history_entry['id'],
                        'timestamp': history_entry['timestamp'],
                        'total_reviews': history_entry['metadata']['total_reviews'],
                        'data_file': history_entry['metadata']['data_file']
                    })
            except Exception as e:
                st.error(f"Error loading history file {filename}: {str(e)}")
        
        # Sort by timestamp descending
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return history_list
    
    def load_dashboard_history(self, history_id: str) -> Optional[Dict[str, Any]]:
        """Load specific dashboard history."""
        history_file = os.path.join(self.history_path, f"history_{history_id}.json")
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return None
    
    def delete_history_entry(self, history_id: str) -> bool:
        """Delete a history entry."""
        history_file = os.path.join(self.history_path, f"history_{history_id}.json")
        
        if os.path.exists(history_file):
            os.remove(history_file)
            return True
        return False
    
    def cleanup_old_files(self, days_threshold: int = 30):
        """Clean up files older than threshold."""
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        for directory in [self.uploads_path, self.processed_path, self.history_path]:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if file_time < threshold_date:
                    os.remove(filepath)


class TestDataGenerator:
    """Generates synthetic test data for development and testing."""
    
    @staticmethod
    def generate_sample_dataset(num_reviews: int = 100) -> pd.DataFrame:
        """
        Generate sample review dataset.
        
        Args:
            num_reviews: Number of reviews to generate
            
        Returns:
            DataFrame with sample review data
        """
        # Sample product aspects and related keywords
        aspects = ['battery', 'screen', 'camera', 'performance', 'design', 'price', 'durability', 'sound', 'storage']
        
        positive_words = ['excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'perfect', 'outstanding', 'brilliant']
        negative_words = ['terrible', 'awful', 'horrible', 'disappointing', 'bad', 'worst', 'useless', 'pathetic']
        neutral_words = ['okay', 'average', 'normal', 'standard', 'fine', 'decent', 'acceptable']
        
        reviews = []
        
        for i in range(num_reviews):
            # Generate random sentiment
            sentiment = random.choice(['positive', 'negative', 'neutral'])
            
            # Select random aspects (1-3 aspects per review)
            selected_aspects = random.sample(aspects, random.randint(1, 3))
            
            # Generate review text based on sentiment
            review_parts = []
            
            for aspect in selected_aspects:
                if sentiment == 'positive':
                    word = random.choice(positive_words)
                    review_parts.append(f"The {aspect} is {word}")
                elif sentiment == 'negative':
                    word = random.choice(negative_words)
                    review_parts.append(f"The {aspect} is {word}")
                else:
                    word = random.choice(neutral_words)
                    review_parts.append(f"The {aspect} is {word}")
            
            review_text = ". ".join(review_parts) + "."
            
            # Occasionally add Hindi text (10% chance)
            if random.random() < 0.1:
                hindi_phrases = [
                    "यह बहुत अच्छा है", "मुझे यह पसंद नहीं आया", "ठीक है", 
                    "बहुत खराब", "उत्कृष्ट गुणवत्ता", "औसत प्रदर्शन"
                ]
                review_text += " " + random.choice(hindi_phrases)
            
            reviews.append({
                'id': f"review_{i+1:04d}",
                'reviews_title': fake.catch_phrase(),
                'review': review_text,
                'date': fake.date_between(start_date='-1y', end_date='today'),
                'user_id': f"user_{random.randint(1000, 9999)}"
            })
        
        return pd.DataFrame(reviews)
    
    @staticmethod
    def generate_complex_reviews(num_reviews: int = 50) -> pd.DataFrame:
        """Generate more complex, realistic reviews."""
        
        product_types = ['smartphone', 'laptop', 'headphones', 'smartwatch', 'tablet']
        
        review_templates = {
            'positive': [
                "I absolutely love this {product}! The {aspect1} is {positive_word1} and the {aspect2} is {positive_word2}. Highly recommend!",
                "This {product} exceeded my expectations. The {aspect1} quality is {positive_word1}. Best purchase I've made!",
                "Amazing {product}! The {aspect1} and {aspect2} work perfectly together. {positive_word1} value for money."
            ],
            'negative': [
                "Very disappointed with this {product}. The {aspect1} is {negative_word1} and {aspect2} is {negative_word2}. Would not recommend.",
                "This {product} is a complete waste of money. {aspect1} stopped working after a week. {negative_word1} quality.",
                "Terrible {product}. The {aspect1} is {negative_word1} and customer service is {negative_word2}."
            ],
            'neutral': [
                "This {product} is {neutral_word1}. The {aspect1} is decent but {aspect2} could be better. It's okay for the price.",
                "Average {product}. {aspect1} works fine, {aspect2} is {neutral_word1}. Nothing special but gets the job done.",
                "The {product} is {neutral_word1}. {aspect1} is good but {aspect2} needs improvement."
            ]
        }
        
        aspects = ['battery life', 'display', 'camera quality', 'performance', 'build quality', 'price', 'software', 'design']
        positive_words = ['excellent', 'amazing', 'outstanding', 'brilliant', 'superb', 'fantastic']
        negative_words = ['terrible', 'awful', 'disappointing', 'poor', 'horrible', 'defective']
        neutral_words = ['okay', 'average', 'decent', 'acceptable', 'standard', 'fine']
        
        reviews = []
        
        for i in range(num_reviews):
            sentiment = random.choice(['positive', 'negative', 'neutral'])
            product = random.choice(product_types)
            template = random.choice(review_templates[sentiment])
            
            # Fill template
            review_text = template.format(
                product=product,
                aspect1=random.choice(aspects),
                aspect2=random.choice(aspects),
                positive_word1=random.choice(positive_words),
                positive_word2=random.choice(positive_words),
                negative_word1=random.choice(negative_words),
                negative_word2=random.choice(negative_words),
                neutral_word1=random.choice(neutral_words)
            )
            
            reviews.append({
                'id': f"complex_review_{i+1:04d}",
                'reviews_title': f"{product.title()} Review",
                'review': review_text,
                'date': fake.date_between(start_date='-6m', end_date='today'),
                'user_id': f"user_{random.randint(10000, 99999)}"
            })
        
        return pd.DataFrame(reviews)


class SessionManager:
    """Manages user sessions and state."""
    
    def __init__(self):
        # Initialize session state attributes if they don't exist
        if 'session_id' not in st.session_state:
            st.session_state.session_id = self._generate_session_id()
        
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        if 'current_filters' not in st.session_state:
            st.session_state.current_filters = {}
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    
    def get_session_id(self) -> str:
        """Get current session ID."""
        return st.session_state.session_id
    
    def reset_session(self):
        """Reset current session."""
        st.session_state.session_id = self._generate_session_id()
        st.session_state.processed_data = None
        st.session_state.current_filters = {}
    
    def set_processed_data(self, data: Dict[str, Any]):
        """Set processed data in session."""
        st.session_state.processed_data = data
    
    def get_processed_data(self) -> Optional[Dict[str, Any]]:
        """Get processed data from session."""
        # Ensure the attribute exists before accessing
        if hasattr(st.session_state, 'processed_data'):
            return st.session_state.processed_data
        else:
            # Initialize if it doesn't exist
            st.session_state.processed_data = None
            return None
    
    def set_filters(self, filters: Dict[str, Any]):
        """Set current filters."""
        st.session_state.current_filters = filters
    
    def get_filters(self) -> Dict[str, Any]:
        """Get current filters."""
        # Ensure the attribute exists before accessing
        if hasattr(st.session_state, 'current_filters'):
            return st.session_state.current_filters
        else:
            # Initialize if it doesn't exist
            st.session_state.current_filters = {}
            return {}


class ConfigManager:
    """Manages application configuration."""
    
    DEFAULT_CONFIG = {
        'app_title': 'Sentiment Analysis Dashboard',
        'max_file_size_mb': 100,
        'supported_file_types': ['csv'],
        'default_chart_height': 400,
        'items_per_page': 20,
        'auto_save_interval': 300,  # seconds
        'cache_timeout': 3600,  # seconds
        'theme': 'light'
    }
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                merged_config = self.DEFAULT_CONFIG.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                st.warning(f"Error loading config: {e}. Using defaults.")
        
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2, cls=CustomJSONEncoder)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)