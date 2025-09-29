"""
Enhanced data processing pipeline for advanced sentiment analysis application.
Handles translation, ABSA, intent classification, priority scoring, and co-occurrence analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from datetime import datetime, timedelta
import logging
from langdetect import detect
import streamlit as st
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates uploaded CSV data format and content."""
    
    REQUIRED_COLUMNS = ['id', 'reviews_title', 'review', 'date', 'user_id']
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate CSV format and content.
        
        Args:
            df: Uploaded dataframe
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(DataValidator.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if not errors:
            # Check for empty values
            if df['review'].isnull().any() or (df['review'] == '').any():
                errors.append("Found empty review entries")
            
            # Validate date format
            try:
                pd.to_datetime(df['date'], errors='coerce')
                if df['date'].isnull().any():
                    errors.append("Invalid date format detected")
            except Exception as e:
                errors.append(f"Date validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        df = df.copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean text data
        df['review'] = df['review'].astype(str).str.strip()
        df['reviews_title'] = df['reviews_title'].astype(str).str.strip()
        
        # Remove rows with null reviews
        df = df.dropna(subset=['review'])
        
        # Remove duplicate reviews
        df = df.drop_duplicates(subset=['review'], keep='first')
        
        return df.reset_index(drop=True)


class TranslationService:
    """Handles translation from Hindi to English using M2M100."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """Load M2M100 model for translation."""
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            
            model_name = "facebook/m2m100_418M"
            _self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            _self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
            st.error(f"Failed to load translation model: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            lang = detect(text)
            return lang
        except:
            return 'unknown'
    
    def translate_to_english(self, text: str, source_lang: str = 'hi') -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Translated text
        """
        if not self.model or not self.tokenizer:
            return text
        
        try:
            # Set source language
            self.tokenizer.src_lang = source_lang
            
            # Encode and translate
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate translation
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id("en"),
                max_length=512,
                num_beams=2,
                early_stopping=True
            )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text
    
    def process_reviews(self, reviews: List[str]) -> Tuple[List[str], List[str]]:
        """
        Process list of reviews for translation.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Tuple of (translated_reviews, detected_languages)
        """
        translated_reviews = []
        detected_languages = []
        
        for review in reviews:
            lang = self.detect_language(review)
            detected_languages.append(lang)
            
            if lang == 'hi':  # Hindi detected
                translated = self.translate_to_english(review, 'hi')
                translated_reviews.append(translated)
            else:
                translated_reviews.append(review)  # Keep original if not Hindi
        
        return translated_reviews, detected_languages


class ABSAProcessor:
    """Handles Aspect-Based Sentiment Analysis using pyABSA."""
    
    def __init__(self):
        self.aspect_extractor = None
        self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """Load pyABSA model."""
        try:
            from pyabsa import ATEPCCheckpointManager
            
            _self.aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
                checkpoint='multilingual',
                auto_device=True,
                task_code='ATEPC'
            )
            
            logger.info("ABSA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ABSA model: {str(e)}")
            st.error(f"Failed to load ABSA model: {str(e)}")
    
    def extract_aspects_and_sentiments(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """
        Extract aspects and sentiments from reviews.
        
        Args:
            reviews: List of review texts
            
        Returns:
            List of dictionaries containing extracted information
        """
        if not self.aspect_extractor:
            return []
        
        try:
            results = self.aspect_extractor.extract_aspect(
                reviews,
                pred_sentiment=True
            )
            
            processed_results = []
            for result in results:
                processed_result = {
                    'sentence': result['sentence'],
                    'aspects': result.get('aspect', []),
                    'sentiments': result.get('sentiment', []),
                    'positions': result.get('position', []),
                    'confidence_scores': result.get('confidence', []),
                    'tokens': result.get('tokens', []),
                    'iob_tags': result.get('IOB', [])
                }
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"ABSA processing error: {str(e)}")
            return []


class IntentClassifier:
    """Enhanced intent classifier with severity scoring for complaints and type classification."""
    
    INTENT_KEYWORDS = {
        'complaint': {
            'high_severity': ['terrible', 'worst', 'horrible', 'awful', 'hate', 'useless', 'trash', 'garbage', 'waste', 'pathetic'],
            'medium_severity': ['bad', 'disappointed', 'frustrated', 'annoying', 'poor', 'defective', 'broken', 'failed'],
            'low_severity': ['problem', 'issue', 'concern', 'slow', 'okay but', 'could be better']
        },
        'praise': {
            'high_positive': ['excellent', 'amazing', 'fantastic', 'wonderful', 'perfect', 'outstanding', 'brilliant', 'superb'],
            'medium_positive': ['good', 'great', 'love', 'best', 'awesome', 'nice', 'satisfied', 'happy'],
            'low_positive': ['fine', 'decent', 'acceptable', 'adequate', 'reasonable']
        },
        'question': ['how', 'what', 'when', 'where', 'why', 'which', 'who', '?', 'can you', 'could you', 'is it possible'],
        'suggestion': ['should', 'could', 'would', 'recommend', 'suggest', 'improve', 'better', 'enhancement', 'feature request'],
        'comparison': ['better than', 'worse than', 'compared to', 'versus', 'vs', 'similar to', 'different from'],
        'neutral': ['okay', 'fine', 'average', 'normal', 'standard', 'typical', 'nothing special']
    }
    
    @classmethod
    def classify_intent_with_severity(cls, review: str) -> Tuple[str, str, float]:
        """
        Classify intent with severity/type scoring.
        
        Args:
            review: Review text
            
        Returns:
            Tuple of (intent, severity/type, confidence_score)
        """
        review_lower = review.lower()
        
        # Check for complaints with severity
        complaint_scores = {}
        for severity, keywords in cls.INTENT_KEYWORDS['complaint'].items():
            score = sum(1 for keyword in keywords if keyword in review_lower)
            if score > 0:
                complaint_scores[severity] = score
        
        if complaint_scores:
            severity = max(complaint_scores, key=complaint_scores.get)
            confidence = min(complaint_scores[severity] / 3.0, 1.0)  # Normalize confidence
            return 'complaint', severity, confidence
        
        # Check for praise with positivity level
        praise_scores = {}
        for positivity, keywords in cls.INTENT_KEYWORDS['praise'].items():
            score = sum(1 for keyword in keywords if keyword in review_lower)
            if score > 0:
                praise_scores[positivity] = score
        
        if praise_scores:
            positivity = max(praise_scores, key=praise_scores.get)
            confidence = min(praise_scores[positivity] / 3.0, 1.0)
            return 'praise', positivity, confidence
        
        # Check other intents
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            if intent not in ['complaint', 'praise']:
                score = sum(1 for keyword in keywords if keyword in review_lower)
                if score > 0:
                    confidence = min(score / 2.0, 1.0)
                    return intent, 'standard', confidence
        
        return 'neutral', 'standard', 0.1
    
    @classmethod
    def classify_batch_enhanced(cls, reviews: List[str]) -> List[Dict[str, Any]]:
        """Classify intents with enhanced information for a batch of reviews."""
        results = []
        for review in reviews:
            intent, severity_type, confidence = cls.classify_intent_with_severity(review)
            results.append({
                'intent': intent,
                'severity_type': severity_type,
                'confidence': confidence
            })
        return results


class AspectAnalytics:
    """Advanced analytics for aspect analysis including priority scoring and co-occurrence."""
    
    @staticmethod
    def calculate_aspect_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate priority scores for negative aspects and strength scores for positive aspects.
        
        Args:
            df: Processed dataframe with aspects and sentiments
            
        Returns:
            Tuple of (areas_of_improvement_df, strength_anchors_df)
        """
        aspect_data = []
        
        # Extract aspect-sentiment pairs with intent information
        for idx, row in df.iterrows():
            aspects = row['aspects'] if isinstance(row['aspects'], list) else []
            sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
            intent = row['intent']
            intent_severity = row.get('intent_severity', 'standard')
            date = row['date']
            
            for aspect, sentiment in zip(aspects, sentiments):
                aspect_data.append({
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'intent': intent,
                    'intent_severity': intent_severity,
                    'date': date,
                    'review_id': row['id']
                })
        
        if not aspect_data:
            empty_df = pd.DataFrame(columns=['aspect', 'score', 'frequency', 'sentiment_ratio'])
            return empty_df, empty_df
        
        aspect_df = pd.DataFrame(aspect_data)
        
        # Calculate metrics for each aspect
        aspect_metrics = {}
        
        for aspect in aspect_df['aspect'].unique():
            aspect_subset = aspect_df[aspect_df['aspect'] == aspect]
            
            total_count = len(aspect_subset)
            positive_count = len(aspect_subset[aspect_subset['sentiment'] == 'Positive'])
            negative_count = len(aspect_subset[aspect_subset['sentiment'] == 'Negative'])
            
            # Calculate sentiment ratios
            positivity_ratio = positive_count / total_count if total_count > 0 else 0
            negativity_ratio = negative_count / total_count if total_count > 0 else 0
            
            # Calculate intent severity weighting
            complaint_subset = aspect_subset[aspect_subset['intent'] == 'complaint']
            severity_weight = 0
            if len(complaint_subset) > 0:
                severity_mapping = {'high_severity': 3, 'medium_severity': 2, 'low_severity': 1, 'standard': 1}
                severity_weight = complaint_subset['intent_severity'].map(severity_mapping).mean()
            
            # Calculate priority score for negative aspects
            priority_score = negativity_ratio * total_count * (1 + severity_weight)
            
            # Calculate strength score for positive aspects  
            strength_score = positivity_ratio * total_count * (1 + (positivity_ratio * 2))
            
            aspect_metrics[aspect] = {
                'frequency': total_count,
                'positivity_ratio': positivity_ratio,
                'negativity_ratio': negativity_ratio,
                'priority_score': priority_score,
                'strength_score': strength_score,
                'intent_severity': severity_weight
            }
        
        # Create Areas of Improvement DataFrame (top negative aspects)
        improvement_data = []
        for aspect, metrics in aspect_metrics.items():
            if metrics['negativity_ratio'] > 0.1:  # Only include aspects with >10% negativity
                improvement_data.append({
                    'aspect': aspect,
                    'negativity_pct': round(metrics['negativity_ratio'] * 100, 1),
                    'intent_severity': round(metrics['intent_severity'], 2),
                    'frequency': metrics['frequency'],
                    'priority_score': round(metrics['priority_score'], 2)
                })
        
        areas_of_improvement = pd.DataFrame(improvement_data).sort_values('priority_score', ascending=False)
        
        # Create Strength Anchors DataFrame (top positive aspects)
        strength_data = []
        for aspect, metrics in aspect_metrics.items():
            if metrics['positivity_ratio'] > 0.3:  # Only include aspects with >30% positivity
                strength_data.append({
                    'aspect': aspect,
                    'positivity_pct': round(metrics['positivity_ratio'] * 100, 1),
                    'intent_type': 'praise',  # Simplified for now
                    'frequency': metrics['frequency'],
                    'strength_score': round(metrics['strength_score'], 2)
                })
        
        strength_anchors = pd.DataFrame(strength_data).sort_values('strength_score', ascending=False)
        
        return areas_of_improvement, strength_anchors
    
    @staticmethod
    def calculate_aspect_cooccurrence(df: pd.DataFrame) -> nx.Graph:
        """
        Calculate aspect co-occurrence for network analysis.
        
        Args:
            df: Processed dataframe with aspects
            
        Returns:
            NetworkX graph with aspect co-occurrence data
        """
        G = nx.Graph()
        
        # Calculate co-occurrence matrix
        cooccurrence_counts = defaultdict(int)
        aspect_sentiments = defaultdict(list)
        aspect_frequencies = defaultdict(int)
        
        for idx, row in df.iterrows():
            aspects = row['aspects'] if isinstance(row['aspects'], list) else []
            sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
            
            # Count individual aspects
            for aspect, sentiment in zip(aspects, sentiments):
                aspect_frequencies[aspect] += 1
                aspect_sentiments[aspect].append(sentiment)
            
            # Count co-occurrences
            for aspect1, aspect2 in combinations(aspects, 2):
                pair = tuple(sorted([aspect1, aspect2]))
                cooccurrence_counts[pair] += 1
        
        # Add nodes with attributes
        for aspect, frequency in aspect_frequencies.items():
            sentiments = aspect_sentiments[aspect]
            positive_pct = sentiments.count('Positive') / len(sentiments) if sentiments else 0
            negative_pct = sentiments.count('Negative') / len(sentiments) if sentiments else 0
            
            # Determine overall sentiment color
            if positive_pct > negative_pct:
                color = 'green'
                sentiment_score = positive_pct
            elif negative_pct > positive_pct:
                color = 'red' 
                sentiment_score = -negative_pct
            else:
                color = 'gray'
                sentiment_score = 0
            
            G.add_node(aspect, 
                      frequency=frequency,
                      sentiment_score=sentiment_score,
                      color=color,
                      positive_pct=positive_pct,
                      negative_pct=negative_pct)
        
        # Add edges with weights
        for (aspect1, aspect2), count in cooccurrence_counts.items():
            if count >= 2:  # Only include co-occurrences that happen at least twice
                G.add_edge(aspect1, aspect2, weight=count)
        
        return G
    
    @staticmethod
    def detect_sentiment_spikes(df: pd.DataFrame, window_days: int = 7) -> List[Dict[str, Any]]:
        """
        Detect week-over-week spikes in negative sentiment for aspects.
        
        Args:
            df: Processed dataframe with date and aspect information
            window_days: Number of days for the rolling window
            
        Returns:
            List of alerts for aspects with significant negative spikes
        """
        alerts = []
        
        if len(df) < 2:
            return alerts
        
        # Extract aspect-sentiment data with dates
        aspect_data = []
        for idx, row in df.iterrows():
            aspects = row['aspects'] if isinstance(row['aspects'], list) else []
            sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
            date = row['date']
            
            for aspect, sentiment in zip(aspects, sentiments):
                aspect_data.append({
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'date': date
                })
        
        if not aspect_data:
            return alerts
        
        aspect_df = pd.DataFrame(aspect_data)
        aspect_df['date'] = pd.to_datetime(aspect_df['date'])
        
        # Group by aspect and analyze trends
        for aspect in aspect_df['aspect'].unique():
            aspect_subset = aspect_df[aspect_df['aspect'] == aspect]
            
            if len(aspect_subset) < 4:  # Need minimum data points
                continue
            
            # Create daily negative sentiment counts
            daily_negative = aspect_subset[aspect_subset['sentiment'] == 'Negative'].groupby(
                aspect_subset['date'].dt.date
            ).size().reindex(
                pd.date_range(aspect_subset['date'].min().date(), 
                             aspect_subset['date'].max().date()).date,
                fill_value=0
            )
            
            if len(daily_negative) >= window_days * 2:
                # Calculate rolling averages
                recent_avg = daily_negative.tail(window_days).mean()
                previous_avg = daily_negative.iloc[-(window_days*2):-window_days].mean()
                
                # Check for spike (>50% increase and at least 2 more complaints)
                if recent_avg > previous_avg * 1.5 and (recent_avg - previous_avg) >= 2:
                    spike_magnitude = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 100
                    
                    alerts.append({
                        'aspect': aspect,
                        'spike_magnitude': round(spike_magnitude, 1),
                        'recent_avg_negative': round(recent_avg, 1),
                        'previous_avg_negative': round(previous_avg, 1),
                        'alert_severity': 'high' if spike_magnitude > 100 else 'medium'
                    })
        
        return sorted(alerts, key=lambda x: x['spike_magnitude'], reverse=True)


class SummaryGenerator:
    """Generates macro and micro-level summaries for aspects and overall sentiment."""
    
    @staticmethod
    def generate_macro_summary(df: pd.DataFrame, areas_of_improvement: pd.DataFrame, 
                              strength_anchors: pd.DataFrame) -> Dict[str, str]:
        """
        Generate high-level summary of sentiment analysis.
        
        Args:
            df: Processed dataframe
            areas_of_improvement: Problem areas dataframe  
            strength_anchors: Strength areas dataframe
            
        Returns:
            Dictionary with macro-level insights
        """
        total_reviews = len(df)
        positive_pct = (df['overall_sentiment'] == 'Positive').mean() * 100
        negative_pct = (df['overall_sentiment'] == 'Negative').mean() * 100
        
        # Top issues and strengths
        top_issue = areas_of_improvement.iloc[0]['aspect'] if len(areas_of_improvement) > 0 else "None identified"
        top_strength = strength_anchors.iloc[0]['aspect'] if len(strength_anchors) > 0 else "None identified"
        
        # Intent distribution
        complaint_pct = (df['intent'] == 'complaint').mean() * 100
        
        summary = {
            'overall_sentiment': f"Out of {total_reviews} reviews, {positive_pct:.1f}% are positive and {negative_pct:.1f}% are negative.",
            'top_issues': f"Primary concern: '{top_issue}' - requires immediate attention based on complaint frequency and severity.",
            'top_strengths': f"Greatest strength: '{top_strength}' - leverage this positive aspect in marketing and product positioning.",
            'intent_insights': f"{complaint_pct:.1f}% of reviews contain complaints, indicating specific areas for product improvement.",
            'recommendation': "Focus on addressing top negative aspects while maintaining and promoting strengths."
        }
        
        return summary
    
    @staticmethod 
    def generate_aspect_micro_summaries(df: pd.DataFrame, top_aspects: List[str], 
                                       max_aspects: int = 5) -> Dict[str, str]:
        """
        Generate detailed summaries for specific aspects.
        
        Args:
            df: Processed dataframe
            top_aspects: List of aspects to analyze
            max_aspects: Maximum number of aspects to summarize
            
        Returns:
            Dictionary with aspect-specific insights
        """
        summaries = {}
        
        for aspect in top_aspects[:max_aspects]:
            # Filter reviews mentioning this aspect
            aspect_reviews = []
            aspect_sentiments = []
            
            for idx, row in df.iterrows():
                aspects = row['aspects'] if isinstance(row['aspects'], list) else []
                sentiments = row['aspect_sentiments'] if isinstance(row['aspect_sentiments'], list) else []
                
                if aspect in aspects:
                    aspect_idx = aspects.index(aspect)
                    if aspect_idx < len(sentiments):
                        aspect_reviews.append(row['translated_review'])
                        aspect_sentiments.append(sentiments[aspect_idx])
            
            if not aspect_reviews:
                continue
            
            positive_count = aspect_sentiments.count('Positive')
            negative_count = aspect_sentiments.count('Negative')
            total_count = len(aspect_sentiments)
            
            # Generate contextual summary
            if negative_count > positive_count:
                sentiment_trend = "predominantly negative"
                key_issues = "Issues include quality concerns, performance problems, and user dissatisfaction."
            elif positive_count > negative_count:
                sentiment_trend = "predominantly positive" 
                key_issues = "Users appreciate the quality, performance, and overall experience."
            else:
                sentiment_trend = "mixed"
                key_issues = "Reviews show both satisfaction and areas for improvement."
            
            summaries[aspect] = f"'{aspect}' mentioned in {total_count} reviews with {sentiment_trend} sentiment ({positive_count} positive, {negative_count} negative). {key_issues}"
        
        return summaries


class DataProcessor:
    """Enhanced main data processing pipeline coordinator."""
    
    def __init__(self):
        self.translator = TranslationService()
        self.absa_processor = ABSAProcessor()
        self.validator = DataValidator()
        self.analytics = AspectAnalytics()
        self.summary_generator = SummaryGenerator()
    
    def process_uploaded_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete enhanced processing pipeline for uploaded data.
        
        Args:
            df: Uploaded dataframe
            
        Returns:
            Dictionary containing all processed results with advanced analytics
        """
        # Validate data
        is_valid, errors = self.validator.validate_csv(df)
        if not is_valid:
            return {'error': errors}
        
        # Clean data
        df_clean = self.validator.clean_data(df)
        
        # Extract reviews
        reviews = df_clean['review'].tolist()
        
        # Translation
        with st.spinner("Translating reviews..."):
            translated_reviews, detected_languages = self.translator.process_reviews(reviews)
        
        # Enhanced intent classification
        with st.spinner("Classifying intents with severity analysis..."):
            intent_results = IntentClassifier.classify_batch_enhanced(translated_reviews)
        
        # ABSA processing
        with st.spinner("Extracting aspects and sentiments..."):
            absa_results = self.absa_processor.extract_aspects_and_sentiments(translated_reviews)
        
        # Combine results with enhanced structure
        df_processed = df_clean.copy()
        df_processed['translated_review'] = translated_reviews
        df_processed['detected_language'] = detected_languages
        df_processed['intent'] = [r['intent'] for r in intent_results]
        df_processed['intent_severity'] = [r['severity_type'] for r in intent_results]
        df_processed['intent_confidence'] = [r['confidence'] for r in intent_results]
        
        # Add ABSA results with enhanced structure
        aspects_list = []
        aspect_sentiments_list = []
        overall_sentiment = []
        
        for result in absa_results:
            aspects_list.append(result['aspects'])
            aspect_sentiments_list.append(result['sentiments'])
            
            # Calculate overall sentiment
            if result['sentiments']:
                positive_count = result['sentiments'].count('Positive')
                negative_count = result['sentiments'].count('Negative')
                if positive_count > negative_count:
                    overall_sentiment.append('Positive')
                elif negative_count > positive_count:
                    overall_sentiment.append('Negative')
                else:
                    overall_sentiment.append('Neutral')
            else:
                overall_sentiment.append('Neutral')
        
        df_processed['aspects'] = aspects_list
        df_processed['aspect_sentiments'] = aspect_sentiments_list
        df_processed['overall_sentiment'] = overall_sentiment
        
        # Advanced analytics
        with st.spinner("Calculating aspect analytics and priority scores..."):
            areas_of_improvement, strength_anchors = self.analytics.calculate_aspect_scores(df_processed)
            aspect_network = self.analytics.calculate_aspect_cooccurrence(df_processed)
            sentiment_alerts = self.analytics.detect_sentiment_spikes(df_processed)
        
        # Generate summaries
        with st.spinner("Generating AI-powered summaries..."):
            macro_summary = self.summary_generator.generate_macro_summary(
                df_processed, areas_of_improvement, strength_anchors
            )
            
            # Get top aspects for micro summaries
            top_negative_aspects = areas_of_improvement['aspect'].head(3).tolist() if len(areas_of_improvement) > 0 else []
            top_positive_aspects = strength_anchors['aspect'].head(3).tolist() if len(strength_anchors) > 0 else []
            top_aspects = top_negative_aspects + top_positive_aspects
            
            micro_summaries = self.summary_generator.generate_aspect_micro_summaries(
                df_processed, top_aspects
            )
        
        return {
            'processed_data': df_processed,
            'absa_details': absa_results,
            'areas_of_improvement': areas_of_improvement,
            'strength_anchors': strength_anchors,
            'aspect_network': aspect_network,
            'sentiment_alerts': sentiment_alerts,
            'macro_summary': macro_summary,
            'micro_summaries': micro_summaries,
            'summary': {
                'total_reviews': len(df_processed),
                'languages_detected': list(set(detected_languages)),
                'intents_distribution': pd.Series([r['intent'] for r in intent_results]).value_counts().to_dict(),
                'sentiment_distribution': pd.Series(overall_sentiment).value_counts().to_dict(),
                'top_problem_areas': len(areas_of_improvement),
                'top_strength_anchors': len(strength_anchors),
                'active_alerts': len(sentiment_alerts)
            }
        }