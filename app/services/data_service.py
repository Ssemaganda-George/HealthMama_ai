"""
Data Service for HealthMama AI
Handles loading and management of health data
"""
import os
import logging
from typing import List, Optional
from config.settings import Config


class DataService:
    """Service for managing health data"""
    
    def __init__(self):
        self.diabetes_data: List[str] = []
        self.preeclampsia_data: List[str] = []
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize data service by loading data files"""
        try:
            self.diabetes_data = self._load_data_file(Config.DIABETES_DATA_PATH)
            self.preeclampsia_data = self._load_data_file(Config.PREECLAMPSIA_DATA_PATH)
            
            self.logger.info(f"Loaded {len(self.diabetes_data)} diabetes data entries")
            self.logger.info(f"Loaded {len(self.preeclampsia_data)} preeclampsia data entries")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize data service: {e}")
            return False
    
    def _load_data_file(self, file_path: str) -> List[str]:
        """Load data from a text file"""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"Data file not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = file.readlines()
                # Clean the data
                return [line.strip() for line in data if line.strip()]
                
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def get_diabetes_data(self) -> List[str]:
        """Get diabetes data"""
        return self.diabetes_data
    
    def get_preeclampsia_data(self) -> List[str]:
        """Get preeclampsia data"""
        return self.preeclampsia_data
    
    def get_data_stats(self) -> dict:
        """Get data statistics"""
        return {
            'diabetes_entries': len(self.diabetes_data),
            'preeclampsia_entries': len(self.preeclampsia_data),
            'total_entries': len(self.diabetes_data) + len(self.preeclampsia_data)
        }
    
    def search_data(self, query: str, model_type: str = 'diabetes', limit: int = 5) -> List[str]:
        """Search for relevant data based on query"""
        if model_type == 'preeclampsia':
            data = self.preeclampsia_data
        else:
            data = self.diabetes_data
        
        if not data:
            return []
        
        return self._retrieve_context_simple(query, data, limit)
    
    def _retrieve_context_simple(self, query: str, data: List[str], top_n: int = 5) -> List[str]:
        """Simple keyword-based context retrieval with better matching for different languages"""
        query_words = query.lower().split()
        scored_data = []
        
        # Add common Luganda-English health term mappings
        luganda_terms = {
            'ssukali': ['diabetes', 'sugar', 'blood sugar'],
            'musujja': ['diabetes', 'sugar'],
            'omusujja': ['diabetes', 'sugar'],
            'omusayi': ['blood', 'pressure'],
            'omutwe': ['head', 'headache'],
            'olumbe': ['disease', 'illness'],
            'obulwadde': ['disease', 'illness', 'condition'],
            'okujjukira': ['remember', 'memory'],
            'endya': ['food', 'eat', 'diet'],
            'emmere': ['food', 'meal', 'diet'],
            'amabeere': ['milk', 'dairy'],
            'ebijjanjalo': ['vegetables'],
            'ebibala': ['fruits'],
            'amazzi': ['water'],
            'okutambula': ['walk', 'exercise'],
            'okukola': ['work', 'exercise']
        }
        
        # Expand query with related terms
        expanded_query = set(query_words)
        for word in query_words:
            if word in luganda_terms:
                expanded_query.update(luganda_terms[word])
        
        expanded_query = list(expanded_query)
        
        for idx, text in enumerate(data):
            text_lower = text.lower()
            score = sum(1 for word in expanded_query if word in text_lower)
            if score > 0:
                scored_data.append((score, idx, text))
        
        # Sort by score and return top results
        scored_data.sort(reverse=True)
        return [text for _, _, text in scored_data[:top_n]]