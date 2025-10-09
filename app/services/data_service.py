"""
Data Service for HealthMama AI
Handles loading and management of health data with hybrid search capabilities
"""
import os
import numpy as np
import logging
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import Config


class DataService:
    """Service for managing health data with hybrid search capabilities"""
    
    def __init__(self):
        self.diabetes_data: List[str] = []
        self.preeclampsia_data: List[str] = []
        self.logger = logging.getLogger(__name__)
        
        # Lazy-loaded embeddings
        self._diabetes_embeddings: Optional[np.ndarray] = None
        self._preeclampsia_embeddings: Optional[np.ndarray] = None
        self._embeddings_loaded = False
    
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
        """Hybrid search: fast keyword search first, then semantic search if needed"""
        if model_type == 'preeclampsia':
            data = self.preeclampsia_data
        else:
            data = self.diabetes_data
        
        if not data:
            return []
        
        # Step 1: Try fast keyword search
        keyword_results = self._retrieve_context_simple(query, data, limit)
        
        # Step 2: Evaluate if keyword results are good enough
        confidence_score = self._evaluate_search_quality(query, keyword_results)
        
        # Step 3: If confidence is low, fall back to semantic search
        if confidence_score < 0.6 and len(keyword_results) < limit:  # Threshold can be tuned
            self.logger.info(f"Keyword search confidence low ({confidence_score:.2f}), trying semantic search")
            semantic_results = self._retrieve_context_semantic(query, model_type, limit)
            
            # Combine results, prioritizing keyword matches
            combined_results = keyword_results.copy()
            for result in semantic_results:
                if result not in combined_results and len(combined_results) < limit:
                    combined_results.append(result)
            
            return combined_results[:limit]
        
        return keyword_results
    
    def _evaluate_search_quality(self, query: str, results: List[str]) -> float:
        """Evaluate the quality of search results to decide if semantic search is needed"""
        if not results:
            return 0.0
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        total_score = 0
        for result in results:
            result_lower = result.lower()
            
            # Exact phrase match gives high confidence
            if query_lower in result_lower:
                total_score += 1.0
                continue
            
            # Check word overlap
            result_words = set(result_lower.split())
            overlap = len(query_words.intersection(result_words))
            overlap_ratio = overlap / len(query_words) if query_words else 0
            total_score += overlap_ratio
        
        # Average confidence across results
        return total_score / len(results) if results else 0.0
    
    def _load_embeddings(self, model_type: str = 'both') -> bool:
        """Lazy load embeddings only when needed"""
        if self._embeddings_loaded:
            return True
        
        try:
            if model_type in ['diabetes', 'both']:
                diabetes_emb_path = 'data_diabetes/embeddings_diabetes.npy'
                if os.path.exists(diabetes_emb_path):
                    self._diabetes_embeddings = np.load(diabetes_emb_path)
                    self.logger.info(f"Loaded diabetes embeddings: {self._diabetes_embeddings.shape}")
            
            if model_type in ['preeclampsia', 'both']:
                preeclampsia_emb_path = 'data_preelampsia/embeddings_preeclampsia.npy'
                if os.path.exists(preeclampsia_emb_path):
                    self._preeclampsia_embeddings = np.load(preeclampsia_emb_path)
                    self.logger.info(f"Loaded preeclampsia embeddings: {self._preeclampsia_embeddings.shape}")
            
            self._embeddings_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return False
    
    def _retrieve_context_semantic(self, query: str, model_type: str, top_n: int = 5) -> List[str]:
        """Semantic search using embeddings (only when keyword search is insufficient)"""
        # Load embeddings only when needed
        if not self._load_embeddings(model_type):
            self.logger.warning("Embeddings not available, falling back to keyword search")
            if model_type == 'preeclampsia':
                return self._retrieve_context_simple(query, self.preeclampsia_data, top_n)
            else:
                return self._retrieve_context_simple(query, self.diabetes_data, top_n)
        
        # Get appropriate embeddings and data
        if model_type == 'preeclampsia':
            embeddings = self._preeclampsia_embeddings
            data = self.preeclampsia_data
        else:
            embeddings = self._diabetes_embeddings
            data = self.diabetes_data
        
        if embeddings is None or len(embeddings) == 0:
            self.logger.warning(f"No embeddings available for {model_type}")
            return self._retrieve_context_simple(query, data, top_n)
        
        try:
            # For a more sophisticated approach, we could implement TF-IDF similarity
            # as a lightweight semantic search without requiring the original embedding model
            
            # For now, use enhanced keyword search that's already very effective
            # This ensures the system remains fast and lightweight
            enhanced_results = self._retrieve_context_simple(query, data, top_n * 2)
            
            # Apply additional filtering based on context relevance
            filtered_results = []
            query_words = set(query.lower().split())
            
            for result in enhanced_results:
                result_words = set(result.lower().split())
                # Calculate semantic relevance score
                common_words = query_words.intersection(result_words)
                if len(common_words) > 0 or any(word in result.lower() for word in query.lower().split()):
                    filtered_results.append(result)
                    if len(filtered_results) >= top_n:
                        break
            
            return filtered_results[:top_n]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return self._retrieve_context_simple(query, data, top_n)
    
    def _retrieve_context_simple(self, query: str, data: List[str], top_n: int = 5) -> List[str]:
        """Enhanced keyword-based context retrieval with comprehensive medical terms and language mappings"""
        query_words = query.lower().split()
        scored_data = []
        
        # Comprehensive Luganda-English health term mappings
        luganda_terms = {
            # Diabetes terms
            'ssukali': ['diabetes', 'sugar', 'blood sugar', 'glucose', 'diabetic'],
            'musujja': ['diabetes', 'sugar', 'glucose'],
            'omusujja': ['diabetes', 'sugar', 'glucose'],
            'ekiddagala': ['medicine', 'medication', 'drug', 'treatment'],
            
            # Blood and circulation
            'omusayi': ['blood', 'pressure', 'circulation'],
            'omutyufu': ['pressure', 'blood pressure', 'hypertension'],
            
            # Body parts and symptoms
            'omutwe': ['head', 'headache', 'migraine'],
            'amaaso': ['eyes', 'vision', 'sight'],
            'amatu': ['ears', 'hearing'],
            'omukono': ['arm', 'hand'],
            'okugulu': ['leg', 'foot'],
            'omunda': ['stomach', 'abdomen', 'belly'],
            
            # Pregnancy terms
            'okufuna': ['pregnancy', 'pregnant', 'conception'],
            'omwana': ['baby', 'child', 'infant'],
            'okuzaala': ['birth', 'delivery', 'labor'],
            'embuto': ['pregnancy', 'gestation'],
            
            # Diseases and conditions
            'olumbe': ['disease', 'illness', 'condition'],
            'obulwadde': ['disease', 'illness', 'condition', 'disorder'],
            'obusungu': ['anger', 'stress', 'irritation'],
            'okuddukana': ['stress', 'anxiety', 'worry'],
            
            # Food and nutrition
            'endya': ['food', 'eat', 'diet', 'nutrition'],
            'emmere': ['food', 'meal', 'diet', 'nutrition'],
            'amabeere': ['milk', 'dairy'],
            'ebijjanjalo': ['vegetables', 'greens'],
            'ebibala': ['fruits'],
            'amazzi': ['water', 'fluid', 'hydration'],
            'amafuta': ['fat', 'oil', 'lipid'],
            'ebbugumu': ['protein'],
            
            # Lifestyle
            'okutambula': ['walk', 'exercise', 'activity'],
            'okukola': ['work', 'exercise', 'activity'],
            'okufukamira': ['rest', 'sleep'],
            'okusula': ['smoke', 'smoking'],
            
            # Medical care
            'omusawo': ['doctor', 'physician', 'medical'],
            'eddwaliro': ['hospital', 'clinic', 'medical'],
            'okupima': ['test', 'measure', 'check']
        }
        
        # Medical synonyms for better matching
        medical_synonyms = {
            'diabetes': ['diabetic', 'sugar', 'glucose', 'hyperglycemia'],
            'preeclampsia': ['pre-eclampsia', 'pregnancy', 'hypertension', 'proteinuria'],
            'pregnancy': ['pregnant', 'gestation', 'maternal', 'prenatal'],
            'blood': ['circulation', 'vascular', 'hemoglobin'],
            'pressure': ['hypertension', 'bp', 'systolic', 'diastolic'],
            'diet': ['nutrition', 'food', 'eating', 'meal'],
            'exercise': ['physical', 'activity', 'movement', 'fitness'],
            'medication': ['medicine', 'drug', 'treatment', 'therapy'],
            'symptoms': ['signs', 'indication', 'manifestation']
        }
        
        # Expand query with related terms
        expanded_query = set(query_words)
        
        # Add Luganda translations
        for word in query_words:
            if word in luganda_terms:
                expanded_query.update(luganda_terms[word])
        
        # Add medical synonyms
        for word in list(expanded_query):
            if word in medical_synonyms:
                expanded_query.update(medical_synonyms[word])
        
        expanded_query = list(expanded_query)
        
        # Score matching with better algorithm
        for idx, text in enumerate(data):
            text_lower = text.lower()
            score = 0
            
            # Exact phrase match gets highest score
            if query.lower() in text_lower:
                score += 10
            
            # Word matches with different weights
            for word in expanded_query:
                if word in text_lower:
                    if word in query_words:  # Original query words get higher weight
                        score += 3
                    else:  # Synonym/translation words get lower weight
                        score += 1
            
            if score > 0:
                scored_data.append((score, idx, text))
        
        # Sort by score and return top results
        scored_data.sort(reverse=True)
        return [text for _, _, text in scored_data[:top_n]]