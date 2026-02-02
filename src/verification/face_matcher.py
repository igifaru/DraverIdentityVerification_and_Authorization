"""
Face Matcher Module
Handles embedding comparison and similarity calculation
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine
from database.db_manager import DatabaseManager
from utils.config import config


class FaceMatcher:
    """Matches live face embeddings against enrolled drivers"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize face matcher
        
        Args:
            db_manager: Database manager instance (creates new if not provided)
        """
        self.db = db_manager or DatabaseManager()
        self.threshold = config.verification_threshold
        print(f"Face matcher initialized (threshold: {self.threshold})")
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Cosine Similarity = 1 - Cosine Distance
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Cosine distance ranges from 0 (identical) to 2 (opposite)
        # Cosine similarity = 1 - cosine_distance
        cos_distance = cosine(embedding1, embedding2)
        similarity = 1 - cos_distance
        
        return similarity
    
    def calculate_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Euclidean distance (lower is more similar)
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def find_best_match(self, live_embedding: np.ndarray) -> Tuple[Optional[int], Optional[str], float]:
        """
        Find best matching driver for live embedding
        
        Args:
            live_embedding: Embedding from live camera feed
            
        Returns:
            Tuple of (driver_id, driver_name, similarity_score)
            Returns (None, None, 0.0) if no match found
        """
        # Get all enrolled embeddings
        enrolled_embeddings = self.db.get_all_embeddings()
        
        if not enrolled_embeddings:
            print("WARNING: No enrolled drivers in database")
            return None, None, 0.0
        
        best_match_id = None
        best_match_name = None
        best_similarity = 0.0
        
        # Compare with each enrolled driver
        for driver_id, driver_name, enrolled_embedding in enrolled_embeddings:
            similarity = self.calculate_cosine_similarity(live_embedding, enrolled_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = driver_id
                best_match_name = driver_name
        
        return best_match_id, best_match_name, best_similarity
    
    def verify_identity(self, live_embedding: np.ndarray, threshold: float = None) -> Tuple[bool, Optional[int], Optional[str], float]:
        """
        Verify if live embedding matches any enrolled driver
        
        Args:
            live_embedding: Embedding from live camera feed
            threshold: Similarity threshold (uses config if not provided)
            
        Returns:
            Tuple of (is_authorized, driver_id, driver_name, similarity_score)
        """
        threshold = threshold or self.threshold
        
        # Find best match
        driver_id, driver_name, similarity = self.find_best_match(live_embedding)
        
        # Check if similarity meets threshold
        is_authorized = similarity >= threshold
        
        if is_authorized:
            print(f"✓ AUTHORIZED: {driver_name} (Similarity: {similarity:.4f})")
        else:
            if driver_id:
                print(f"✗ UNAUTHORIZED: Best match '{driver_name}' below threshold "
                      f"(Similarity: {similarity:.4f} < {threshold:.4f})")
            else:
                print(f"✗ UNAUTHORIZED: No enrolled drivers")
        
        return is_authorized, driver_id, driver_name, similarity
    
    def get_all_similarities(self, live_embedding: np.ndarray) -> List[Tuple[int, str, float]]:
        """
        Get similarity scores for all enrolled drivers
        
        Args:
            live_embedding: Embedding from live camera feed
            
        Returns:
            List of tuples (driver_id, driver_name, similarity_score) sorted by similarity
        """
        enrolled_embeddings = self.db.get_all_embeddings()
        
        similarities = []
        for driver_id, driver_name, enrolled_embedding in enrolled_embeddings:
            similarity = self.calculate_cosine_similarity(live_embedding, enrolled_embedding)
            similarities.append((driver_id, driver_name, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        return similarities
    
    def set_threshold(self, threshold: float):
        """
        Update similarity threshold
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        print(f"Similarity threshold updated to: {threshold}")
    
    def get_threshold(self) -> float:
        """Get current similarity threshold"""
        return self.threshold
    
    def benchmark_threshold(self, test_embeddings: List[Tuple[int, np.ndarray, bool]]) -> dict:
        """
        Benchmark different thresholds to find optimal value
        
        Args:
            test_embeddings: List of (driver_id, embedding, is_genuine) tuples
                           is_genuine=True for authorized, False for impostors
            
        Returns:
            Dictionary with threshold analysis
        """
        thresholds = np.arange(0.3, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for driver_id, embedding, is_genuine in test_embeddings:
                is_authorized, _, _, _ = self.verify_identity(embedding, threshold)
                
                if is_genuine and is_authorized:
                    true_positives += 1
                elif is_genuine and not is_authorized:
                    false_negatives += 1
                elif not is_genuine and is_authorized:
                    false_positives += 1
                else:
                    true_negatives += 1
            
            # Calculate metrics
            total_genuine = sum(1 for _, _, is_gen in test_embeddings if is_gen)
            total_impostor = len(test_embeddings) - total_genuine
            
            far = false_positives / total_impostor if total_impostor > 0 else 0  # False Acceptance Rate
            frr = false_negatives / total_genuine if total_genuine > 0 else 0  # False Rejection Rate
            accuracy = (true_positives + true_negatives) / len(test_embeddings)
            
            results.append({
                'threshold': threshold,
                'far': far,
                'frr': frr,
                'accuracy': accuracy,
                'eer': abs(far - frr)  # Equal Error Rate (when FAR ≈ FRR)
            })
        
        # Find optimal threshold (minimum EER)
        optimal = min(results, key=lambda x: x['eer'])
        
        return {
            'optimal_threshold': optimal['threshold'],
            'optimal_far': optimal['far'],
            'optimal_frr': optimal['frr'],
            'optimal_accuracy': optimal['accuracy'],
            'all_results': results
        }


if __name__ == "__main__":
    # Test face matcher
    print("Testing face matcher...")
    
    matcher = FaceMatcher()
    
    # Get statistics
    stats = matcher.db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total drivers: {stats['total_drivers']}")
    print(f"  Current threshold: {matcher.get_threshold()}")
    
    # Test with random embeddings
    if stats['total_drivers'] > 0:
        print("\nTesting with random embedding...")
        random_embedding = np.random.randn(128)
        
        is_authorized, driver_id, driver_name, similarity = matcher.verify_identity(random_embedding)
        print(f"Result: {'AUTHORIZED' if is_authorized else 'UNAUTHORIZED'}")
        print(f"Best match: {driver_name} (Similarity: {similarity:.4f})")
