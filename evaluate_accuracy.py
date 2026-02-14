"""
Accuracy Evaluation Script
Running batch verification tests against a dataset of images
"""
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tabulate import tabulate
import time

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enrollment.face_processor import FaceProcessor
from verification.face_matcher import FaceMatcher
from database.db_manager import DatabaseManager
from utils.config import config

def evaluate_accuracy(test_dir: str):
    """
    Evaluate accuracy by running verification on a folder of images
    Assumes filenames contain the expected driver name or 'unknown'
    e.g., "john_doe_1.jpg", "unknown_person.jpg"
    """
    print("="*60)
    print(f"STARTING ACCURACY EVALUATION")
    print(f"Test Directory: {test_dir}")
    print("="*60)

    # Initialize Components
    db = DatabaseManager()
    stats = db.get_statistics()
    print(f"Enrolled Drivers: {stats['total_drivers']}")
    
    if stats['total_drivers'] == 0:
        print("ERROR: No drivers enrolled in the database. Cannot perform evaluation.")
        return

    face_processor = FaceProcessor()
    face_matcher = FaceMatcher()
    
    # Process Images
    image_files = list(Path(test_dir).glob('*.[jJ][pP]*[gG]')) + list(Path(test_dir).glob('*.png'))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return

    results = []
    
    print(f"\nProcessing {len(image_files)} images...")
    
    start_time_all = time.time()
    
    for img_path in image_files:
        filename = img_path.name
        print(f"Processing {filename}...", end='\r')
        
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to read {filename}")
            continue
            
        start_time = time.time()
        
        # Process
        # 1. Detection & Preprocessing
        preprocessed, status = face_processor.process_for_enrollment(frame)
        
        if preprocessed is None:
            results.append({
                'file': filename,
                'status': 'Face Not Detected',
                'match': False,
                'score': 0.0,
                'time': (time.time() - start_time) * 1000
            })
            continue
            
        # 2. Embedding
        embedding = face_processor.generate_embedding(preprocessed)
        if embedding is None:
            results.append({
                'file': filename,
                'status': 'Embedding Failed',
                'match': False,
                'score': 0.0,
                'time': (time.time() - start_time) * 1000
            })
            continue
            
        # 3. Matching
        is_authorized, driver_id, driver_name, similarity = face_matcher.verify_identity(embedding)
        
        results.append({
            'file': filename,
            'status': f"Match: {driver_name}" if is_authorized else "Unknown",
            'match': is_authorized,
            'driver': driver_name,
            'score': similarity,
            'time': (time.time() - start_time) * 1000
        })

    total_time = time.time() - start_time_all
    print(f"\n\nEvaluation Complete in {total_time:.2f}s")
    
    # Print Table
    table_data = [[r['file'], r['status'], f"{r['score']:.4f}", f"{r['time']:.1f}ms"] for r in results]
    print(tabulate(table_data, headers=["File", "Result", "Similarity", "Time (ms)"], tablefmt="grid"))
    
    # Summary Metrics
    passed = sum(1 for r in results if r['match'])
    failed = len(results) - passed
    avg_time = np.mean([r['time'] for r in results])
    
    print("\nSUMMARY METRICS")
    print(f"Total Images: {len(results)}")
    print(f"Authorized Matches: {passed}")
    print(f"Unauthorized/Unknown: {failed}")
    print(f"Avg Processing Time: {avg_time:.2f} ms")
    print(f"Throughput: {len(results)/total_time:.1f} images/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy Evaluation Tool")
    parser.add_argument("test_dir", help="Directory containing test images")
    args = parser.parse_args()
    
    if os.path.isdir(args.test_dir):
        evaluate_accuracy(args.test_dir)
    else:
        print(f"Directory not found: {args.test_dir}")
