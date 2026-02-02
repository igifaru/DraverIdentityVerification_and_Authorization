"""
Logger Module
Handles performance logging and CSV export
"""

import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List
from database.models import VerificationLog
from database.db_manager import DatabaseManager
from utils.config import config


class PerformanceLogger:
    """Logs verification performance metrics to CSV"""
    
    def __init__(self, log_path: str = None):
        """
        Initialize performance logger
        
        Args:
            log_path: Path to CSV log file (uses config if not provided)
        """
        self.log_path = log_path or config.log_path
        self.db = DatabaseManager()
        
        # Ensure log directory exists
        log_dir = Path(self.log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV file with headers if it doesn't exist
        if not Path(self.log_path).exists():
            self._create_csv_file()
        
        print(f"✓ Performance logger initialized: {self.log_path}")
    
    def _create_csv_file(self):
        """Create CSV file with headers"""
        headers = [
            'timestamp',
            'driver_id',
            'driver_name',
            'similarity_score',
            'authorized',
            'liveness_passed',
            'processing_time_ms',
            'image_path'
        ]
        
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_verification(self, log: VerificationLog):
        """
        Log a verification attempt to CSV
        
        Args:
            log: VerificationLog object
        """
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'driver_id', 'driver_name', 'similarity_score',
                    'authorized', 'liveness_passed', 'processing_time_ms', 'image_path'
                ])
                
                writer.writerow(log.to_csv_row())
        
        except Exception as e:
            print(f"ERROR: Failed to log to CSV: {e}")
    
    def export_recent_logs(self, limit: int = 100, output_path: str = None) -> str:
        """
        Export recent logs from database to CSV
        
        Args:
            limit: Number of recent logs to export
            output_path: Output CSV path (uses default if not provided)
            
        Returns:
            Path to exported CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.log_path).parent / f"export_{timestamp}.csv"
        
        # Get recent logs from database
        logs = self.db.get_recent_logs(limit)
        
        # Write to CSV
        with open(output_path, 'w', newline='') as f:
            if logs:
                writer = csv.DictWriter(f, fieldnames=logs[0].to_csv_row().keys())
                writer.writeheader()
                
                for log in logs:
                    writer.writerow(log.to_csv_row())
        
        print(f"✓ Exported {len(logs)} logs to {output_path}")
        return str(output_path)
    
    def get_statistics(self) -> dict:
        """
        Calculate performance statistics from logs
        
        Returns:
            Dictionary with statistics
        """
        try:
            df = pd.read_csv(self.log_path)
            
            if df.empty:
                return {
                    'total_verifications': 0,
                    'authorized_count': 0,
                    'unauthorized_count': 0,
                    'avg_processing_time_ms': 0,
                    'max_processing_time_ms': 0,
                    'min_processing_time_ms': 0,
                    'avg_similarity_authorized': 0,
                    'avg_similarity_unauthorized': 0
                }
            
            stats = {
                'total_verifications': len(df),
                'authorized_count': len(df[df['authorized'] == 'YES']),
                'unauthorized_count': len(df[df['authorized'] == 'NO']),
                'avg_processing_time_ms': df['processing_time_ms'].astype(float).mean(),
                'max_processing_time_ms': df['processing_time_ms'].astype(float).max(),
                'min_processing_time_ms': df['processing_time_ms'].astype(float).min(),
            }
            
            # Calculate average similarity for authorized and unauthorized
            authorized_df = df[df['authorized'] == 'YES']
            unauthorized_df = df[df['authorized'] == 'NO']
            
            if not authorized_df.empty:
                stats['avg_similarity_authorized'] = authorized_df['similarity_score'].astype(float).mean()
            else:
                stats['avg_similarity_authorized'] = 0
            
            if not unauthorized_df.empty:
                stats['avg_similarity_unauthorized'] = unauthorized_df['similarity_score'].astype(float).mean()
            else:
                stats['avg_similarity_unauthorized'] = 0
            
            return stats
            
        except Exception as e:
            print(f"ERROR: Failed to calculate statistics: {e}")
            return {}
    
    def print_statistics(self):
        """Print performance statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Total Verifications: {stats.get('total_verifications', 0)}")
        print(f"  Authorized: {stats.get('authorized_count', 0)}")
        print(f"  Unauthorized: {stats.get('unauthorized_count', 0)}")
        print(f"\nProcessing Time:")
        print(f"  Average: {stats.get('avg_processing_time_ms', 0):.2f} ms")
        print(f"  Maximum: {stats.get('max_processing_time_ms', 0):.2f} ms")
        print(f"  Minimum: {stats.get('min_processing_time_ms', 0):.2f} ms")
        print(f"\nSimilarity Scores:")
        print(f"  Authorized (avg): {stats.get('avg_similarity_authorized', 0):.4f}")
        print(f"  Unauthorized (avg): {stats.get('avg_similarity_unauthorized', 0):.4f}")
        print("="*60 + "\n")
    
    def rotate_log(self, max_entries: int = None):
        """
        Rotate log file if it exceeds maximum entries
        
        Args:
            max_entries: Maximum entries before rotation (uses config if not provided)
        """
        max_entries = max_entries or config.get('logging.max_log_entries', 10000)
        
        try:
            df = pd.read_csv(self.log_path)
            
            if len(df) > max_entries:
                # Create backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = Path(self.log_path).parent / f"backup_{timestamp}.csv"
                
                # Keep only recent entries
                df_recent = df.tail(max_entries)
                df_recent.to_csv(self.log_path, index=False)
                
                # Save old entries to backup
                df.to_csv(backup_path, index=False)
                
                print(f"✓ Log rotated: {len(df)} entries backed up to {backup_path}")
                print(f"  Kept {len(df_recent)} recent entries")
        
        except Exception as e:
            print(f"ERROR: Failed to rotate log: {e}")


if __name__ == "__main__":
    # Test performance logger
    print("Testing performance logger...")
    
    logger = PerformanceLogger()
    
    # Print statistics
    logger.print_statistics()
    
    # Test log rotation
    # logger.rotate_log(max_entries=100)
