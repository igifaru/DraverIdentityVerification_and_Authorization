"""
Verification Runner Script
Command-line interface for running driver verification
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verification.verification_engine import VerificationEngine
from database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser(
        description="Run driver identity verification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_verification.py
  python run_verification.py --no-preview
  python run_verification.py --no-liveness
  python run_verification.py --threshold 0.7
        """
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable video preview window'
    )
    
    parser.add_argument(
        '--no-liveness',
        action='store_true',
        help='Disable liveness detection (not recommended)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Override similarity threshold (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    # Check if drivers are enrolled
    db = DatabaseManager()
    stats = db.get_statistics()
    
    if stats['total_drivers'] == 0:
        print("\n" + "="*60)
        print("ERROR: No drivers enrolled!")
        print("="*60)
        print("\nPlease enroll at least one driver before running verification.")
        print("\nTo enroll a driver, run:")
        print("  python scripts/enroll_driver.py --name \"Your Name\"")
        print("\nOr to see all options:")
        print("  python scripts/enroll_driver.py --help")
        print("="*60 + "\n")
        return 1
    
    # Initialize verification engine
    engine = VerificationEngine()
    
    # Override threshold if specified
    if args.threshold is not None:
        if not (0 <= args.threshold <= 1):
            print("ERROR: Threshold must be between 0.0 and 1.0")
            return 1
        engine.face_matcher.set_threshold(args.threshold)
        print(f"Similarity threshold set to: {args.threshold}")
    
    # Run verification
    show_preview = not args.no_preview
    enable_liveness = not args.no_liveness
    
    print(f"\nFound {stats['total_drivers']} enrolled driver(s)")
    print("Starting verification system...\n")
    
    try:
        engine.run_continuous_verification(
            show_preview=show_preview,
            enable_liveness=enable_liveness
        )
        return 0
    
    except KeyboardInterrupt:
        print("\n\nVerification stopped by user")
        return 0
    
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
