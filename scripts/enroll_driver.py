"""
Driver Enrollment Script
Command-line interface for enrolling new drivers
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enrollment.enrollment_manager import EnrollmentManager


def main():
    parser = argparse.ArgumentParser(
        description="Enroll a new driver in the verification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enroll_driver.py --name "John Doe"
  python enroll_driver.py --name "Jane Smith" --email "jane@example.com"
  python enroll_driver.py --name "Bob Johnson" --no-interactive
        """
    )
    
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Driver name (required)'
    )
    
    parser.add_argument(
        '--email',
        type=str,
        default=None,
        help='Driver email address (optional)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive camera preview (auto-capture mode)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all enrolled drivers and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize enrollment manager
    manager = EnrollmentManager()
    
    # List drivers if requested
    if args.list:
        manager.list_enrolled_drivers()
        return
    
    # Enroll driver
    print(f"\nEnrolling driver: {args.name}")
    if args.email:
        print(f"Email: {args.email}")
    
    interactive = not args.no_interactive
    
    success, message = manager.enroll_driver(
        name=args.name,
        email=args.email,
        interactive=interactive
    )
    
    if success:
        print(f"\n✓ SUCCESS: {message}")
        print("\nYou can now run verification:")
        print("  python scripts/run_verification.py")
        return 0
    else:
        print(f"\n✗ FAILED: {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
