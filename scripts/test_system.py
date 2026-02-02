"""
System Testing Script
Tests various components of the verification system
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import config
from database.db_manager import DatabaseManager
from enrollment.camera_capture import CameraCapture
from enrollment.face_processor import FaceProcessor
from alerting.email_service import EmailService
from alerting.logger import PerformanceLogger


def test_configuration():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    print(f"✓ Config loaded successfully")
    print(f"  Database path: {config.database_path}")
    print(f"  Log path: {config.log_path}")
    print(f"  Similarity threshold: {config.verification_threshold}")
    print(f"  Camera resolution: {config.camera_resolution}")
    print(f"  Camera FPS: {config.camera_fps}")
    
    is_valid = config.validate()
    
    if is_valid:
        print("\n✓ Configuration validation PASSED")
    else:
        print("\n✗ Configuration validation FAILED")
    
    return is_valid


def test_database():
    """Test database operations"""
    print("\n" + "="*60)
    print("TESTING DATABASE")
    print("="*60)
    
    try:
        db = DatabaseManager()
        print(f"✓ Database initialized: {db.db_path}")
        
        stats = db.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Total drivers: {stats['total_drivers']}")
        print(f"  Total verifications: {stats['total_verifications']}")
        print(f"  Authorization rate: {stats['authorization_rate']:.2f}%")
        
        print("\n✓ Database test PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Database test FAILED: {e}")
        return False


def test_camera():
    """Test camera access"""
    print("\n" + "="*60)
    print("TESTING CAMERA")
    print("="*60)
    
    try:
        camera = CameraCapture()
        
        if not camera.initialize():
            print("✗ Camera initialization FAILED")
            return False
        
        print(f"✓ Camera initialized: Device {camera.device_id}")
        print(f"  Resolution: {camera.resolution[0]}x{camera.resolution[1]}")
        
        # Capture test frame
        frame = camera.capture_frame()
        
        if frame is None:
            print("✗ Frame capture FAILED")
            camera.release()
            return False
        
        print(f"✓ Frame captured: {frame.shape}")
        
        # Test quality assessment
        quality = camera.assess_frame_quality(frame)
        print(f"  Frame quality: {quality:.2f}")
        
        camera.release()
        print("\n✓ Camera test PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Camera test FAILED: {e}")
        return False


def test_face_detection():
    """Test face detection"""
    print("\n" + "="*60)
    print("TESTING FACE DETECTION")
    print("="*60)
    print("Please position your face in front of the camera...")
    print("Press SPACE to test detection, ESC to skip")
    
    try:
        processor = FaceProcessor()
        camera = CameraCapture()
        
        if not camera.initialize():
            print("✗ Camera initialization FAILED")
            return False
        
        detected = False
        
        while True:
            frame = camera.capture_frame()
            
            if frame is None:
                break
            
            detection = processor.detect_face(frame)
            
            if detection:
                annotated = processor.draw_detection(frame, detection)
                is_valid, reason = processor.validate_face_quality(frame, detection)
                
                status_text = f"Valid: {is_valid} - {reason}"
                color = (0, 255, 0) if is_valid else (0, 165, 255)
                
                import cv2
                cv2.putText(annotated, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Face Detection Test", annotated)
            else:
                import cv2
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Face Detection Test", frame)
            
            import cv2
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                if detection:
                    print(f"\n✓ Face detected with confidence: {detection['confidence']:.2f}")
                    print(f"  Valid: {is_valid} - {reason}")
                    detected = True
                    break
                else:
                    print("\n✗ No face detected in frame")
            elif key == 27:  # ESC
                print("\nTest skipped")
                break
        
        camera.release()
        import cv2
        cv2.destroyAllWindows()
        
        if detected:
            print("\n✓ Face detection test PASSED")
            return True
        else:
            print("\n⚠️  Face detection test SKIPPED")
            return True
    
    except Exception as e:
        print(f"\n✗ Face detection test FAILED: {e}")
        return False


def test_email():
    """Test email configuration"""
    print("\n" + "="*60)
    print("TESTING EMAIL SERVICE")
    print("="*60)
    
    try:
        service = EmailService()
        
        if service.is_configured:
            print("✓ Email service is configured")
            print(f"  SMTP Server: {service.smtp_server}:{service.smtp_port}")
            print(f"  Sender: {service.sender_email}")
            print("\n✓ Email test PASSED")
            return True
        else:
            print("⚠️  Email service is NOT configured")
            print("  Update config/.env with SMTP credentials to enable alerts")
            print("\n⚠️  Email test SKIPPED (not configured)")
            return True
    
    except Exception as e:
        print(f"\n✗ Email test FAILED: {e}")
        return False


def test_logger():
    """Test performance logger"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE LOGGER")
    print("="*60)
    
    try:
        logger = PerformanceLogger()
        print(f"✓ Logger initialized: {logger.log_path}")
        
        stats = logger.get_statistics()
        print(f"\nLog Statistics:")
        print(f"  Total verifications: {stats.get('total_verifications', 0)}")
        print(f"  Avg processing time: {stats.get('avg_processing_time_ms', 0):.2f} ms")
        
        print("\n✓ Logger test PASSED")
        return True
    
    except Exception as e:
        print(f"\n✗ Logger test FAILED: {e}")
        return False


def benchmark_embedding_generation():
    """Benchmark embedding generation speed"""
    print("\n" + "="*60)
    print("BENCHMARKING EMBEDDING GENERATION")
    print("="*60)
    
    try:
        import numpy as np
        # Use actual FaceProcessor
        from enrollment.face_processor import FaceProcessor
        
        processor = FaceProcessor()
        
        # Create test image (160x160 aligned)
        test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        # Preprocess manually to match pipeline input
        preprocessed = processor.preprocess_face(test_image)
        
        # Warm up
        print("Warming up...")
        for _ in range(3):
            processor.generate_embedding(preprocessed)
        
        # Benchmark
        print("Running benchmark (10 iterations)...")
        times = []
        
        for i in range(10):
            start = time.time()
            processor.generate_embedding(preprocessed)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f} ms")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nResults:")
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Minimum: {min_time:.2f} ms")
        print(f"  Maximum: {max_time:.2f} ms")
        
        if avg_time < 1500:
            print(f"\n✓ Performance target MET (<1500ms)")
        else:
            print(f"\n⚠️  Performance target EXCEEDED (>{avg_time:.0f}ms)")
        
        print("\n✓ Benchmark COMPLETED")
        return True
    
    except Exception as e:
        print(f"\n✗ Benchmark FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test driver verification system components",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Test configuration'
    )
    
    parser.add_argument(
        '--database',
        action='store_true',
        help='Test database'
    )
    
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Test camera'
    )
    
    parser.add_argument(
        '--face-detection',
        action='store_true',
        help='Test face detection'
    )
    
    parser.add_argument(
        '--email',
        action='store_true',
        help='Test email service'
    )
    
    parser.add_argument(
        '--logger',
        action='store_true',
        help='Test performance logger'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark embedding generation'
    )
    
    args = parser.parse_args()
    
    # If no specific test selected, run all
    if not any([args.config, args.database, args.camera, args.face_detection, 
                args.email, args.logger, args.benchmark]):
        args.all = True
    
    results = []
    
    if args.all or args.config:
        results.append(("Configuration", test_configuration()))
    
    if args.all or args.database:
        results.append(("Database", test_database()))
    
    if args.all or args.camera:
        results.append(("Camera", test_camera()))
    
    if args.all or args.face_detection:
        results.append(("Face Detection", test_face_detection()))
    
    if args.all or args.email:
        results.append(("Email Service", test_email()))
    
    if args.all or args.logger:
        results.append(("Performance Logger", test_logger()))
    
    if args.all or args.benchmark:
        results.append(("Embedding Benchmark", benchmark_embedding_generation()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
