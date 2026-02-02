# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd DraverIdentityVerification_and_Authorization

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

**Note**: First installation may take 2-3 minutes as it downloads TensorFlow and other ML libraries.

### Step 2: Configure Email Alerts (Optional, 1 minute)

```bash
# Copy environment template
copy config\.env.example config\.env

# Edit config\.env with your Gmail credentials
# SMTP_EMAIL=your-email@gmail.com
# SMTP_PASSWORD=your-app-specific-password
```

**Skip this step if you don't need email alerts** - the system will work without it.

### Step 3: Test Your System (1 minute)

```bash
# Run system tests
python scripts/test_system.py --camera --face-detection
```

This will:
- ✅ Test camera access
- ✅ Test face detection
- ✅ Show live preview

Press **SPACE** to test face detection, **ESC** to continue.

### Step 4: Enroll Yourself (1 minute)

```bash
# Enroll with your name
python scripts/enroll_driver.py --name "Your Name"
```

When the camera preview appears:
1. Position your face in the center
2. Ensure good lighting
3. Press **SPACE** to capture
4. Wait for "ENROLLMENT SUCCESSFUL"

### Step 5: Run Verification (30 seconds)

```bash
# Start verification system
python scripts/run_verification.py
```

You should see:
- ✅ Live camera feed
- ✅ Your name when detected
- ✅ "AUTHORIZED" status
- ✅ Similarity score (should be > 0.6)

Press **Q** or **ESC** to stop.

---

## 🎨 Alternative: Use Web Dashboard

For a better visual experience, use the web dashboard:

```bash
# Start Flask dashboard
python src/dashboard/app.py
```

Then open your browser to: **http://localhost:5000**

You'll see:
- 📹 Live video feed
- 📊 Real-time statistics
- 📝 Verification logs
- ⚡ Performance metrics

---

## 🧪 Test Unauthorized Access

To test the alert system:

1. **Enroll yourself** (if not already done)
2. **Ask a friend** to stand in front of the camera
3. **Watch the system** detect them as unauthorized
4. **Check the logs** in `data/logs/verification_log.csv`
5. **Check captured images** in `data/alerts/captured_images/`

If email is configured, you'll receive an alert email!

---

## 📊 View Performance Statistics

```bash
# View detailed statistics
python -c "from src.alerting.logger import PerformanceLogger; logger = PerformanceLogger(); logger.print_statistics()"
```

This shows:
- Total verifications
- Authorization rate
- Average processing time
- Similarity scores

---

## 🎯 Common Commands

### Enrollment
```bash
# Enroll with email
python scripts/enroll_driver.py --name "John Doe" --email "john@example.com"

# List all enrolled drivers
python scripts/enroll_driver.py --list

# Auto-capture mode (no preview)
python scripts/enroll_driver.py --name "Jane Smith" --no-interactive
```

### Verification
```bash
# Standard verification
python scripts/run_verification.py

# Without video preview (headless)
python scripts/run_verification.py --no-preview

# Adjust threshold (more strict)
python scripts/run_verification.py --threshold 0.7

# Disable liveness detection (not recommended)
python scripts/run_verification.py --no-liveness
```

### Testing
```bash
# Test all components
python scripts/test_system.py --all

# Test specific components
python scripts/test_system.py --camera
python scripts/test_system.py --database
python scripts/test_system.py --benchmark
```

### Dashboard
```bash
# Start web dashboard
python src/dashboard/app.py

# Access at: http://localhost:5000
```

---

## ⚙️ Configuration Tips

### Adjust Similarity Threshold

Edit `config/config.yaml`:

```yaml
verification:
  similarity_threshold: 0.6  # Default
  # 0.5 = More lenient (fewer false rejects, more false accepts)
  # 0.7 = More strict (more false rejects, fewer false accepts)
```

### Change Camera

If using an external camera:

```yaml
camera:
  device_id: 1  # Try 0, 1, 2, etc.
```

### Adjust Performance

For faster processing (lower quality):

```yaml
camera:
  resolution_width: 480
  resolution_height: 360
```

---

## 🐛 Troubleshooting

### "Camera not found"
```bash
# Try different device IDs
python scripts/test_system.py --camera
```

### "No face detected"
- Ensure good lighting
- Move closer to camera
- Face should be centered
- Remove glasses/hats if possible

### "Slow performance"
- Close other applications
- Reduce camera resolution in config
- Check CPU usage

### "Email not sending"
- Verify Gmail app password
- Check `config/.env` file
- Test with: `python scripts/test_system.py --email`

---

## 📈 Next Steps

1. **Enroll multiple people** to test the system
2. **Tune the threshold** based on your security needs
3. **Review the logs** to analyze performance
4. **Configure email alerts** for production use
5. **Read the full README** for advanced features

---

## 🎓 For Academic Use

### Collect Data for Analysis

```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager()

# Export logs for analysis
from src.alerting.logger import PerformanceLogger
logger = PerformanceLogger()
logger.export_recent_logs(limit=1000, output_path="my_experiment.csv")
```

### Benchmark Different Thresholds

```python
from src.verification.face_matcher import FaceMatcher

matcher = FaceMatcher()

# Collect test embeddings with labels
test_data = [
    # (driver_id, embedding, is_genuine)
]

results = matcher.benchmark_threshold(test_data)
print(f"Optimal threshold: {results['optimal_threshold']}")
```

### Measure Performance

```bash
# Benchmark embedding generation speed
python scripts/test_system.py --benchmark
```

---

## 💡 Pro Tips

1. **Good Lighting**: Natural daylight works best
2. **Face Position**: Center your face, fill 20-50% of frame
3. **Consistency**: Enroll in similar lighting to verification
4. **Multiple Enrollments**: Enroll same person multiple times for better accuracy
5. **Threshold Tuning**: Start with 0.6, adjust based on results

---

## 📞 Need Help?

1. Check the **Troubleshooting** section above
2. Run **system tests**: `python scripts/test_system.py --all`
3. Review **logs**: `data/logs/verification_log.csv`
4. Read the **full README.md**

---

**Ready to start?** Run the commands in order and you'll be up and running in 5 minutes! 🚀
