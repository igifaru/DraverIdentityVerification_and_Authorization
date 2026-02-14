import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    if hasattr(mp, 'solutions'):
        print("mp.solutions found")
    else:
        print("mp.solutions NOT found")
        # Try direct import
        import mediapipe.python.solutions as solutions
        print("Imported solutions directly")
except Exception as e:
    print(f"Error: {e}")
