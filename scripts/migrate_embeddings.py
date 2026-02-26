"""
migrate_embeddings.py
=====================
Re-generate all existing driver embeddings using the correct pipeline:
  - Raw BGR face crop  ->  DeepFace FaceNet  ->  L2-normalise  ->  Store

Run this ONCE after pulling the embedding pipeline fix:
    (venv) python migrate_embeddings.py

Drivers without a valid photo_path are skipped with a warning.
"""

import sys
import pickle
import numpy as np
import cv2
import psycopg2
import psycopg2.extras
from pathlib import Path

sys.path.insert(0, 'src')
from utils.config import config
from enrollment.face_processor import FaceProcessor

print("=" * 60)
print("EMBEDDING MIGRATION")
print("Re-normalising all stored driver embeddings")
print("=" * 60)

conn = psycopg2.connect(config.database_url)
cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("SELECT driver_id, name, biometric_embedding, photo_path FROM drivers ORDER BY driver_id")
drivers = cur.fetchall()
print(f"\nFound {len(drivers)} driver(s) in database.\n")

processor = FaceProcessor()

updated = 0
skipped = 0

for row in drivers:
    did   = row['driver_id']
    name  = row['name']
    photo = row['photo_path']

    print(f"[{did}] {name}")

    # ------------------------------------------------------------------ #
    # Strategy A: re-generate from saved enrollment photo (best quality)   #
    # ------------------------------------------------------------------ #
    if photo and Path(photo).is_file():
        frame = cv2.imread(str(photo))
        if frame is None:
            print(f"  -> Could not read photo: {photo} — falling back to re-normalise")
        else:
            face_crop, status = processor.process_for_enrollment(frame)
            if face_crop is not None:
                embedding = processor.generate_embedding(face_crop)
                if embedding is not None and not np.isnan(embedding).any():
                    blob = psycopg2.Binary(pickle.dumps(embedding))
                    cur.execute(
                        "UPDATE drivers SET biometric_embedding = %s WHERE driver_id = %s",
                        (blob, did)
                    )
                    conn.commit()
                    print(f"  -> Re-generated from photo  norm={np.linalg.norm(embedding):.6f}  dim={embedding.shape[0]}")
                    updated += 1
                    continue
            print(f"  -> Face detection failed on photo ({status}) — falling back to re-normalise")

    # ------------------------------------------------------------------ #
    # Strategy B: L2-normalise the existing stored embedding in place      #
    # (no photo available or face detection failed on the photo)           #
    # ------------------------------------------------------------------ #
    existing = pickle.loads(bytes(row['biometric_embedding']))
    norm = np.linalg.norm(existing)
    if norm < 1e-6:
        print(f"  -> SKIP: degenerate embedding (norm={norm:.6f})")
        skipped += 1
        continue

    normalised = (existing / norm).astype(np.float32)
    blob = psycopg2.Binary(pickle.dumps(normalised))
    cur.execute(
        "UPDATE drivers SET biometric_embedding = %s WHERE driver_id = %s",
        (blob, did)
    )
    conn.commit()
    print(f"  -> Normalised in-place  norm_before={norm:.4f}  norm_after={np.linalg.norm(normalised):.6f}")
    updated += 1

cur.close()
conn.close()

print(f"\n{'='*60}")
print(f"Migration complete: {updated} updated, {skipped} skipped")
print(f"{'='*60}")
print("\nRestart the server so the verification engine reloads embeddings.")
