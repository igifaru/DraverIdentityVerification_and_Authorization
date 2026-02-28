"""
Diagnostic: list all enrolled drivers and compute pairwise cosine similarity.
Run from project root: python scripts/check_drivers.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from database.db_manager import DatabaseManager

db = DatabaseManager()
rows = db.get_all_embeddings()   # [(driver_id, name, embedding), ...]

print(f"\n{'='*60}")
print(f"  ENROLLED DRIVERS: {len(rows)}")
print(f"{'='*60}")
for did, name, emb in rows:
    norm = np.linalg.norm(emb)
    print(f"  ID={did:3d}  name={name!r:20s}  emb_dim={emb.shape[0]}  norm={norm:.6f}")

print(f"\n{'='*60}")
print("  PAIRWISE COSINE SIMILARITY (dot product of unit vectors)")
print(f"{'='*60}")
if len(rows) < 2:
    print("  Only one driver enrolled — no pairwise comparison possible.")
else:
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            id_a, name_a, emb_a = rows[i]
            id_b, name_b, emb_b = rows[j]
            u_a = emb_a / np.linalg.norm(emb_a)
            u_b = emb_b / np.linalg.norm(emb_b)
            sim = float(np.dot(u_a, u_b))
            flag = "  ⚠ TOO SIMILAR" if sim > 0.85 else ""
            print(f"  {name_a!r:20s} vs {name_b!r:20s}  sim={sim:.4f}{flag}")

print()
