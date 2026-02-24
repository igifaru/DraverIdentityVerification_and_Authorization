import sqlite3
import pickle
import numpy as np
from scipy.spatial.distance import cosine

db_path = 'data/database/drivers.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get all drivers
c.execute("SELECT driver_id, name, license_number, biometric_embedding FROM drivers ORDER BY driver_id")
rows = c.fetchall()

print("=== ALL ENROLLED DRIVERS ===")
embeddings = []
for row in rows:
    driver_id = row[0]
    name = row[1]
    license_num = row[2]
    embedding = pickle.loads(row[3])
    
    print(f"\nDriver ID: {driver_id}")
    print(f"Name: {name}")
    print(f"License: {license_num}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding mean: {np.mean(embedding):.4f}")
    print(f"Embedding std: {np.std(embedding):.4f}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    embeddings.append((driver_id, name, embedding))

# Check for duplicate embeddings (likely the root cause)
print("\n\n=== SIMILARITY MATRIX (Between All Drivers) ===")
for i, (id1, name1, emb1) in enumerate(embeddings):
    for j, (id2, name2, emb2) in enumerate(embeddings):
        if i < j:  # Only upper triangle
            similarity = 1 - cosine(emb1, emb2)
            print(f"{name1} (ID {id1}) vs {name2} (ID {id2}): {similarity:.4f}")
            if similarity > 0.95 and name1 != name2:
                print(f"  ⚠️  WARNING: High similarity between different people!")

conn.close()
