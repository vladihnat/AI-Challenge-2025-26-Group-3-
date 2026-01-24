"""
POLLINATOR DATASET SPLIT SCRIPT

PURPOSE:
This script processes image data from H5 files and CSV/JPG sources to create 
a balanced Train/Test split for the AI Pollinator Challenge.

KEY FEATURES:
1. DATA UNIFICATION:
   - Harmonizes labels: 0 (No Visitor), 1-10 mapped to 1 (Visitor), -1 removed.
   - Merges sequences from structured H5 files and individual sampled JPGs.

2. SEQUENCE-BASED LEAKAGE PREVENTION:
   - Groups images by 'videoName' (for JPGs) or 'h5_filename' (for H5).
   - Ensures all images from a single video sequence stay together (either in Train or Test).

3. COMBINATORIAL OPTIMIZATION SPLIT (Greedy/Random Search):
   - Instead of simple random splitting, the script performs a 60-second 
     intensive search across hundreds of thousands of combinations.
   - TARGETS: 
     * 25% of the total insect stock for the Test set.
     * 25% of the total image volume for the Test set.
     * Perfect distribution matching (same internal % of insects in both sets).

FINAL DATASET STATISTICS (Current Version):
---------------------------------------------------------------------------
TRAIN SET:
  - Total Images:  30,670 (75.6% of global volume)
  - Insect Images: 833    (75.0% of insect stock)
  - Internal Balance: 97.28% Class 0 | 2.72% Class 1

TEST SET:
  - Total Images:  9,878  (24.4% of global volume)
  - Insect Images: 278    (25.0% of insect stock)
  - Internal Balance: 97.19% Class 0 | 2.81% Class 1
---------------------------------------------------------------------------
Result: Near-perfect stratification preserved while keeping sequences intact.
"""

import os
# ... reste du code ...

import os
import h5py
import numpy as np
import pandas as pd
import cv2
import random
import time
from tqdm import tqdm

# --- CONFIGURATION ---
H5_DIR = "Pollinator-Data-Sample_5pct_fraisiers_hdf5_observed_series0"
CSV_PATH = "sampled_imgs_10classesmetadata/sampled_imgs_10classes_metadata.csv"
JPG_ROOT = "sampled_imgs_10classesmetadata" 
IMG_SIZE = (224, 224) 

def process_label(lbl):
    try:
        val = int(lbl)
        if val == 0: return 0
        if 1 <= val <= 10: return 1
    except: pass
    return -1

def create_empty_data_h5(filename):
    f = h5py.File(filename, 'w')
    f.create_dataset("images", shape=(0, *IMG_SIZE, 3), maxshape=(None, *IMG_SIZE, 3), 
                     dtype='uint8', compression="gzip", compression_opts=4)
    return f

# --- 1. COLLECTING SEQUENCES ---
sequences = {} 
global_0, global_1 = 0, 0

print("[*] Scanning sources for analysis...")

# CSV Scan (JPG images)
print("[*] Scanning CSV...")
df_jpg = pd.read_csv(CSV_PATH)
for _, row in df_jpg.iterrows():
    l_bin = process_label(row['label'])
    if l_bin != -1:
        # Using videoName to identify the sequence
        seq_id = str(row['videoName']) if 'videoName' in row else os.path.dirname(row['jpg_path'])
        if seq_id not in sequences: sequences[seq_id] = []
        sequences[seq_id].append({'src': 'jpg', 'path': row['jpg_path'], 'label': l_bin})
        if l_bin == 0: global_0 += 1
        else: global_1 += 1

# H5 Scan (H5 images)
print(f"[*] Scanning H5 folder...")
h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
for h5_name in h5_files:
    with h5py.File(os.path.join(H5_DIR, h5_name), 'r') as f:
        lbls = f['labels'][:]
        for idx, l in enumerate(lbls):
            # Each H5 file is a unique sequence
            l_bin = process_label(l)
            if l_bin != -1:
                if h5_name not in sequences: sequences[h5_name] = []
                sequences[h5_name].append({'src': 'h5', 'file': h5_name, 'idx': idx, 'label': l_bin})
                if l_bin == 0: global_0 += 1
                else: global_1 += 1

# Grouping images by sequence prevent images from the same sequence being in both Train AND Test to avoid data leakage
seq_ids = list(sequences.keys())
seq_data = [{'id': sid, 
             'n0': sum(1 for r in sequences[sid] if r['label'] == 0), 
             'n1': sum(1 for r in sequences[sid] if r['label'] == 1),
             'total': len(sequences[sid])} for sid in seq_ids]

# --- 2. SEARCHING FOR THE BEST SPLIT ---
# We want to split sequences into Train and Test sets (75% / 25%) while:
# 1. Ensuring that class 1 (insects) is represented as close as possible to 25% in the Test set
# 2. Ensuring that the total number of images in the Test set is close to 25% of the total stock
# 3. Ensuring that the ratio of class 1 in the Test set is similar to the global ratio
target_ratio = 0.25  # 25% of data in Test set
search_duration = 60  # seconds
best_penalty = float('inf')
best_test_ids = []

print(f"[*] Searching for the best split ({search_duration}s)...")
start_time = time.time()
iters = 0

while (time.time() - start_time) < search_duration:
    iters += 1
    # Testing test set sizes ranging from 20% to 30% of sequences
    k = random.randint(max(1, int(len(seq_ids)*0.20)), int(len(seq_ids)*0.30))
    current_test_ids = random.sample(seq_ids, k)
    
    t0 = sum(s['n0'] for s in seq_data if s['id'] in current_test_ids)
    t1 = sum(s['n1'] for s in seq_data if s['id'] in current_test_ids)
    tt = t0 + t1
    
    r0 = t0 / global_0 if global_0 > 0 else 0
    r1 = t1 / global_1 if global_1 > 0 else 0
    rt = tt / (global_0 + global_1)
    
    # PENALTY CALCULATION
    # 1. Deviation of class 1 from 25% of the stock (Absolute priority)
    p1 = abs(r1 - target_ratio) * 20
    # 2. Deviation of total volume from 25% (Important for file size balancing)
    pt = abs(rt - target_ratio) * 5
    # 3. Distribution leakage (The ratio of class 1 in the Test set should be the same as globally)
    # This ensures that Train and Test have the same % of class 1
    pl = abs(r1 - rt) * 10
    
    total_penalty = p1 + pt + pl
    
    if total_penalty < best_penalty:
        best_penalty = total_penalty
        best_test_ids = current_test_ids

print(f"[✔] Optimization completed ({iters} combinations tested).")

train_refs, test_refs = [], []
for sid in seq_ids:
    if sid in best_test_ids: test_refs.extend(sequences[sid])
    else: train_refs.extend(sequences[sid])

# --- 3. FILE GENERATION ---
def build_dataset(refs, data_h5_name, label_npy_name):
    print(f"\n>>> Creating {data_h5_name} ({len(refs)} images)...")
    h5_f = create_empty_data_h5(data_h5_name)
    final_labels = []
    batch_size = 200
    for i in tqdm(range(0, len(refs), batch_size), desc="Writing Batches"):
        batch = refs[i:i+batch_size]
        imgs, lbls = [], []
        opened_h5 = {}
        for r in batch:
            try:
                if r['src'] == 'h5':
                    if r['file'] not in opened_h5: opened_h5[r['file']] = h5py.File(os.path.join(H5_DIR, r['file']), 'r')
                    img = opened_h5[r['file']]['images'][r['idx']]
                else:
                    img = cv2.imread(os.path.join(JPG_ROOT, r['path']))
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != IMG_SIZE: img = cv2.resize(img, IMG_SIZE)
                imgs.append(img); lbls.append(r['label'])
            except: continue
        for f in opened_h5.values(): f.close()
        if imgs:
            curr = h5_f["images"].shape[0]
            h5_f["images"].resize(curr + len(imgs), axis=0)
            h5_f["images"][curr:curr+len(imgs)] = np.array(imgs)
            final_labels.extend(lbls)
    h5_f.close()
    np.save(label_npy_name, np.array(final_labels))
    return np.array(final_labels)

y_tr = build_dataset(train_refs, "train_data.h5", "train_labels.npy")
y_te = build_dataset(test_refs, "test_data.h5", "test_labels.npy")

# --- 4. SANITY CHECK ---
def print_stats(name, y, g_total, g_pos):
    c0, c1 = np.sum(y == 0), np.sum(y == 1)
    print(f"\nStats {name}:")
    print(f"  Images: {len(y)} ({100*len(y)/g_total:.1f}% of total stock)")
    print(f"  Insects: {c1} ({100*c1/g_pos:.1f}% of insect stock)")
    print(f"  Internal ratio of class 0: {100*c0/len(y):.3f}% of class 0 in this set")
    print(f"  Internal ratio of class 1: {100*c1/len(y):.3f}% of class 1 in this set")

g_t = global_0 + global_1
print_stats("TRAIN", y_tr, g_t, global_1)
print_stats("TEST", y_te, g_t, global_1)