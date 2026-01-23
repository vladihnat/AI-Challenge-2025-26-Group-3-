import os
import h5py
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# --- CONFIGURATION ---
H5_DIR = "Pollinator-Data-Sample_5pct_fraisiers_hdf5_observed_series0"
CSV_PATH = "sampled_imgs_10classesmetadata/sampled_imgs_10classes_metadata.csv"
JPG_ROOT = "sampled_imgs_10classesmetadata" 
IMG_SIZE = (224, 224) 

def process_label(lbl):
    """ Filter -1, keep 0, transform 1-10 to 1. """
    try:
        val = int(lbl)
        if val == 0: return 0
        if 1 <= val <= 10: return 1
    except:
        pass
    return -1

def create_empty_data_h5(filename):
    """Initialize the H5 file without metadata."""
    f = h5py.File(filename, 'w')
    f.create_dataset("images", shape=(0, *IMG_SIZE, 3), maxshape=(None, *IMG_SIZE, 3), 
                     dtype='uint8', compression="gzip", compression_opts=4)
    return f

# --- 1. COLLECTING REFERENCES ---
all_refs = []

# CSV Scan (JPG images)
print("[*] Scanning CSV...")
df_jpg = pd.read_csv(CSV_PATH)
for _, row in tqdm(df_jpg.iterrows(), total=len(df_jpg), desc="CSV"):
    l_bin = process_label(row['label'])
    if l_bin != -1:
        # Using videoName to identify the sequence
        seq_id = str(row['videoName']) if 'videoName' in row else os.path.dirname(row['jpg_path'])
        all_refs.append({
            'src': 'jpg', 
            'path': row['jpg_path'], 
            'label': l_bin, 
            'group': seq_id
        })

# H5 Folder Scan
print(f"[*] Scanning H5 folder...")
h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
for h5_name in tqdm(h5_files, desc="H5 Files"):
    with h5py.File(os.path.join(H5_DIR, h5_name), 'r') as f:
        lbls = f['labels'][:]
        for idx, l in enumerate(lbls):
            l_bin = process_label(l)
            if l_bin != -1:
                # Each H5 file is a unique sequence
                all_refs.append({
                    'src': 'h5', 
                    'file': h5_name, 
                    'idx': idx, 
                    'label': l_bin, 
                    'group': h5_name
                })

# --- 2. GROUP SHUFFLE SPLIT (75/25) ---
# Prevent images from the same sequence being in both Train AND Test to avoid data leakage
print(f"[*] Performing Group Shuffle Split...")
groups = [r['group'] for r in all_refs]
labels = [r['label'] for r in all_refs]

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(all_refs, labels, groups))

train_refs = [all_refs[i] for i in train_idx]
test_refs = [all_refs[i] for i in test_idx]

# --- 3. FILE GENERATION ---
def build_dataset(refs, data_h5_name, label_npy_name):
    print(f"\n>>> Building {data_h5_name}...")
    h5_f = create_empty_data_h5(data_h5_name)
    final_labels = []
    
    batch_size = 200
    for i in tqdm(range(0, len(refs), batch_size), desc="Batches"):
        batch = refs[i:i+batch_size]
        imgs, lbls = [], []
        
        # Keep H5 files open only during the batch to optimize RAM usage
        opened_h5 = {}
        
        for r in batch:
            try:
                if r['src'] == 'h5':
                    if r['file'] not in opened_h5:
                        opened_h5[r['file']] = h5py.File(os.path.join(H5_DIR, r['file']), 'r')
                    img = opened_h5[r['file']]['images'][r['idx']]
                else:
                    img = cv2.imread(os.path.join(JPG_ROOT, r['path']))
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img.shape[:2] != IMG_SIZE:
                    img = cv2.resize(img, IMG_SIZE)
                
                imgs.append(img)
                lbls.append(r['label'])
            except:
                continue
        
        for f in opened_h5.values(): f.close()
            
        if imgs:
            # Writing batch to H5
            n_new = len(imgs)
            curr = h5_f["images"].shape[0]
            h5_f["images"].resize(curr + n_new, axis=0)
            h5_f["images"][curr:curr+n_new] = np.array(imgs)
            final_labels.extend(lbls)

    h5_f.close()
    np.save(label_npy_name, np.array(final_labels))
    return np.array(final_labels)

y_tr = build_dataset(train_refs, "train_data.h5", "train_labels.npy")
y_te = build_dataset(test_refs, "test_data.h5", "test_labels.npy")

# --- 4. SANITY CHECK ---
def print_stats(name, y):
    count_0 = np.sum(y == 0)
    count_1 = np.sum(y == 1)
    total = len(y)
    print(f"\nStats {name}:")
    print(f"  Total: {total}")
    print(f"  No Visitor (0): {count_0} ({100*count_0/total:.2f}%)")
    print(f"  Visitor (1):    {count_1} ({100*count_1/total:.2f}%)")

print_stats("TRAIN", y_tr)
print_stats("TEST", y_te)