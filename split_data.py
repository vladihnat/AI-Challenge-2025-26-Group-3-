import os
import h5py
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# This script processes image data from JPG files and H5 files,
# splits them into training and testing sets while ensuring that
# images from the same group (video sequence) remain in the same set,
# and saves the resulting datasets into new H5 files along with their labels in NPY format

# USE THIS SCRIPT ONLY TO READ THE DATA GIVEN BY THE ORGANIZERS AND CREATE TRAIN/TEST SPLITS.
# DO NOT USE THIS SCRIPT TO READ DATA FROM OUR OWN COMPETITION.

# --- CONFIGURATION ---
H5_DIR = "Pollinator-Data-Sample_5pct_fraisiers_hdf5_observed_series0"
CSV_PATH = "sampled_imgs_10classesmetadata/sampled_imgs_10classes_metadata.csv"
IMG_SIZE = (224, 224) 

def process_label(lbl):
    """Binary labeling: 0 remains 0, 1-10 becomes 1 (visit)."""
    if lbl == 0: return 0
    if 1 <= lbl <= 10: return 1
    return -1

def create_empty_data_h5(filename):
    """Initialize an empty H5 file for storing images."""
    f = h5py.File(filename, 'w')
    f.create_dataset("images", shape=(0, *IMG_SIZE, 3), maxshape=(None, *IMG_SIZE, 3), 
                     dtype='uint8', compression="gzip", compression_opts=4)
    return f

def append_to_files(h5_file, label_list, imgs, lbls):
    """Add only images and labels to the respective files."""
    n_new = len(imgs)
    curr_size = h5_file["images"].shape[0]
    new_size = curr_size + n_new
    
    h5_file["images"].resize(new_size, axis=0)
    h5_file["images"][curr_size:new_size] = imgs
    label_list.extend(lbls)

# --- 1. COLLECTING REFERENCES TO ALL IMAGES ---
all_refs = []

print("[*] Scanning CSV...")
df_jpg = pd.read_csv(CSV_PATH)
for _, row in tqdm(df_jpg.iterrows(), total=len(df_jpg), desc="Processing CSV"):
    l_bin = process_label(row['label'])
    if l_bin != -1:
        # Group is defined by parent directory to avoid video sequence leakage
        group_id = os.path.dirname(row['jpg_path']) 
        all_refs.append({'src': 'jpg', 'path': row['jpg_path'], 'label': l_bin, 'group': group_id})

print(f"[*] Scanning H5 directory...")
h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
for h5_name in tqdm(h5_files, desc="Processing H5 files"):
    with h5py.File(os.path.join(H5_DIR, h5_name), 'r') as f:
        lbls = f['labels'][:]
        valid_idx = np.where(lbls != -1)[0]
        for idx in valid_idx:
            # Group is the filename (one file = one sequence)
            all_refs.append({'src': 'h5', 'file': h5_name, 'idx': idx, 
                             'label': process_label(lbls[idx]), 'group': h5_name})

# --- 2. GROUP SHUFFLE SPLIT (75/25) ---
print(f"[*] Total images collected: {len(all_refs)}")
groups = [r['group'] for r in all_refs]
labels = [r['label'] for r in all_refs]

# GroupShuffleSplit ensures images from the same video sequence stay in the same split
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(all_refs, labels, groups))

train_refs = [all_refs[i] for i in train_idx]
test_refs = [all_refs[i] for i in test_idx]

# --- 3. GENERATING FILES ---
def build_dataset(refs, data_h5_name, label_npy_name):
    print(f"\n>>> Creating {data_h5_name}...")
    h5_f = create_empty_data_h5(data_h5_name)
    final_labels = []
    
    batch_size = 100
    for i in tqdm(range(0, len(refs), batch_size), desc=f"Building {data_h5_name}"):
        batch = refs[i:i+batch_size]
        imgs, lbls = [], []
        
        for r in batch:
            if r['src'] == 'h5':
                # Open H5 source file
                with h5py.File(os.path.join(H5_DIR, r['file']), 'r') as f_src:
                    img = f_src['images'][r['idx']]
                    if img.shape[:2] != IMG_SIZE: 
                        img = cv2.resize(img, IMG_SIZE)
                    imgs.append(img)
                    lbls.append(r['label'])
            else:
                # Load JPG image
                img_path = os.path.join("sampled_imgs_10classesmetadata", r['path'])
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)
                    imgs.append(img)
                    lbls.append(r['label'])
        
        if imgs:
            append_to_files(h5_f, final_labels, np.array(imgs), lbls)

    h5_f.close()
    np.save(label_npy_name, np.array(final_labels))
    print(f"[OK] {data_h5_name} and {label_npy_name} saved successfully.")

# Processing both splits
build_dataset(train_refs, "train_data.h5", "train_labels.npy")
build_dataset(test_refs, "test_data.h5", "test_labels.npy")

print("\n--- DATASET GENERATION COMPLETE ---")