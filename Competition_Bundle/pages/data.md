# Dataset Description

This is a CODE SUBMISSION competition with INTERNET ACCESS.

The provided data is packaged as HDF5 + NumPy files and is distributed through the Starting Kit (the notebook downloads the files automatically when you run the data-loading cell) or through Codabench.

You may train your model on the provided training set and/or any external data of your choice. During evaluation, your submitted code must produce predictions for the hidden test set.

## Data files

All files are downloaded into the `data/` folder by the Starting Kit.

### `data/train_data.h5`
- HDF5 file containing the training images under the dataset key `images`.

### `data/train_labels.npy`
- NumPy array of binary labels aligned with `train_data.h5`:
  - `0` = no pollinator
  - `1` = pollinator present

### `data/train_metadata.npy`
- NumPy array of sequence IDs aligned with the training samples.
- Use it to split train/validation by sequence (to avoid leakage across highly similar frames).

## How to download / access the data

- **From the Starting Kit:** open the notebook in the **Starting Kit** tab and run the data-loading section.  
  The code will create `./data/` and download the 4 files above automatically.
- **From codabench:** https://www.codabench.org/datasets/download/06cc0732-f176-4726-aa5a-33a7486901f2/
