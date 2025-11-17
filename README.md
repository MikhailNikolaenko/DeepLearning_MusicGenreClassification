# Music Genre Classification — GTZAN (MusicRecNet Baseline)

This repository implements a baseline 2D CNN (MusicRecNet) trained on the GTZAN music-genre dataset using pre-generated mel-spectrogram images (from the Kaggle dataset). It reproduces the architecture from the paper and provides scripts for training, evaluation, feature extraction and visualization.

## Features
- Train baseline MusicRecNet CNN
- Plot training curves
- Generate confusion matrix
- Extract Dense_2 (128-d) embeddings
- t-SNE visualization of embeddings

## Project structure
```
project/
│
├── train_musicrecnet_kaggle.py
├── extract_features_kaggle.py
├── confusion_matrix_eval.py
├── tsne_visualization.py
├── gtzan_kaggle_dataset.py
├── musicrecnet.py
│
└── Data/
    └── images_original/
         ├── blues/
         ├── classical/
         ├── country/
         ├── ...
```

## Requirements
Install the required Python packages:
```bash
pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn tqdm
```

## Usage

### 1. Training
Train the model:
```bash
python train_musicrecnet_kaggle.py
```
Outputs:
- `musicrecnet_best.pt` — best model checkpoint
- `training_curve.png` — training/validation loss & accuracy plot

### 2. Generate confusion matrix
After training, evaluate and save confusion matrix:
```bash
python confusion_matrix_eval.py
```
Output:
- `confusion_matrix.png`

### 3. Extract Dense_2 features
Extract 128-d feature vectors from the Dense_2 layer:
```bash
python extract_features_kaggle.py
```
Outputs:
- `dense2_features.npy`
- `dense2_labels.npy`

### 4. t-SNE visualization
Create a t-SNE plot from extracted Dense_2 features:
```bash
python tsne_visualization.py
```
Output:
- `tsne_dense2.png`

## Notes
- All scripts assume the dataset is located at `Data/images_original/`. If your dataset path differs, update the path in the scripts.
- The Kaggle GTZAN dataset included here already contains mel-spectrogram PNGs; no additional audio preprocessing is required.
- Adjust hyperparameters in `train_musicrecnet_kaggle.py` as needed.

