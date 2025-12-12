# Music Genre Classification — GTZAN & Spotify (MusicRecNet Baseline)

This repository implements baseline models (2D CNN, LSTM, SVM) trained on GTZAN and Spotify music-genre datasets using pre-generated mel-spectrogram images. It reproduces the MusicRecNet architecture from the paper and provides scripts for training, evaluation, feature extraction, ensemble prediction, and visualization. Paper: Music genre classification and music recommendation by using deep learning, A. Elbir✉ and N. Aydin.

## Features
- Train MusicRecNet 2D CNN on GTZAN and Spotify datasets
- Train LSTM on Spotify dataset
- Train SVM classifier on extracted CNN features
- Ensemble predictions across all three architectures
- Generate confusion matrices for all models
- Extract Dense_2 (128-d) embeddings
- Visualizations: radar charts, probability heatmaps, training curves, four-panel probability plots, t-SNE embeddings

## Project structure
```
project/
│
├── scrape_data/
│   ├── everynoise_scraper.py
│   └── download_playlists.py
│
├── train_musicrecnet_kaggle.py
├── train_musicrecnet_spotify.py
├── train_musicrecnetLSTM_spotify.py
├── train_svm.py
├── extract_features.py
├── ensemble_pred.py
├── confusion_matrix_eval.py
├── tsne_visualization.py
│
├── musicrecnet.py
└── musicrecnet_lstm.py
│
├── gtzan_kaggle_dataset.py
└── spotify_dataset.py
│
└── Data/
    └── images_original/
         ├── blues/
         ├── classical/
         ├── country/
         └── ...
```

## Requirements
Install the required Python packages:
```bash
pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn tqdm joblib
```

## Usage

### 1. Data Acquisition
For Spotify dataset, run the scraping scripts (GTZAN dataset images should already be present):
```bash
python everynoise_scraper.py
python download_playlists.py
```

### 2. Train 2D CNN
Train MusicRecNet on Spotify data (or modify for GTZAN):
```bash
python train_musicrecnet_spotify.py
```
Output:
- `musicrecnet_best.pt`
- `training_curve.png`

### 3. Train LSTM
Train LSTM on Spotify dataset:
```bash
python train_musicrecnetLSTM_spotify.py
```
Output:
- `musicrecnet_lstm_best.pt`

### 4. Extract CNN Features
Extract 128-d Dense_2 embeddings for SVM training:
```bash
python extract_features.py
```
Outputs:
- `dense2_features.npy`
- `dense2_labels.npy`

### 5. Train SVM
Train SVM classifier on extracted CNN features:
```bash
python train_svm.py
```
Output:
- `svm_dense2.joblib`

### 6. Ensemble Prediction
Run ensemble predictions across all three architectures:
```bash
python ensemble_pred.py
```
Outputs:
- `ensemble_radar.png`
- `probability_heatmap.png`
- `four_panel_probs.png`

### 7. Generate Confusion Matrices
Evaluate all four models and generate confusion matrices:
```bash
python confusion_matrix_eval.py
```
Output:
- `confusion_matrices_all_4.png`

## Notes
- GTZAN dataset mel-spectrogram images must be present in `Data/images_original/`; no additional audio preprocessing required.
- Spotify dataset requires running scraping scripts first to download playlist data.
- Modify dataset paths in scripts if needed.
- Adjust hyperparameters in training scripts as required.