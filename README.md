# Satellite Imagery–Based Property Valuation (Multimodal Machine Learning)

## Project Overview

This project implements a multimodal regression system to predict residential property prices by combining traditional tabular real-estate data with satellite imagery. The motivation is to enhance conventional pricing models by incorporating visual neighborhood context such as greenery, road density, urban layout, and proximity to water bodies.

The solution integrates structured numerical features with image-based features extracted using a convolutional neural network (CNN), resulting in a more comprehensive valuation framework.

---

## Objectives

- Develop a tabular-only baseline model for house price prediction
- Programmatically acquire satellite images using latitude and longitude
- Train a multimodal deep learning model combining tabular and image data
- Compare performance between tabular-only and multimodal approaches
- Generate a final prediction file suitable for submission

---

## Modeling Strategy

### Tabular Baseline Model
- Algorithm: HistGradientBoostingRegressor
- Input: Numerical and categorical tabular features
- Target variable: log(price + 1)
- Purpose: Establish a baseline for comparison

### Multimodal Model
- Image branch: Pretrained ResNet-18 (ImageNet weights)
- Tabular branch: Multi-layer perceptron (MLP) on standardized numeric features
- Fusion: Concatenation of image and tabular embeddings
- Output: Regression head predicting log(price + 1)

---

## Project Structure

project_root/
│
├── data/
│   ├── raw/
│   │   ├── train(1).xlsx
│   │   └── test2.xlsx
│   └── images/
│       ├── train/
│       └── test/
│
├── src/
│   ├── data_fetcher.py
│   ├── train_tabular_baseline.py
│   ├── train_multimodal.py
│   └── predict.py
│
├── notebooks/
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── outputs/
│   ├── preds/
│   │   ├── tabular_baseline.csv
│   │   └── submission.csv
│   └── models/
│       └── best_multimodal.pt
│
└── README.md

---

## Dataset Description

- Source: Kaggle House Sales Dataset
- Files used:
  - train(1).xlsx (includes target price)
  - test2.xlsx
- Key attributes:
  - price (target variable)
  - bedrooms, bathrooms
  - sqft_living, sqft_lot
  - latitude and longitude
  - condition, grade, view, waterfront

---

## Satellite Image Collection

Satellite images are fetched using geographic coordinates (latitude and longitude). Each image corresponds to a single property and captures neighborhood-level context.

Images are stored as:
- data/images/train/{id}.png
- data/images/test/{id}.png

Image characteristics:
- Resolution: 256 × 256
- View type: Satellite
- Zoom level: Neighborhood scale

---

## Environment Setup

Create and activate a conda environment:

conda create -n housegpu python=3.10  
conda activate housegpu  

Install dependencies:

pip install pandas numpy scikit-learn openpyxl requests pillow tqdm matplotlib torch torchvision

---

## How to Run the Project

### Step 1: Train Tabular Baseline

python src/train_tabular_baseline.py

Output:
- outputs/preds/tabular_baseline.csv

---

### Step 2: Download Satellite Images

Set API key (example):

set GOOGLE_MAPS_API_KEY=YOUR_API_KEY

Run:

python src/data_fetcher.py

---

### Step 3: Train Multimodal Model

python src/train_multimodal.py

Output:
- outputs/models/best_multimodal.pt

Validation RMSE is printed during training.

---

### Step 4: Generate Final Predictions

python src/predict.py

Output:
- outputs/preds/submission.csv

Submission format:
id, predicted_price

---

## Model Performance

- Best validation RMSE (log scale): approximately 0.29
- Evaluation performed on log(price + 1) to reduce skew and stabilize variance

---

## Notebooks Included

### preprocessing.ipynb
- Dataset loading and inspection
- Price distribution analysis
- Log transformation of the target variable
- Visualization of sample satellite images

### model_training.ipynb
- Train–validation split
- Model architecture definition
- Training loop (demonstrative epochs)
- Validation RMSE reporting

---

## Submission Artifacts

- Final prediction CSV file
- Image fetching pipeline
- Training and inference scripts
- Reproducible notebooks
- This README file

---

## Author

Devanshi Jolhe
Satellite Imagery–Based Property Valuation Project