# Forest Cover Type Classifier with LightGBM

## Problem Statement
Predicting forest cover type from cartographic variables collected by the US Forest Service
for the Roosevelt National Forest in northern Colorado. Accurate cover type prediction
supports forest management, fire risk assessment, and biodiversity monitoring.
The dataset presents a multi-class classification challenge with 7 cover types and natural class imbalance.

## Project Objectives
- Perform comprehensive EDA to understand feature distributions and class imbalance
- Engineer 8 domain-informed features to enhance predictive power
- Train and tune a LightGBM classifier using Bayesian optimization (Optuna)
- Handle multi-class imbalance via sample weighting
- Evaluate with macro and micro F1, precision, recall, and confusion matrix
- Compare against XGBoost and Random Forest baselines
- Document full workflow for reproducibility

## Dataset
- Source: `sklearn.datasets.fetch_covtype()`
- Samples: 581,012
- Features: 54 (10 continuous, 4 wilderness area binary, 40 soil type binary)
- Target: 7 cover types (Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz)
- Class imbalance: Classes 1 and 2 dominate (~85% of samples combined)

## Feature Engineering Summary
| Feature | Formula | Hypothesis |
|---|---|---|
| euclidean_dist_to_hydrology | sqrt(H² + V²) | True 3D water proximity |
| hillshade_mean | mean(9am, Noon, 3pm) | Daily solar exposure |
| hillshade_range | 9am - 3pm | Slope orientation proxy |
| elevation_water_level | Elevation - Vertical_Dist_Hydro | Absolute water elevation |
| slope_hydrology_interaction | Slope × H_Dist_Hydro | Terrain-moisture stress |
| human_impact_distance | Roadways + Fire_Points dist | Human footprint proxy |
| aspect_sin | sin(Aspect × π/180) | Circular aspect encoding |
| aspect_cos | cos(Aspect × π/180) | Circular aspect encoding |

## Project Structure
```
Forest-Cover-Type-Classifier-with-LightGBM/
├── notebook/
│   └── forest_cover_classifier.ipynb   # Primary deliverable
├── outputs/
│   ├── figures/                         # All generated plots
│   ├── models/                          # Saved LightGBM model
│   └── reports/                         # Classification report CSV
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py           # Feature engineering module
│   └── utils.py                         # Utility functions
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── environment.yml
└── README.md
```

## Environment Setup

### Option 1: pip
```bash
git clone https://github.com/Rushikesh-5706/Forest-Cover-Type-Classifier-with-LightGBM.git
cd Forest-Cover-Type-Classifier-with-LightGBM
pip install -r requirements.txt
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate forest-cover-lgbm
```

## Running the Notebook
```bash
jupyter notebook notebook/forest_cover_classifier.ipynb
```
Execute all cells top to bottom. The notebook downloads the dataset automatically via sklearn.
All outputs are saved to the outputs/ directory.

## Docker
```bash
docker pull rushi5706/forest-cover-lgbm:latest
docker run -p 8888:8888 rushi5706/forest-cover-lgbm:latest
```
Then open: http://localhost:8888

## Key Findings
- Elevation is the single most important feature by both split and gain importance
- euclidean_dist_to_hydrology ranked in top 5 by gain, validating the engineering hypothesis
- LightGBM tuned achieved significant macro F1 improvement over the untuned baseline
- LightGBM training was dramatically faster than Random Forest with superior accuracy
- Cottonwood/Willow (class 4) was the hardest to classify due to severe imbalance — sample weighting improved its recall

## License
MIT
