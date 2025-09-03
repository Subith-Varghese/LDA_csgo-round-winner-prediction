# 🏆 CSGO Round Winner Prediction 🎯

Predict the winning team (CT or T) in Counter-Strike: Global Offensive (CSGO) rounds using Machine Learning.
This project leverages LDA-based feature selection and compares Logistic Regression, Decision Tree, and Random Forest classifiers to deliver accurate predictions.
We implemented data preprocessing, feature selection, model training, evaluation, and prediction pipelines to build an end-to-end ML solution.

---
📂 Dataset
- File: DataCGGO.csv
- Rows: 122,410
- Columns: 97
- Target Column: round_winner
- Includes player stats, weapons, grenades, money, and map details.
---

🗂 Project Structure
```
CSGO_Round_Winner_Prediction/
│── data/
│   └── DataCGGO.csv                # Dataset
│
│── notebooks/
│   └── model_notebook.ipynb        # Initial EDA & model experiments
│
│── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, encoding, scaling
│   ├── feature_selection.py        # LDA-based feature selection
│   ├── model_training.py           # Train models & save artifacts
│   ├── model_evaluation.py         # Evaluate models & generate reports
│   ├── predict.py                  # Predict on new data
│   ├── logger.py                  # Central logging utility
│
│── artifacts/
│   ├── lda_model.pkl              # Saved LDA model
│   ├── scaler.pkl                 # StandardScaler
│   ├── label_encoders.pkl         # Encoders for categorical data
│   ├── LogisticRegression_model.pkl
│   ├── DecisionTree_model.pkl
│   ├── RandomForest_model.pkl
│
│── logs/                          # Logs for debugging & monitoring
│── main.py                        # Pipeline entry point
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation
│── .gitignore                     # Ignore unnecessary files

```
---

## Project Workflow

### Step 1 — Data Preprocessing
File: src/data_preprocessing.py
  - Load dataset
  - Handle duplicates
  - Label encode bomb_planted & map
  - Standardize features using StandardScaler
  - Save label encoders & scaler in artifacts/

---

### Step 2 — Feature Selection
File: src/feature_selection.py
  - Apply Linear Discriminant Analysis (LDA)
  - Compute feature importance using coefficients
  - Select top 20 most important features for modeling

--- 

### Step 3 — Model Training
File: src/model_training.py
- Train 3 ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Save trained models into artifacts/
  
---

### Step 4 — Model Evaluation

File: src/model_evaluation.py
- Evaluate models using:
  - Accuracy
  - Classification Report
  - Confusion Matrix
- Random Forest achieved the best performance (~85%)

---
### Step 5 — Prediction

File: src/predict.py
- Loads:
  - Trained models
  - LDA model
  - Scaler
  - Label encoders
- Preprocesses new unseen data
- Selects top 20 features
- Generates predictions for each model

---

## ⚙️ Installation

```
# Clone the repository
git clone https://github.com/your-username/CSGO_Round_Winner_Prediction.git

# Navigate to the project directory
cd CSGO_Round_Winner_Prediction

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---
## 🚀 Model Training

Run the entire ML pipeline:

```
python main.py
```
---
This will:
  - Preprocess data
  - Perform LDA-based feature selection
  - Train all models
  - Evaluate model performance
  - Save trained models & artifacts in artifacts/

---

## Prediction On New Data

Once models are trained, predict winners for new data:
```
python src/predict.py
```

This will:
  - Load the trained models & artifacts
  - Preprocess new data
  - Select top 20 LDA features
  - Generate predictions for Logistic Regression, Decision Tree, and Random Forest

---
## Results

| Model               | Accuracy    |
| ------------------- | ----------- |
| Logistic Regression | **75.8%**   |
| Decision Tree       | **80.7%**   |
| Random Forest       | **85.4%** ✅ |
