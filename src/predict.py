import numpy as np
import pandas as pd
import joblib
from src.logger import logger
import os

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "data\DataCGGO.csv"

def load_artifacts():
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    lda_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lda_model.pkl"))
    models = {
        "LogisticRegression": joblib.load(os.path.join(ARTIFACTS_DIR, "LogisticRegression_model.pkl")),
        "DecisionTree": joblib.load(os.path.join(ARTIFACTS_DIR, "DecisionTree_model.pkl")),
        "RandomForest": joblib.load(os.path.join(ARTIFACTS_DIR, "RandomForest_model.pkl"))
    }
    label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))

    logger.info("Artifacts loaded successfully")
    return scaler, lda_model, models, label_encoders

def preprocess_new_data(df, label_encoders):
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
            logger.info(f"Column '{col}' encoded")
    return df

def get_top_features_from_lda(df_columns, lda_model, top_n=20):
    logger.info("Selecting top features using LDA model...")
    ldacoef = np.exp(np.abs(lda_model.coef_)).flatten()
    coef_df = pd.DataFrame({
        "Column Name": df_columns,
        "Coef Value": ldacoef
    })
    top_features = coef_df.nlargest(top_n, "Coef Value")["Column Name"].tolist()
    logger.info(f"Top {top_n} features selected: {top_features}")
    return top_features

def main():
    logger.info("Starting prediction pipeline")
    
    # Load artifacts
    scaler, lda_model, models, label_encoders = load_artifacts()
    
    # Load new data
    df_new = pd.read_csv(DATA_PATH)
    df_new = df_new.head(10)
    logger.info("orginal dataset with target column")
    logger.info(df_new)

    df_new = df_new.iloc[:, :-1]
    logger.info(f"New data shape: {df_new.shape}")
    
    # Preprocess
    df_new = preprocess_new_data(df_new, label_encoders)
    
    # Standardize
    x_new_std = scaler.transform(df_new)
    x_new_std = pd.DataFrame(x_new_std, columns=df_new.columns)
    
    # Get top features using LDA model
    top_features = get_top_features_from_lda(df_new.columns, lda_model, top_n=20)
    x_new_selected = x_new_std[top_features]
    
    # Predict
    for name, model in models.items():
        predictions = model.predict(x_new_selected)
        logger.info(f"{name} predictions (first 10): {predictions[:10]}")
    
    logger.info("Prediction pipeline completed")

if __name__ == "__main__":
    main()
