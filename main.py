import os
from src.data_preprocessing import load_and_preprocess_data
from src.feature_selection import lda_feature_selection
from src.model_training import train_models
from src.model_evaluation import evaluate_models
import joblib
from src.logger import logger 

DATA_PATH = "data/DataCGGO.csv"
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    logger.info("========== Starting CSGO Round Winner Prediction Pipeline ==========")

    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    df, x_train_std, x_test_std, y_train, y_test,scaler = load_and_preprocess_data(DATA_PATH)

    # Step 2: LDA feature selection
    logger.info("Step 2: Performing LDA feature selection")
    top_features, lda_model = lda_feature_selection(x_train_std, y_train, top_n=20)

    # Step 3: Keep only selected features
    logger.info("Step 3: Selecting top features from training and test sets")
    x_train_selected = x_train_std[top_features]
    x_test_selected = x_test_std[top_features]

    # Step 4: Train models
    logger.info("Step 4: Training models")
    trained_models = train_models(x_train_selected, y_train, ARTIFACTS_DIR)

    # Step 5: Evaluate models
    logger.info("Step 5: Evaluating models")
    results = evaluate_models(trained_models, x_test_selected, y_test)

    for model_name, metrics in results.items():
        logger.info(f"==== {model_name} ====")
        logger.info(f"Classification Report:\n{metrics['classification_report']}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    # Step 6: Save LDA and Scaler
    logger.info("Step 6: Saving LDA model and scaler")
    lda_file = os.path.join(ARTIFACTS_DIR, "lda_model.pkl") 
    joblib.dump(lda_model,lda_file)
    logger.info(f"LDA model saved to {lda_file}")

    scaler_file = os.path.join(ARTIFACTS_DIR, "scaler.pkl") 
    joblib.dump(scaler,scaler_file)
    logger.info(f"Scaler saved to {scaler_file}")
    logger.info("========== Pipeline completed successfully ==========")



if __name__ == "__main__":
    main()
