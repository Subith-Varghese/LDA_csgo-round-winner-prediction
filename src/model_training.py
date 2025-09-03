import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.logger import logger
import os

def train_models(x_train, y_train, artifacts_path="artifacts"):
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    trained_models = {}
    logger.info("Starting model training")
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model

        model_file = os.path.join(artifacts_path, f"{name}_model.pkl")
        joblib.dump(model, model_file)
        logger.info(f"{name} saved to {model_file}")

    logger.info("All models trained and saved successfully")
    return trained_models
