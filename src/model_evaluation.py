from sklearn.metrics import classification_report, confusion_matrix
from src.logger import logger

def evaluate_models(models, x_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] =  {
            "classification_report": report,
            "confusion_matrix": cm
        }
    logger.info("Model evaluation completed")
    return results
