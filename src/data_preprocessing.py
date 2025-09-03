import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from src.logger import logger

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_and_preprocess_data(data_path):
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Initial dataset shape: {df.shape}")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Label encoding
    encoders = {}
    label_cols = ['bomb_planted', 'map']
    
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    encoder_file = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
    joblib.dump(encoders,encoder_file)
    logger.info(f"Saved label encoders to {encoder_file}")

    # Split features & target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    logger.info(f"Split data: x_train={x_train.shape}, x_test={x_test.shape}")

    # Standardization
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_test_std = scaler.transform(x_test)

    # Convert back to DataFrame so columns & indices are preserved
    x_train_std = pd.DataFrame(x_train_std, columns=x_train.columns, index=x_train.index)
    x_test_std = pd.DataFrame(x_test_std, columns=x_test.columns, index=x_test.index)

    logger.info("Standardization completed on train and test sets")
    return df, x_train_std, x_test_std, y_train, y_test,scaler
