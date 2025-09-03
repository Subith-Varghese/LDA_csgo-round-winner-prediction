import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.logger import logger

def lda_feature_selection(x_train_std, y_train, top_n=20):
    logger.info("Starting LDA feature selection")

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(x_train_std, y_train)

    logger.info("LDA model fitted on training data")

    # Get feature importance
    ldacoef = np.exp(np.abs(lda.coef_)).flatten()
    logger.info("Computed LDA coefficients for feature importance")

    coef_df = pd.DataFrame({
        'Column Name': x_train_std.columns,
        'Coef Value': ldacoef
    })

    # Select top N features
    top_features = coef_df.nlargest(top_n, 'Coef Value')['Column Name'].tolist()
    logger.info(f"Top {top_n} features selected: {top_features}")

    return top_features, lda
