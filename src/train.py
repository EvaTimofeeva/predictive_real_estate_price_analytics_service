"""Module for model training"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)

MODEL_NAME = "catboost_regression_v1.pkl"

def train_model(train_path):
    """Train the model"""
    logger.info("Training model...")
    try:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if not Path(train_path).exists():
            raise FileNotFoundError("Train file not found")
        
        logger.info(f"Using train file: {train_path}")

        train_df = pd.read_csv(train_path)

        required_columns = [
            "total_meters",
            "floors_count",
            "floor",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor"
        ]
        
        for col in required_columns:
            if col not in train_df.columns:
                raise ValueError(f"Missing required column: {col}")

        X = train_df[required_columns]
        y = train_df['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
   
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.01,
            depth=5,
            l2_leaf_reg=2,
            random_seed=42,
            verbose=100
        )
        model.fit(X_train, y_train)
       
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
    
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"R² Score: {r2:.6f}")
        logger.info(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} rubles")

        # Выводим важность признаков
        feature_importance = pd.DataFrame({
            'Feature': required_columns,
            'Importance': model.get_feature_importance()
        })
        logger.info("\nFeature importance:")
        logger.info(feature_importance.to_string())

        model_path = models_dir / MODEL_NAME
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    train_model("data/processed/train.csv") 