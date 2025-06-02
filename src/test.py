"""Module for model testing"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)

def test_model(model_path, test_path):
    """Test the model"""
    logger.info("Testing model...")
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError("Model file not found")
        if not Path(test_path).exists():
            raise FileNotFoundError("Test file not found")
            
        logger.info(f"Using model: {model_path}")
        logger.info(f"Using test file: {test_path}")
        
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)
        
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
            if col not in test_df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        X_test = test_df[required_columns]
        y_test = test_df['price']
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\nTest results:")
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"R² Score: {r2:.6f}")
        logger.info(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} rubles")
        
        # Сохраняем результаты тестирования
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': np.mean(np.abs(y_test - y_pred))
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_path = Path("metrics") / "test_metrics.csv"
        metrics_path.parent.mkdir(exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Test metrics saved to {metrics_path}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_model("models/catboost_regression_v1.pkl", "data/processed/test.csv") 