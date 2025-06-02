"""Module for data preprocessing"""

import os
import glob
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def extract_flat_id(url):
    """Extract flat ID from Cian URL"""
    return url.split('/')[-2]

def preprocess_data():
    """Preprocess the data"""
    logger.info("Starting data preprocessing...")
    try:
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = Path("data/raw")
        raw_files = glob.glob(str(raw_dir / "*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw data files found in data/raw/")
        
        latest_file = max(raw_files, key=os.path.getctime)
        logger.info(f"Processing file: {latest_file}")

        main_dataframe = pd.read_csv(latest_file)
        main_dataframe['url_id'] = main_dataframe['url'].apply(extract_flat_id)
        df = main_dataframe[['url_id', 'total_meters', 'floor', 'floors_count', 'rooms_count', 'price']].set_index('url_id')
        
        df = df.sort_index()
        df = df.dropna()
        df = df[df['price'] < 1000000000]
        
        df["rooms_1"] = df["rooms_count"] == 1
        df["rooms_2"] = df["rooms_count"] == 2
        df["rooms_3"] = df["rooms_count"] == 3
        df["first_floor"] = df["floor"] == 1
        df["last_floor"] = df["floor"] == df["floors_count"]

        df = df[['total_meters', 'floors_count', 'floor', 
                'rooms_1', 'rooms_2', 'rooms_3', 'first_floor', 'last_floor', 'price']]
        
        print("\nДатасет после предобработки:")
        print(df)
        
        logger.info("\nPreprocessed data statistics:")
        logger.info(f"Number of samples after preprocessing: {len(df)}")
        logger.info(f"Price range after preprocessing: {df['price'].min()} - {df['price'].max()}")
        logger.info(f"Average price by floor after preprocessing:")
        logger.info(df.groupby('floor')['price'].mean().to_string())
        
        train_size = int(len(df) * 0.8)  # 80% for training
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
      
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
        logger.info(f"Number of samples in train: {len(train_df)}")
        logger.info(f"Number of samples in test: {len(test_df)}")
        
        return train_path, test_path
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    preprocess_data() 