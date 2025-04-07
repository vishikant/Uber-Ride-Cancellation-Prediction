import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vaishalikant/Downloads/data science project/Uber-Ride-Cancellation-Prediction/src')))

from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('/Users/vaishalikant/Downloads/data science project/Uber-Ride-Cancellation-Prediction/Ride_data_final.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            # Further split the temporary set into validation and test sets
            val_set, val_test_set = train_test_split(test_set, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return train_set, test_set
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)