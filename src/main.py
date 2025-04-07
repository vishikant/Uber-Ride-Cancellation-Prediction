import sys
import os

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    best_model_name, best_model_score, best_model_metrics = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    print(f"Best Model: {best_model_name} with score: {best_model_score}")
    print(f"Detailed Metrics: {best_model_metrics}")

 