import os
import sys
import pandas as pd
import pickle
import logging
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/vaishalikant/Downloads/data science project/Uber-Ride-Cancellation-Prediction/src')))

from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join('artifacts', "transformed_data.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data, test_data):
        logging.info("Entered the data transformation method or component")


        try:
            # Identify categorical and numerical columns
            categorical_cols = train_data.select_dtypes(include=['object']).columns
            numerical_cols = train_data.select_dtypes(exclude=['object']).columns

            # Define the preprocessing steps for categorical and numerical columns
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('label_encoder', LabelEncoder()),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ])

            # Combine the preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )

            # Fit and transform the train and test data
            X_train = preprocessor.fit_transform(train_data)
            X_test = preprocessor.transform(test_data)

            # Extract the target variable
            y_train = train_data.iloc[:, -1].values
            y_test = test_data.iloc[:, -1].values

            # Save the transformed data to a pickle file
            with open(self.transformation_config.transformed_data_path, 'wb') as f:
                pickle.dump((X_train, y_train, X_test, y_test), f)
            logging.info("Transformed data saved to pickle file")

            return X_train, y_train, X_test, y_test

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)