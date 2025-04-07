import os
import sys
from dataclasses import dataclass
from math import sqrt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Initializing model training pipeline...")
        except Exception as e:
            logging.error("Error in model trainer")
            raise CustomException(e, sys)
           
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet Regression": ElasticNet(),
            "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, reg_alpha=1, reg_lambda=1, random_state=42),
            "CatBoost": CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=0),
            "KNN": KNeighborsRegressor()
        }

        # Step 2: Tuning Regularized Models (Ridge, Lasso, ElasticNet)
        tuned_models = {}

        logging.info("Tuning Ridge Regression...")
        ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0]}
        ridge = RandomizedSearchCV(Ridge(), ridge_params, scoring='r2', cv=5)
        ridge.fit(X_train, y_train)
        tuned_models["Ridge Regression"] = ridge.best_estimator_

        logging.info("Tuning Lasso Regression...")
        lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
        lasso = RandomizedSearchCV(Lasso(), lasso_params, scoring='r2', cv=5)
        lasso.fit(X_train, y_train)
        tuned_models["Lasso Regression"] = lasso.best_estimator_

        logging.info("Tuning ElasticNet Regression...")
        elastic_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.7, 0.9]
        }
        elastic = RandomizedSearchCV(ElasticNet(), elastic_params, scoring='r2', cv=5)
        elastic.fit(X_train, y_train)
        tuned_models["ElasticNet Regression"] = elastic.best_estimator_

        # Replace default models with tuned versions
        models.update(tuned_models)

        # Step 3: Evaluate all models
        model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
        logging.info(f"Evaluation Report: {model_report}")

        # Step 4: Find best model (excluding overfitted ones)
        best_model_name = None
        best_model_score = -float("inf")
        best_model = None

        for name, metrics in model_report.items():
            if metrics['r2'] < 0.999:
                if metrics['r2'] > best_model_score:
                    best_model_score = metrics['r2']
                    best_model_name = name
                    best_model = models[name]

        logging.info(f"Best Model Selected: {best_model_name} (R²: {best_model_score:.3f})")

        # Step 5: Save best model
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

        return best_model_name, best_model_score, model_report[best_model_name]

    
