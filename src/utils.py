import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from math import sqrt

def save_object(file_path, obj):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)  # Calculate RMSE manually
        report[model_name] = {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    return report