# train_save.py
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def tune_and_train(model, X_train, y_train, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def save_model(model, file_path='model.joblib'):
    joblib.dump(model, file_path)

def load_model(file_path='model.joblib'):
    return joblib.load(file_path)
