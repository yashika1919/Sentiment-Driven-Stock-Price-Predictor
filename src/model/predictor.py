# predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

class StockPricePredictor:
    def __init__(self):
        self.model = XGBRegressor()

    def train(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Model trained with MSE: {mse}')
        return mse

    def predict(self, new_data):
        return self.model.predict(new_data)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)