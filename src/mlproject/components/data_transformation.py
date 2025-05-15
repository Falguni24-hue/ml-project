import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

@dataclass
class LiquidityModelConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    test_size: float = 0.2
    random_state: int = 1

class LiquidityModelPipeline:
    def __init__(self, data_path: str, config: LiquidityModelConfig = LiquidityModelConfig()):
        self.data_path = data_path
        self.config = config
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

    def run(self):
        """Full pipeline: Load data → preprocess → train → evaluate → save model/scaler"""
        try:
            df = self._load_and_engineer_data()
            X_train, X_test, y_train, y_test, scaler = self._prepare_data(df)
            model = self._train_model(X_train, y_train)
            self._evaluate_model(model, scaler, X_test, y_test)

            # Save model and scaler
            joblib.dump(model, self.config.model_path)
            joblib.dump(scaler, self.config.scaler_path)
            logging.info(f"Model saved to {self.config.model_path}")
            logging.info(f"Scaler saved to {self.config.scaler_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def _load_and_engineer_data(self):
        """Loads data, applies feature engineering and outlier removal."""
        logging.info(f"Reading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df.dropna(inplace=True)

        logging.info("Applying feature engineering")
        df['moving_average'] = df['price'].rolling(window=5).mean()
        df['volatility'] = df['price'].rolling(window=5).std()
        df['liquidity_ratio'] = df['24h_volume'] / df['mkt_cap']
        df.dropna(inplace=True)

        Q1 = df['liquidity_ratio'].quantile(0.25)
        Q3 = df['liquidity_ratio'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['liquidity_ratio'] >= Q1 - 1.5 * IQR) & (df['liquidity_ratio'] <= Q3 + 1.5 * IQR)]
        logging.info(f"After outlier removal: {df.shape}")
        return df

    def _prepare_data(self, df):
        """Splits data and applies MinMax scaling."""
        logging.info("Splitting and scaling data")
        df.to_csv("artifacts/processed_full.csv", index=False)

        X = df[['24h', '7d', '24h_volume', 'mkt_cap']]
        y = df['liquidity_ratio']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
 
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def _train_model(self, X_train, y_train):
        """Trains RandomForest model with GridSearchCV."""
        logging.info("Training RandomForestRegressor with GridSearchCV")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100],
            'max_depth': [10, None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt']
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        logging.info(f"Best model parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _evaluate_model(self, model, scaler, X_test, y_test):
        """Evaluates and visualizes model performance."""
        logging.info("Evaluating model performance")
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"R² Score: {r2}")
        print("R² Score:", r2)

        self._visualize_feature_importance(model)
        self._visualize_predictions(y_test, y_pred)

    def _visualize_feature_importance(self, model):
        """Plots feature importance for trained model."""
        features = ['24h', '7d', '24h_volume', 'mkt_cap']
        importances = model.feature_importances_
        pd.Series(importances, index=features).sort_values(ascending=False).plot(kind='bar', figsize=(10, 5), title='Feature Importance')
        plt.tight_layout()
        plt.show()

    def _visualize_predictions(self, y_test, y_pred):
        """Shows scatter and residual plots."""
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.show()

        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True)
        plt.title("Distribution of Residuals")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        data_path = os.path.join("artifacts", "raw.csv")
        pipeline = LiquidityModelPipeline(data_path=data_path)
        pipeline.run()
    except Exception as e:
        raise CustomException(e, sys)
