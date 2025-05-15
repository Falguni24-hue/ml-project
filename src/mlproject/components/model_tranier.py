import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
from dataclasses import dataclass

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def initiate_model_training(self, data_path: str):
        try:
            # Step 1: Load the processed, feature-engineered data
            df = pd.read_csv(data_path)
            logging.info(f"Data loaded from {data_path}")

            # Step 2: Split into features and targets
            X = df[['24h','7d','24h_volume','mkt_cap']]
            y = df["liquidity_ratio"]

            # Step 3: Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info(f"Training on features: {X.columns.tolist()}")

            # Step 4: Train a model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Step 5: Evaluate the model
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            logging.info(f"Model R² Score: {score}")
            print("R² Score:", score)

            # Step 6: Save the trained model
            joblib.dump(model, self.config.model_path)
            logging.info(f"Model saved at {self.config.model_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training(data_path="artifacts/processed_full.csv")
