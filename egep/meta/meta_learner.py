import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class WeeklyMetaLearner:
    def __init__(self, lag_window=3, n_trials=20):
        self.lag_window = lag_window
        self.n_trials = n_trials
        self.model = None

    def _build_features(self, preds: pd.DataFrame, actuals: pd.Series):
        # Clip negative predictions to 0
        df = preds.clip(lower=0).copy()
        df["actual"] = actuals.values

        features = []
        # Lag features for model predictions
        for col in preds.columns:
            for i in range(1, self.lag_window + 1):
                name = f"{col}_lag{i}"
                df[name] = df[col].shift(i)
                features.append(name)

        # Lag features for actuals
        for i in range(1, self.lag_window + 1):
            name = f"actual_lag{i}"
            df[name] = df["actual"].shift(i)
            features.append(name)

        # Drop NaN rows created by shifting
        df = df.dropna()
        return df[features], df["actual"]

    def tune_and_train(self, preds, actuals):
        X, y = self._build_features(preds, actuals)

        def objective(trial):
            # Split for internal validation during tuning
            train_x, val_x, train_y, val_y = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "verbosity": 0
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(train_x, train_y)
            pred = model.predict(val_x)
            return mean_squared_error(val_y, pred, squared=False)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        # Retrain on full Period B data
        self.model = xgb.XGBRegressor(
            **study.best_params,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def predict(self, preds, actuals):
        if self.model is None:
            raise RuntimeError("Meta learner not trained.")
        
        # Note: dropna() in _build_features will remove the first 'lag_window' rows
        # Make sure input data includes context if you need full predictions
        X, y = self._build_features(preds, actuals)
        return self.model.predict(X), y