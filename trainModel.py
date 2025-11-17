import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, BaseCrossValidator
from sklearn.metrics import root_mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
from functools import partial
import optuna

LAST_RECEIVAL_DATE = pd.Timestamp("2024-12-19", tz="UTC")

def quantile_loss_scorer(y_true, y_pred, alpha):
    """
    Pinball loss for quantile regression
    """
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))

def randomSearchCV_lgbm(df_train):
    X = df_train.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y = df_train["cum_weight"]

    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],  
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.4, 0.6, 0.8, 1.0],         
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],  
        "min_child_samples": [10, 20, 30, 50],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        "min_split_gain": [0.0, 0.1, 0.2, 0.5]
    }

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    tscv = TimeSeriesSplit(n_splits=5)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_iter=5,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X, y)

    print("Best parameters:", random_search.best_params_)
    print("Best RMSE:", -random_search.best_score_)

    return random_search.best_estimator_, random_search.cv_results_


def train_all_rm_ids(df_train, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    rm_ids = df_train["rm_id"].unique()
    results = []

    for rid in tqdm(rm_ids):
        print(f"\nTraining rm_id {rid}")
        df_rm = df_train[df_train["rm_id"] == rid].copy()

        df_rm = df_rm.reset_index(drop=True)

        model, cv_results = randomSearchCV_lgbm(df_rm)

        model_path = os.path.join(model_dir, f"lgbm_rm_{rid}.txt")
        model.booster_.save_model(model_path)

        results.append({"rm_id": rid, "model_path": model_path})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(model_dir, "training_results.csv"), index=False)
    print(f"\nTraining completed. Results saved to {model_dir}")
    
    return results_df

def quantile_loss(y_true, y_pred, alpha=0.2):
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))

def objective(trial, X_train, y_train, X_valid, y_valid):
    params = dict(
        objective="quantile",
        alpha=0.2,
        learning_rate=trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
        random_state=42,
        n_jobs=-1
    )

    model = lgb.LGBMRegressor(**params, n_estimators=5000)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="quantile",
        categorical_feature=["rm_id"],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    preds = model.predict(X_valid)
    return quantile_loss(y_valid, preds, alpha=0.2)


def train_global_model_lgbm_optuna(df_train, model_dir):

    # Define training/validation split by time
    cutoff = LAST_RECEIVAL_DATE - pd.Timedelta(days=150)
    df_train["forecast_start"] = pd.to_datetime(df_train["forecast_start"], utc=True)
    df_train["forecast_end"] = pd.to_datetime(df_train["forecast_end"], utc=True)

    train = df_train[df_train["forecast_end"] < cutoff]
    valid = df_train[df_train["forecast_start"] >= cutoff]

    X_train = train.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y_train = train["cum_weight"]
    X_valid = valid.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y_valid = valid["cum_weight"]

    # === Optuna tuning ===
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=10)
    best_params = study.best_params
    print("Best params:", best_params)

    # Add fixed parameters
    best_params.update({
        "objective": "quantile",
        "alpha": 0.2,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    })

    # === Retrain model with best params ===
    best_model = lgb.LGBMRegressor(**best_params, n_estimators=1000)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="quantile",
        categorical_feature=["rm_id"],
        callbacks=[lgb.early_stopping(200)]
    )

    best_iteration = best_model.best_iteration_
    print(f"Best iteration found: {best_iteration}")

    # === Train final model on all data ===
    X_full = df_train.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y_full = df_train["cum_weight"]

    final_model = lgb.LGBMRegressor(**best_params, n_estimators=best_iteration)
    final_model.fit(X_full, y_full, categorical_feature=["rm_id"])

    final_model.booster_.save_model(model_dir)
    return final_model


def train_global_model_lgbm(df_train, model_dir):

    # Define training and validation split by time
    cutoff = LAST_RECEIVAL_DATE - pd.Timedelta(days=150)
    df_train["forecast_start"] = pd.to_datetime(df_train["forecast_start"], utc=True)
    df_train["forecast_end"] = pd.to_datetime(df_train["forecast_end"], utc=True)
    
    train = df_train[df_train["forecast_end"] < cutoff] 
    valid = df_train[df_train["forecast_start"] >= cutoff]

    # Define features and target
    X_train = train.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y_train = train["cum_weight"]

    X_valid = valid.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y_valid = valid["cum_weight"]
    
    # Define model

    params = dict(
        objective="quantile",
        alpha = 0.2,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model = lgb.LGBMRegressor(**params, n_estimators=1000)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="quantile",
        categorical_feature=["rm_id"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )

    best_iteration = model.best_iteration_
    print(f"\nBest iteration found: {best_iteration}")

    X_full = df_train.drop(columns=["cum_weight", "forecast_start", "forecast_end"])
    y_full = df_train["cum_weight"]

    final_model = lgb.LGBMRegressor(**params, n_estimators=best_iteration)
    final_model.fit(
        X_full, y_full,
        categorical_feature=["rm_id"],
    )

    final_model.booster_.save_model(model_dir)
    
    return final_model

def main():
    df_train = pd.read_csv("trainingData/newTrainingSet.csv")

    #df_model, df_test = split_train_test(df_train)
    #train_all_rm_ids(df_model, model_dir="models_holdOut")
    #train_global_model_lgbm(df_train, model_dir="models/lgbm_global_quantile02_noLeak.txt")
    train_global_model_lgbm_optuna(df_train, "optuna_lgbm.txt")

    return


if __name__ == "__main__":
    main()