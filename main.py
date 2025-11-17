import buildTrainFeats
import trainModel
import forecasting
import createSubmission
import lightgbm as lgb
import pandas as pd

FORECAST_START = pd.to_datetime("2025-01-01", utc=True)
FORECAST_END = pd.to_datetime("2024-05-31", utc=True)


def main():
    # == Prepare and build features and training set ==
    receivals, purchase_orders, daily_receivals, qty_2025_rm_id, scheduled_rm_ids, delays, delay_stats, active_rm_ids = buildTrainFeats.prepareData()
    df_train = buildTrainFeats.buildTrainSet(receivals, daily_receivals, purchase_orders, delays, scheduled_rm_ids, qty_2025_rm_id, "fullTrainingSet.csv")
    df_train = pd.read_csv("trainingData/newTrainingSet.csv")

    # == Train LGBM for each rm_id and global LGBM ==
    model_dir_per_rm = "modelsPerRm"
    trainModel.train_all_rm_ids(df_train, model_dir=model_dir_per_rm) # Per RM_ID model

    model_dir_global = "lgbm_global.txt"
    trainModel.train_global_model_lgbm(df_train, model_dir=model_dir_global)

    model_dir_optuna = "lgbm_global_optuna.txt"
    trainModel.train_global_model_lgbm_optuna(df_train, model_dir=model_dir_global)

    # == Forecast for 2025 ==
    pred_path_per = "lgbm_per_rm.csv"
    forecasting.forecast2025_per_rm(FORECAST_START, FORECAST_END, scheduled_rm_ids, 
                                    daily_receivals, purchase_orders, delays, qty_2025_rm_id, receivals,
                                    models_dir=model_dir_per_rm, pred_path=pred_path_per)
    
    booster = lgb.Booster(model_file=model_dir_global)
    pred_path_global = "lgbm_global.csv"
    forecasting.forecast2025(booster, FORECAST_START, FORECAST_END, scheduled_rm_ids, 
                                 daily_receivals, purchase_orders, delays, qty_2025_rm_id, receivals, 
                                 pred_path=pred_path_global)
    
    booster_optuna = lgb.Booster(model_file=model_dir_optuna)
    pred_path_optuna = "lgbm_optuna.csv"
    forecasting.forecast2025(booster_optuna, FORECAST_START, FORECAST_END, scheduled_rm_ids, 
                                 daily_receivals, purchase_orders, delays, qty_2025_rm_id, receivals, 
                                 pred_path=pred_path_optuna)

    # == Create kaggle submission == 
    print("Creating submission files.")
    createSubmission.createSubmissionFile(pred_path_per, "lgbm_per_rm_sub.csv")
    createSubmission.createSubmissionFile(pred_path_global, "lgbm_global_sub.csv")
    createSubmission.createSubmissionFile(pred_path_optuna, "lgbm_optuna_sub.csv")

    print("Done!")
    return

if __name__ == "__main__":
    main()