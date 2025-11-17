from tqdm import tqdm
import buildTrainFeats
import pandas as pd
import os
import lightgbm as lgb
import numpy as np

def forecast2025(model, forecast_start, forecast_end, active_rm_ids, daily_receivals, purchase_orders, delays,
             mapping, receivals, pred_path):

    print("Forecasting for 2025!")
    pred_rows = []

    forecast_range = pd.date_range(forecast_start, forecast_end - pd.Timedelta(days=1))

    receivals_by_rm = {rid: df for rid, df in receivals.groupby("rm_id")}
    daily_by_rm = {rid: df for rid, df in daily_receivals.groupby("rm_id")}
    delays_by_rm = {rid: df for rid, df in delays.groupby("rm_id")}
    
    for rm_id in tqdm(active_rm_ids):
        print("Building features for rm_id ", rm_id)
        # Build features for the given rm_id and forecast window
        for date in forecast_range:
            end_date = date.date() + pd.Timedelta(days=1)
            feats = buildTrainFeats.build_features(rm_id, forecast_start.date(), end_date,
                                                    daily_by_rm.get(rm_id, pd.DataFrame()), 
                                                    purchase_orders, 
                                                    delays_by_rm.get(rm_id, pd.DataFrame()), 
                                                    mapping, 
                                                    receivals_by_rm.get(rm_id, pd.DataFrame()))
            X_test = pd.DataFrame([feats]).drop(columns=["forecast_start", "forecast_end"])

            # Predict using your trained model
            pred_cum_weight = model.predict(X_test)[0]
            pred_cum_weight = max(pred_cum_weight, 0)

            pred_rows.append({
                "rm_id": rm_id,
                "forecast_start_date": forecast_start.date(),
                "forecast_end_date": end_date,
                "cum_weight": pred_cum_weight
            })


    pred_df = pd.DataFrame(pred_rows)

    pred_df.to_csv(pred_path)
        
    return pred_df


def forecast2025_per_rm(
    forecast_start,
    forecast_end,
    active_rm_ids,
    daily_receivals,
    purchase_orders,
    delays,
    mapping,
    receivals,
    models_dir,
    pred_path
):
    print(f"Forecasting from {forecast_start.date()} to {forecast_end.date()} for {len(active_rm_ids)} rm_ids...")

    pred_rows = []
    forecast_range = pd.date_range(forecast_start, forecast_end - pd.Timedelta(days=1))

    # Group data for fast lookup
    receivals_by_rm = {rid: df for rid, df in receivals.groupby("rm_id")}
    daily_by_rm = {rid: df for rid, df in daily_receivals.groupby("rm_id")}
    delays_by_rm = {rid: df for rid, df in delays.groupby("rm_id")}

    for rm_id in tqdm(active_rm_ids, desc="Forecasting per rm_id"):
        model_path = os.path.join(models_dir, f"lgbm_rm_{rm_id}.txt")

        # Skip if model doesn’t exist
        if not os.path.exists(model_path):
            print(f" No model found for rm_id {rm_id}, skipping.")
            continue

        # Load LightGBM model
        model = lgb.Booster(model_file=model_path)

        print(f"Building features and forecasting for rm_id {rm_id}...")

        for date in forecast_range:
            end_date = date.date() + pd.Timedelta(days=1)

            feats = buildTrainFeats.build_features(
                rm_id,
                forecast_start.date(),
                end_date,
                daily_by_rm.get(rm_id, pd.DataFrame()),
                purchase_orders,
                delays_by_rm.get(rm_id, pd.DataFrame()),
                mapping,
                receivals_by_rm.get(rm_id, pd.DataFrame())
            )

            # Prepare feature vector
            X_test = pd.DataFrame([feats]).drop(columns=["forecast_start", "forecast_end"], errors="ignore")

            # Predict
            pred_cum_weight = model.predict(X_test)[0]
            pred_cum_weight = max(pred_cum_weight, 0)  # no negative forecasts

            pred_rows.append({
                "rm_id": rm_id,
                "forecast_start_date": forecast_start.date(),
                "forecast_end_date": end_date,
                "cum_weight": pred_cum_weight
            })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nForecasting complete. Saved predictions to {pred_path}")


def main():
    
    return

if __name__ == "__main__":
    main()