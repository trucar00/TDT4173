import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

import forecasting
import exploreData

RECEIVALS_PATH = "data/kernel/receivals.csv"
PO_PATH = "data/kernel/purchase_orders.csv"
MAT_PATH = "data/extended/materials.csv"

FORECAST_START = pd.to_datetime("2025-01-01", utc=True)
FORECAST_END = pd.to_datetime("2025-05-31", utc=True)

LAST_RECEIVAL_DATE = pd.Timestamp("2024-12-19", tz="UTC") # receivals["date_arrival"].max() last receival date

def prepareData(receivals_path=RECEIVALS_PATH, po_path=PO_PATH, mat_path=MAT_PATH):
    
    # === Clean receivals ===
    receivals = pd.read_csv(receivals_path)
    receivals = receivals.dropna(subset=["rm_id", "purchase_order_id", "product_id"])
    receivals["rm_id"] = receivals["rm_id"].astype(int)
    receivals["purchase_order_id"] = receivals["purchase_order_id"].astype(int)
    receivals["product_id"] = receivals["product_id"].astype(int)
    receivals["date_arrival"] = pd.to_datetime(receivals["date_arrival"], utc=True)

    # == Clean purchase orders ==
    purchase_orders = pd.read_csv(po_path)
    purchase_orders = purchase_orders[purchase_orders["quantity"] > 0] # Remove negative and 0 quantity orders
    purchase_orders["delivery_date"] = pd.to_datetime(purchase_orders["delivery_date"], utc=True)

    # == Clean materials ==
    materials = pd.read_csv(mat_path)
    materials = materials.dropna(subset=["product_id", "rm_id"])
    materials["product_id"] = materials["product_id"].astype(int)
    materials["rm_id"] = materials["rm_id"].astype(int)

    # Find rm_ids scheduled for 2025
    po_2025 = purchase_orders[purchase_orders["delivery_date"] >= FORECAST_START].copy()
    po_2025 = po_2025.sort_values(by="delivery_date")
    materials_unique = materials.drop_duplicates(subset=["rm_id", "product_id"])

    merged_po2025_mat = po_2025.merge(materials_unique, on=["product_id"], how="left")

    one_year_before = pd.to_datetime("2024-01-01", utc=True) # Only rm_ids that had a delivery in 2024, if not they are likely out of production
    active_rm_ids = receivals.loc[receivals["date_arrival"] >= one_year_before, "rm_id"].unique()

    merged_active = merged_po2025_mat[merged_po2025_mat["rm_id"].isin(active_rm_ids)].copy()
    scheduled_active_rm_ids = merged_active["rm_id"].unique() # 38 found to be active with this method

    # Find scheduled quantity for all active rm_ids/product_ids in 2025.
    qty_rm_id_2025 = merged_active[["rm_id", "product_id", "delivery_date", "quantity"]]
    qty_rm_id_2025["delivery_date"] = qty_rm_id_2025["delivery_date"].dt.date

    # Get daily total receivals per rm_id and mean/std(delay) for each rm_id
    merged = pd.merge(purchase_orders, receivals, on=["purchase_order_id", "purchase_order_item_no"], suffixes=("_receival", "_order"))
    merged = merged.sort_values(by="date_arrival")

    merged["delay"] = (merged["date_arrival"] - merged["delivery_date"]).dt.days
    merged["date_arrival"] = merged["date_arrival"].dt.date
    merged["delivery_date"] = merged["delivery_date"].dt.date

    daily_receivals = merged.groupby(["rm_id", "date_arrival"], as_index=False)["net_weight"].sum()
    daily_receivals["date_arrival"] = pd.to_datetime(daily_receivals["date_arrival"], utc=True)

    delays = merged[["rm_id", "date_arrival", "delay"]]

    delay_stats = merged.groupby("rm_id")["delay"].agg(["mean", "std"]).reset_index()

    purchase_orders["delivery_date"] = purchase_orders["delivery_date"].dt.date
    receivals["date_arrival"] = receivals["date_arrival"].dt.date
    daily_receivals["date_arrival"] = daily_receivals["date_arrival"].dt.date 

    return receivals, purchase_orders, daily_receivals, qty_rm_id_2025, scheduled_active_rm_ids, delays, delay_stats, active_rm_ids


def build_features(rm_id, forecast_start, forecast_end, daily_receivals, purchase_orders, delays, mapping, receivals):
    hist_delays = delays[delays["date_arrival"] < forecast_start]
    hist_daily_receivals = daily_receivals[daily_receivals["date_arrival"] < forecast_start]
    
    recent_from_150 = (forecast_start - pd.Timedelta(days=150))
    recent_from_60 = (forecast_start - pd.Timedelta(days=60))
    recent_from_30 = (forecast_start - pd.Timedelta(days=30))
    
    recent_150 = daily_receivals[(daily_receivals["date_arrival"] >= recent_from_150) 
                           & (daily_receivals["date_arrival"] < forecast_start)].copy()
    
    recent_60 = daily_receivals[(daily_receivals["date_arrival"] >= recent_from_60) 
                           & (daily_receivals["date_arrival"] < forecast_start)].copy()
    
    recent_30 = daily_receivals[(daily_receivals["date_arrival"] >= recent_from_30) 
                           & (daily_receivals["date_arrival"] < forecast_start)].copy()
    

    full_date_range_150 = pd.date_range(
        start=recent_from_150, 
        end=(forecast_start - pd.Timedelta(days=1)), 
        freq="D"
    )

    full_date_range_60 = pd.date_range(
        start=recent_from_60, 
        end=(forecast_start - pd.Timedelta(days=1)), 
        freq="D"
    )

    full_date_range_30 = pd.date_range(
        start=recent_from_30, 
        end=(forecast_start - pd.Timedelta(days=1)), 
        freq="D"
    )

    recent_150_full = (
        recent_150
        .set_index("date_arrival")
        .reindex(full_date_range_150, fill_value=0)
        .rename_axis("date_arrival")
        .reset_index()
    )

    recent_60_full = (
        recent_60
        .set_index("date_arrival")
        .reindex(full_date_range_60, fill_value=0)
        .rename_axis("date_arrival")
        .reset_index()
    )

    recent_30_full = (
        recent_30
        .set_index("date_arrival")
        .reindex(full_date_range_30, fill_value=0)
        .rename_axis("date_arrival")
        .reset_index()
    )

    recent_150_full["rm_id"] = rm_id
    recent_60_full["rm_id"] = rm_id
    recent_30_full["rm_id"] = rm_id

    total_expected_quantity = purchase_orders.loc[
        (purchase_orders["delivery_date"] >= forecast_start) &
        (purchase_orders["delivery_date"] <= forecast_end),
        "quantity"
    ].sum()

    mean_daily_weight_150 = recent_150_full["net_weight"].mean()
    mean_daily_weight_60 = recent_60_full["net_weight"].mean()
    mean_daily_weight_30 = recent_30_full["net_weight"].mean()
    window_length = (forecast_end-forecast_start).days + 1

    expectQty_rm, bufferQty_rm = exploreData.get_exp_qty_rm_id(rm_id, mapping, receivals, purchase_orders, forecast_start, forecast_end, 150)

    last_delivery = hist_daily_receivals["date_arrival"].max()
    if pd.isna(last_delivery):
        days_since_last_delivery = np.nan
    else:
        days_since_last_delivery = (forecast_start - last_delivery).days

    features = {
        "rm_id": rm_id,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "window_length": window_length,
        "mean_daily_weight_150": mean_daily_weight_150,
        "mean_daily_weight_60": mean_daily_weight_60,
        "mean_daily_weight_30": mean_daily_weight_30,
        "days_since_last_delivery": days_since_last_delivery,
        "num_deliveries_last_150": len(recent_150),
        "avg_delivery_time": hist_delays["delay"].mean(),
        "std_delivery_time": hist_delays["delay"].std(),
        "total_expected_qty": total_expected_quantity,
        "expected_qty_rm_id": expectQty_rm,
        "buffer_qty": bufferQty_rm

    }

    # Can make month and day etc on forecast_end

    return features

def buildTrainSet(receivals, daily_receivals, purchase_orders, delays, active_rm_ids, mapping, output_path):
    train_rows = []
    window_lengths = range(1, 151) 
    start_dates = pd.date_range(pd.to_datetime("2018-01-01", utc=True), FORECAST_START, freq="14D")  # slide every 14 days
    
    receivals_by_rm = {rid: df for rid, df in receivals.groupby("rm_id")}
    daily_by_rm = {rid: df for rid, df in daily_receivals.groupby("rm_id")}
    delays_by_rm = {rid: df for rid, df in delays.groupby("rm_id")}
    
    for rm_id in tqdm(active_rm_ids):
        print("Processing rm_id: ", rm_id)

        for wl in tqdm(window_lengths):
            for start in start_dates:
                end = start + pd.Timedelta(days=wl - 1)
                if end >= LAST_RECEIVAL_DATE: # last date in receivals
                    break
                feats = build_features(rm_id, start.date(), end.date(), 
                                       daily_by_rm.get(rm_id, pd.DataFrame()), 
                                       purchase_orders, 
                                       delays_by_rm.get(rm_id, pd.DataFrame()), 
                                       mapping, receivals_by_rm.get(rm_id, pd.DataFrame()))

                cum_weight = daily_receivals.loc[
                    (daily_receivals["rm_id"] == rm_id) &
                    (daily_receivals["date_arrival"] >= start.date()) &
                    (daily_receivals["date_arrival"] <= end.date()),
                    "net_weight"
                ].sum()

                feats["cum_weight"] = cum_weight
                train_rows.append(feats)

    df_train = pd.DataFrame(train_rows)

    df_train.to_csv(output_path, index=False)
    df_train.to_csv("trainingData/NewTrainingSetWithIndex.csv")
    #df_train.to_parquet("fullTraining.parquet", engine="pyarrow")

    return df_train

def main():
    receivals, purchase_orders, daily_receivals, qty_2025_rm_id, scheduled_rm_ids, delays, delay_stats, active_rm_ids = prepareData()
    
    #df_train = buildTrainSet(receivals, daily_receivals, purchase_orders, delays, scheduled_rm_ids, qty_2025_rm_id, "fullTrainingSet.csv")
    df_train = pd.read_csv("trainingData/newTrainingSet.csv")

    booster = lgb.Booster(model_file="models/lgbm_global_quantile.txt")

    forecasting.forecast2025_new(booster, FORECAST_START, FORECAST_END, scheduled_rm_ids, daily_receivals,
                             purchase_orders, delays, qty_2025_rm_id, receivals, "predictions/lgbm_global_quantile.csv")
    
    forecasting.forecast2025_per_rm(FORECAST_START, FORECAST_END, scheduled_rm_ids, daily_receivals, purchase_orders, 
                                    delays, "predictions/lgbm_per_rm_sorted_10split.csv", qty_2025_rm_id, receivals)

    
    return

if __name__ == "__main__":
    main()



