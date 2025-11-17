import matplotlib.pyplot as plt
import pandas as pd

scheduled_2025 = [2161, 2124, 2125, 2123, 2130, 3781, 3865, 3121, 3122, 3123, 3124, 3125, 3126, 3201, 
                  3265, 3282, 3461, 3701, 4222, 4263, 2134, 2145, 2135, 3421, 3381, 2132, 2143, 2131]

receivals = pd.read_csv("data/kernel/receivals.csv")
receivals = receivals.dropna(subset=["rm_id", "purchase_order_id"])
receivals["rm_id"] = receivals["rm_id"].astype(int)
receivals["purchase_order_id"] = receivals["purchase_order_id"].astype(int)
receivals["date_arrival"] = pd.to_datetime(receivals["date_arrival"], utc=True)
receivals = receivals[receivals["date_arrival"] >= pd.to_datetime("2018-01-01", utc=True)]

pred = pd.read_csv("predictions/lgbm_per_rm.csv")
pred["forecast_end_date"] = pd.to_datetime(pred["forecast_end_date"])

pred2 = pd.read_csv("predictions/lgbm_global_quantile02_noLeak.csv")
pred2["forecast_end_date"] = pd.to_datetime(pred2["forecast_end_date"])

pred3 = pd.read_csv("predictions/optuna.csv")
pred3["forecast_end_date"] = pd.to_datetime(pred3["forecast_end_date"])

for rm_id in scheduled_2025:
    receivals_plot = receivals[receivals["rm_id"] == rm_id] 
    receivals_plot["cum_weight"] = receivals_plot["net_weight"].cumsum()
    
    df = pred[pred["rm_id"] == rm_id].copy()
    
    df2 = pred2[pred2["rm_id"] == rm_id].copy()

    df3 = pred3[pred3["rm_id"] == rm_id].copy()


    plt.figure(figsize=(12, 6))

    plt.step(
        receivals_plot["date_arrival"], 
        receivals_plot["cum_weight"], 
        label=f"Historic cum_weight",
        color="tab:green"
    )

    plt.step(
        df["forecast_end_date"],
        df["cum_weight"],
        label=f"Per rm_id RMSE",
        color="tab:blue"
    )

    plt.step(
        df2["forecast_end_date"],
        df2["cum_weight"],
        label=f"Global quantile",
        color="tab:red"
    )

    plt.step(
        df3["forecast_end_date"],
        df3["cum_weight"],
        label=f"Optuna",
        color="tab:orange"
    )

    plt.title(f"RM_ID: {rm_id}")
    plt.legend()
    plt.show()