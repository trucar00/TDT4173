import pandas as pd

def createSubmissionFile(pred_path, sub_path, pred_map_path="data/prediction_mapping.csv"):
    forecast = pd.read_csv(pred_path)
    mapping = pd.read_csv(pred_map_path)

    merged = mapping.merge(
        forecast,
        on=["rm_id", "forecast_start_date", "forecast_end_date"],
        how="left"
    )

    # Replace missing cum_weight values with 0
    merged["cum_weight"] = merged["cum_weight"].fillna(0)

    # Prepare submission format
    submission = merged[["ID", "cum_weight"]].rename(columns={"cum_weight": "predicted_weight"})

    # Save to CSV
    submission.to_csv(sub_path, index=False)
    
    return submission