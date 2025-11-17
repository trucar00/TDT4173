import pandas as pd
from tqdm import tqdm

def getActiveRmIds(rec_path, po_path, mat_path):
    po = pd.read_csv(po_path)

    po["delivery_date"] = pd.to_datetime(po["delivery_date"], utc=True)

    cutoff = pd.to_datetime("2025-01-01", utc=True)

    po_2025 = po[po["delivery_date"] >= cutoff].copy()

    po_2025 = po_2025.sort_values(by="delivery_date")

    #print(po_2025[["purchase_order_id", "product_id", "delivery_date", "quantity"]].head())

    materials = pd.read_csv(mat_path)
    materials = materials.dropna(subset=["product_id", "rm_id"])

    materials["product_id"] = materials["product_id"].astype(int)
    materials["rm_id"] = materials["rm_id"].astype(int)
    materials_unique = materials.drop_duplicates(subset=["rm_id", "product_id"])

    merged = po_2025.merge(materials_unique, on=["product_id"], how="left")

    receivals = pd.read_csv(rec_path)
    receivals = receivals.dropna(subset=["rm_id", "purchase_order_id"])
    receivals["rm_id"] = receivals["rm_id"].astype(int)
    receivals["date_arrival"] = pd.to_datetime(receivals["date_arrival"], utc=True)

    cutoff_recent = pd.to_datetime("2024-01-01", utc=True) # Only rm_ids that had a delivery in 2024, if not they are likely out of production
    rm_ids_recent = receivals.loc[receivals["date_arrival"] >= cutoff_recent, "rm_id"].unique()

    merged_active = merged[merged["rm_id"].isin(rm_ids_recent)].copy()

    # This are the scheduled quantity for all active rm_ids/product_ids in 2025.
    qty_rm_id_2025 = merged_active[["rm_id", "product_id", "delivery_date", "quantity"]]
    qty_rm_id_2025["delivery_date"] = qty_rm_id_2025["delivery_date"].dt.date

    return merged_active["rm_id"].unique(), qty_rm_id_2025

def getProductIdRmIDMap(df_active_rm_ids):
    product_id_to_rm = (
        df_active_rm_ids.groupby("product_id")["rm_id"]
                    .apply(lambda x: list(set(x)))  # only unique rm_ids
                    .reset_index()
    )

    return product_id_to_rm

def getProdId(rm_id, mapping):
    row = mapping.loc[mapping["rm_id"] == rm_id]
    if row.empty:
        return None
    return row["product_id"].iloc[0]


def getExpectedQty(rm_id, purchase_orders, mapping, forecast_start, forecast_end):
    prod_id = getProdId(rm_id, mapping)
    if prod_id is None:
        return 0.0
    expected_qty_product_id = purchase_orders.loc[
        (purchase_orders["product_id"] == prod_id) &
        (purchase_orders["delivery_date"] >= forecast_start) &
        (purchase_orders["delivery_date"] <= forecast_end),
        "quantity"
    ].sum()

    return expected_qty_product_id

def prob_rm_id(rm_id, mapping, receivals, forecast_start, days):
    # Find prob of
    prod_id = getProdId(rm_id, mapping)
    check_from = (forecast_start - pd.Timedelta(days=days))
    recent_prodId = receivals.loc[
        (receivals["product_id"] == prod_id) & # can be a problem with float vs int here. Maybe drop NaNs in receivals and initfy
        (receivals["date_arrival"] >= check_from) &
        (receivals["date_arrival"] < forecast_start)
    ]
    
    if recent_prodId.empty:
        return 1.0
     
    total_count = len(recent_prodId)
    rm_count = (recent_prodId["rm_id"] == rm_id).sum()
    return rm_count/total_count

def get_exp_qty_rm_id(rm_id, mapping, receivals, purchase_orders, forecast_start, forecast_end, histDays):
    totExpectedQty = getExpectedQty(rm_id, purchase_orders, mapping, forecast_start, forecast_end)
    probRmID = prob_rm_id(rm_id, mapping, receivals, forecast_start, histDays)
    expectQty = totExpectedQty*probRmID
    bufferQty = totExpectedQty-expectQty
    return expectQty, bufferQty


def buildExpQty(forecast_start, scheduled_rm_ids, last_receival, mapping, receivals, purchase_orders):
    train_rows = []
    window_lengths = range(1, 151) 
    start_dates = pd.date_range(pd.to_datetime("2018-01-01", utc=True), forecast_start, freq="14D")  # slide every 14 days

    for rm_id in tqdm(scheduled_rm_ids):
        print("Processing rm_id: ", rm_id)

        for wl in tqdm(window_lengths):
            for start in start_dates:
                end = start + pd.Timedelta(days=wl - 1)
                if end >= last_receival: # last date in receivals
                    break
                expectQty, bufferQty = get_exp_qty_rm_id(rm_id, mapping, receivals, purchase_orders, start.date(), end.date(), 150)
                feats = {
                    "rm_id": rm_id,
                    "f_start": start.date(),
                    "f_end": end.date(),
                    "expected_qty_rm_id": expectQty,
                    "buffer_qty_rm_id": bufferQty
                         }

                train_rows.append(feats)

    df_add_exp_qty = pd.DataFrame(train_rows)
    return df_add_exp_qty
def main():
    active_rm_ids, qty_2025_rm_id = getActiveRmIds("data/kernel/receivals.csv","data/kernel/purchase_orders.csv", "data/extended/materials.csv")
    #mapping = getProductIdRmIDMap(qty_2025_rm_id)
    #getExpectedQty(4481, mapping=mapping)
    
    print(getProdId(3781, qty_2025_rm_id))
    return

if __name__ == "__main__":
    main()

# PLAN:
# Find active rm_ids based on some cutoffdate
# Find scheduled rm_ids for 2025 from purchase_orders merged with materials
# Merge these two to find expected rm_ids in 2025
# For these rm_ids find expected qty based on the scheduled corresponding product_id within the time window.
# These product_ids will include x nr of rm_ids, find the prob of 
# expected_qty_rm_id = prob(rm_id) * expected_qty_product_id