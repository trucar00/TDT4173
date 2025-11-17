import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved booster
booster = lgb.Booster(model_file="optuna_lgbm.txt")

# Get feature names
feature_names = booster.feature_name()

# Get importance by gain
importance_gain = booster.feature_importance(importance_type="gain")

# Normalize to 0-1
importance_gain_norm = importance_gain / importance_gain.sum()

# Create DataFrame
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance_gain": importance_gain,
})

feat_imp["importance_gain_norm"] = importance_gain_norm

# Sort descending
feat_imp = feat_imp.sort_values(by="importance_gain_norm", ascending=False)

df_feats = feat_imp[["feature", "importance_gain_norm"]]

df_feats = df_feats.sort_values(by="importance_gain_norm", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_feats["feature"], df_feats["importance_gain_norm"], color="blue")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()