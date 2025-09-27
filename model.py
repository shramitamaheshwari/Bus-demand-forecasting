import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ==============================
# Load Data
# ==============================
train = pd.read_csv("train.csv", parse_dates=['doj'])
test = pd.read_csv("test.csv", parse_dates=['doj'])
transactions = pd.read_csv("transactions.csv", parse_dates=['doj', 'doi'])

# ==============================
# Feature Engineering
# ==============================

# Filter transactions 15 days before journey
trans_15 = transactions[transactions['dbd'] == 15]

# Aggregate features on day -15
grouped_15 = (
    trans_15.groupby(['doj', 'srcid', 'destid'])
    .agg({
        'cumsum_seatcount': 'sum',
        'cumsum_searchcount': 'sum'
    })
    .reset_index()
)
grouped_15.columns = ['doj', 'srcid', 'destid', 'seatcount_15d', 'searchcount_15d']

# Add calendar features
for df in [train, test]:
    df['dow'] = df['doj'].dt.dayofweek  # Day of week (0=Monday, 6=Sunday)
    df['month'] = df['doj'].dt.month    # Month

# Merge aggregated features
train = train.merge(grouped_15, on=['doj', 'srcid', 'destid'], how='left')
test = test.merge(grouped_15, on=['doj', 'srcid', 'destid'], how='left')

# Fill missing values
for df in [train, test]:
    df['seatcount_15d'] = df['seatcount_15d'].fillna(0)
    df['searchcount_15d'] = df['searchcount_15d'].fillna(0)

# ==============================
# Prepare Data for Model
# ==============================
features = ['srcid', 'destid', 'dow', 'month', 'seatcount_15d', 'searchcount_15d']
target = 'final_seatcount'

X = train[features]
y = train[target]
X_test = test[features]

# ==============================
# Model Training with LightGBM
# ==============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = []
oof_preds = np.zeros(len(train))

for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    models.append(model)

# Cross-validation RMSE
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"CV RMSE: {rmse:.4f}")

# ==============================
# Predict on Test Set
# ==============================
test_preds = np.mean([model.predict(X_test) for model in models], axis=0)

# ==============================
# Submission
# ==============================
submission = pd.read_csv("sample_submission.csv")
submission['final_seatcount'] = test_preds
submission.to_csv("submission.csv", index=False)

print("✅ Submission saved as submission.csv")
