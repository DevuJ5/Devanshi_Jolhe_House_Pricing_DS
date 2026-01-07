import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("data/raw")
TRAIN_PATH = DATA_DIR / "train(1).xlsx"
TEST_PATH  = DATA_DIR / "test2.xlsx"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    train_df = pd.read_excel(TRAIN_PATH)
    test_df  = pd.read_excel(TEST_PATH)

    assert "price" in train_df.columns, "Train must contain 'price'"
    assert "id" in test_df.columns or "id" in train_df.columns, "Need an id column"

    # Separate features/target
    y = np.log1p(train_df["price"].values)  # log target
    X = train_df.drop(columns=["price"])

    # Train/valid split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Detect numeric/categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocess
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)

    print("TABULAR baseline (log target):")
    print("RMSE(log):", rmse(y_val, val_pred))
    print("R2(log):", r2_score(y_val, val_pred))

    # Train on full train, predict test
    pipe.fit(X, y)
    test_pred_log = pipe.predict(test_df)
    test_pred = np.expm1(test_pred_log)

    id_col = "id" if "id" in test_df.columns else test_df.columns[0]
    sub = pd.DataFrame({"id": test_df[id_col].values, "predicted_price": test_pred})
    out_path = Path("outputs/preds/tabular_baseline.csv")
    sub.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
