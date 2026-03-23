from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

RAW_PATH = Path("data/raw/AB_NYC_2019.csv")
PROCESSED_DIR = Path("data/processed")
REFERENCE_DIR = Path("data/reference")
REPORTS_DIR = Path("reports")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_stratify_key(df: pd.DataFrame) -> pd.Series:
    price_bins = pd.qcut(
        df["target"],
        q=5,
        duplicates="drop"
    ).astype(str)

    stratify_key = (
        df["neighbourhood_group"].astype(str) + "__" +
        df["room_type"].astype(str) + "__" +
        price_bins
    )

    rare_keys = stratify_key.value_counts()
    rare_keys = rare_keys[rare_keys < 2].index
    stratify_key = stratify_key.where(~stratify_key.isin(rare_keys), "other")

    return stratify_key


def main():
    df = pd.read_csv(RAW_PATH)

    initial_shape = df.shape

    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Удаляем очевидный мусор
    df = df.drop_duplicates()
    df = df[df["price"] > 0].copy()

    # Заполняем почти пустые текстовые пропуски
    df["name"] = df["name"].fillna("unknown")
    df["host_name"] = df["host_name"].fillna("unknown")

    # Обработка пропусков в review-признаках
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    max_review_date = df["last_review"].max()
    df["days_since_last_review"] = (max_review_date - df["last_review"]).dt.days
    df["days_since_last_review"] = df["days_since_last_review"].fillna(-1)

    df["has_last_review"] = df["last_review"].notna().astype(int)

    # Таргет
    df["target"] = np.log1p(df["price"])

    # Стратификация
    stratify_key = build_stratify_key(df)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify_key
    )

    temp_stratify = build_stratify_key(temp_df)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_stratify
    )

    # Сохраняем сплиты
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    # Reference dataset для drift-monitoring
    train_df.to_parquet(REFERENCE_DIR / "reference.parquet", index=False)

    # Краткий отчет
    summary = {
        "initial_shape": initial_shape,
        "final_shape": df.shape,
        "train_shape": train_df.shape,
        "val_shape": val_df.shape,
        "test_shape": test_df.shape,
        "rows_removed_price_le_0": int((pd.read_csv(RAW_PATH)["price"] <= 0).sum()),
        "missing_after_cleaning": df.isna().sum().to_dict(),
        "target_column": "target",
        "business_target_column": "price",
        "max_last_review_date": str(max_review_date.date()) if pd.notna(max_review_date) else None
    }

    with open(REPORTS_DIR / "data_prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Data preparation finished.")
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print("Reference dataset saved to:", REFERENCE_DIR / "reference.parquet")
    print("Summary saved to:", REPORTS_DIR / "data_prep_summary.json")


if __name__ == "__main__":
    main()
