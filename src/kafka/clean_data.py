import pandas as pd
from pathlib import Path
import re


def to_snake_case(col):
    col = col.strip()
    col = col.replace("?", "")
    col = re.sub(r"[^\w\s]", "", col)
    col = col.replace(" ", "_")
    col = col.lower()
    return col


def clean_column_names_and_save(df, save_path):
    df.columns = [to_snake_case(col) for col in df.columns]
    df.to_csv(save_path, index=False)


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"

    user_df = pd.read_csv(data_dir / "sd254_users.csv")
    card_df = pd.read_csv(data_dir / "sd254_cards.csv")
    transaction_df = pd.read_csv(data_dir / "credit_card_transactions-ibm_v2.csv")

    clean_column_names_and_save(user_df, data_dir / "user_cleaned.csv")
    clean_column_names_and_save(card_df, data_dir / "card_cleaned.csv")
    clean_column_names_and_save(
        transaction_df, data_dir / "transaction_cleaned.csv"
    )


if __name__ == "__main__":
    main()
