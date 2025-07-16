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


def clean_column_names(df):
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def add_idx(df):
    df.insert(0, "idx", range(len(df)))


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"

    user_df = pd.read_csv(data_dir / "sd254_users.csv")
    card_df = pd.read_csv(data_dir / "sd254_cards.csv")
    transaction_df = pd.read_csv(data_dir / "credit_card_transactions-ibm_v2.csv")

    add_idx(user_df)
    add_idx(card_df)
    add_idx(transaction_df)

    user_df = clean_column_names(user_df)
    card_df = clean_column_names(card_df)
    transaction_df = clean_column_names(transaction_df)

    user_df["user"] = user_df["idx"].apply(lambda x: f"user{int(x)}")
    
    card_df["user"] = card_df["user"].apply(lambda x: f"user{int(x)}")
    
    transaction_df["card"] = transaction_df["card"].apply(lambda x: f"card{int(x)}")
    transaction_df["user"] = transaction_df["user"].apply(lambda x: f"user{int(x)}")

    user_df.to_csv(data_dir / "user_cleaned.csv", index=False)
    card_df.to_csv(data_dir / "card_cleaned.csv", index=False)
    transaction_df.to_csv(data_dir / "transaction_cleaned.csv", index=False)


if __name__ == "__main__":
    main()
