import pandas as pd

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days = 1)
    return (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency =("InvoiceNo",'nunique'),
            Monetary=("TotalPrice", "sum"),

        )
        .reset_index()
    )