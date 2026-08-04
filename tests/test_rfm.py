import pandas as pd
from src.clustering_library.clustering_with_rfm_features import compute_rfm


def test_compute_rfm_values():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 1, 2],
            "InvoiceNo": ["A1", "A1", "A2", "B1"],  # customer 1 has 2 invoices
            "InvoiceDate": pd.to_datetime(
                ["2021-01-01", "2021-01-01", "2021-01-10", "2021-01-05"]
            ),
            "TotalPrice": [10.0, 5.0, 20.0, 7.0],
        }
    )
    rfm = compute_rfm(df).set_index("CustomerID")

    assert list(rfm.columns) == ["Recency","Frequency","Monetary"]
    assert rfm.loc[1, "Recency"] == 1     # last purchase 2021-01-10
    assert rfm.loc[2, "Recency"] == 6     # last purchase 2021-01-05
    assert rfm.loc[1, "Frequency"] == 2   # distinct invoices A1, A2
    assert rfm.loc[2, "Frequency"] == 1
    assert rfm.loc[1, "Monetary"] == 35.0
    assert rfm.loc[2, "Monetary"] == 7.0
