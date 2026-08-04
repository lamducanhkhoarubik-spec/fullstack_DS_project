import pytest
from src.clustering_library.cleaner import DataCleaner

def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DataCleaner("/definitely/not/a/real/path.xlsx").load_data()

def test_clean_data_applies_all_rules(raw_excel):
    cleaner = DataCleaner(str(raw_excel))
    cleaner.load_data()
    df = cleaner.clean_data()

    assert not df.empty
    assert not df["InvoiceNo"].astype(str).str.startswith('c').any()
    assert (df["Country"] == "United Kingdom").all()
    assert df["CustomerID"].notna().all()
    assert (df["Quantity"] > 0).all()
    assert (df["UnitPrice"] > 0).all()
    assert "TotalPrice" in df.columns
    assert (df["TotalPrice"] == df["Quantity"] * df["UnitPrice"]).all()


def test_transform_datetime(raw_excel):
    cleaner = DataCleaner(str(raw_excel))
    cleaner.load_data()
    cleaner.clean_data()
    cleaner.transform_datetime(column_name="InvoiceDate")
    df = cleaner.df_uk
    assert df["DayOfWeek"].between(0, 6).all()
    assert df["HourOfDay"].between(0,23).all()