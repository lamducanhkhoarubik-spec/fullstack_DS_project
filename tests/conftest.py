# -*- coding: utf-8 -*-
"""Shared fixtures: a compact synthetic retail dataset for smoke tests."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.clustering_library.cleaner import DataCleaner

_MONTHS = [1, 3, 5, 7, 9, 11]
_SKUS = [f"SKU{i:03d}" for i in range(1, 11)]
_UK = "United Kingdom"


def _build_raw_frame() -> pd.DataFrame:
    """8 UK customers with rich behaviour + noise rows that cleaning must drop."""
    rows = []
    inv = 500000
    for c, cid in enumerate(range(10001, 10009)):  # 8 UK customers
        n_invoices = 3 + (c % 3)                   # 3..5 invoices each
        for k in range(n_invoices):
            inv += 1
            month = _MONTHS[(c + k) % len(_MONTHS)]
            day = 1 + ((c + k) % 27)
            hour = 9 + ((c + k) % 8)
            date = f"2011-{month:02d}-{day:02d} {hour:02d}:00:00"
            n_lines = 1 + ((c + k) % 4)             # 1..4 line items
            for line in range(n_lines):
                sku = _SKUS[(c + k + line) % len(_SKUS)]
                rows.append(
                    dict(
                        InvoiceNo=str(inv),
                        StockCode=sku,
                        Description=f"ITEM {sku}",
                        Quantity=1 + ((c + k + line) % 6) * 2,
                        InvoiceDate=date,
                        UnitPrice=round(1.25 + ((c + line) % 5) * 0.75, 2),
                        CustomerID=float(cid),
                        Country=_UK,
                    )
                )

    # Two cancelled invoices (returns)
    for cid in (10001, 10002):
        inv += 1
        rows.append(
            dict(
                InvoiceNo=f"C{inv}",
                StockCode=_SKUS[0],
                Description=f"ITEM {_SKUS[0]}",
                Quantity=-3,
                InvoiceDate="2011-11-15 10:00:00",
                UnitPrice=2.5,
                CustomerID=float(cid),
                Country=_UK,
            )
        )

    # Noise rows that DataCleaner must remove
    noise = [
        dict(InvoiceNo="600001", StockCode=_SKUS[0], Description="FR", Quantity=5,
             InvoiceDate="2011-06-10 10:00:00", UnitPrice=3.0,
             CustomerID=float(20001), Country="France"),          # non-UK
        dict(InvoiceNo="600002", StockCode=_SKUS[1], Description="NAN", Quantity=5,
             InvoiceDate="2011-06-11 10:00:00", UnitPrice=3.0,
             CustomerID=np.nan, Country=_UK),                     # missing CustomerID
        dict(InvoiceNo="600003", StockCode=_SKUS[2], Description="ZEROQ", Quantity=0,
             InvoiceDate="2011-06-12 10:00:00", UnitPrice=3.0,
             CustomerID=float(10001), Country=_UK),               # Quantity <= 0
        dict(InvoiceNo="600004", StockCode=_SKUS[3], Description="ZEROP", Quantity=5,
             InvoiceDate="2011-06-13 10:00:00", UnitPrice=0.0,
             CustomerID=float(10002), Country=_UK),               # UnitPrice <= 0
    ]
    rows.extend(noise)
    return pd.DataFrame(rows)


@pytest.fixture
def raw_excel(tmp_path) -> Path:
    """Tạo file Excel giả lập (.xlsx) cho DataCleaner đọc."""
    path = tmp_path / "raw.xlsx"
    df = _build_raw_frame()
    # Giả lập cột 'Customer ID' có khoảng trắng để test logic chuẩn hóa tên cột
    df = df.rename(columns={"CustomerID": "Customer ID"})
    df.to_excel(path, index=False)
    return path