
# -*- coding: utf-8 -*-
"""Visualization module for customer segmentation pipeline."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visual_style import (
    BLUE as ACCENT_BLUE,
    CLUSTER_COLORS,
    SEQUENTIAL_CMAP as CMAP_SEQ,
)


class DataVisualizer:
    """Visualization manager using predefined design tokens from visual_style."""

    def __init__(self) -> None:
        # Load các màu & cmap có sẵn từ visual_style
        self.cluster_colors = CLUSTER_COLORS
        self.accent_blue = ACCENT_BLUE
        self.cmap_seq = CMAP_SEQ

        # Thiết lập style mặc định cho toàn bộ biểu đồ
        self._setup_style()

    def _setup_style(self) -> None:
        """Apply global matplotlib and seaborn styles."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(self.cluster_colors)

    # =========================================================================
    # IMPLEMENTATION
    # =========================================================================

   

    def plot_revenue_over_time(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12,5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot(color = ACCENT_BLUE)
        plt.title("Daily Revenue")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12,5))
        monthly_revenue = df.groupby(pd.Grouper(key = "InvoiceDate", freq="ME"))["TotalPrice"].sum()
        monthly_revenue.plot(kind="bar", color=ACCENT_BLUE)
        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12,5))
        day_hour_counts = df.groupby(["DayOfWeek","HourOfDay"]).size().unstack(fill_value=0)
        plt.title("Purchase Activity by Day and Hour")
        plt.xlabel("Hour Of Day")
        plt.ylabel("Day Of Week")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df: pd.DataFrame, top_n: int = 10) -> None:
        plt.figure(figsize=(12,5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index, color=self.accent_blue)
        plt.title(f"Top {top_n} Products by Units Sold")
        plt.xlabel("Units Sold")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index, color=self.accent_blue)
        plt.title(f"Top {top_n} Products by Revenue")
        plt.xlabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(12,5))
        transactions_per_cust = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_cust, bins = 30, kde = True, color = self.accent_blue)
        plt.title("Distribution of Transactions per Customer")
        plt.xlabel("Number of Transactions")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()

    def plot_rfm(self, rfm_data: pd.DataFrame) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0], color=self.accent_blue)
        axes[0].set_title("Recency Distribution (Days Since Last Purchase)")
        axes[0].set_xlabel("Days")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1], color=self.accent_blue)
        axes[1].set_title("Frequency Distribution (Number of Transactions)")
        axes[1].set_xlabel("Number of Transactions")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data.loc[monetary_filter, "Monetary"],
            bins=30,
            kde=True,
            ax=axes[2],
            color=self.accent_blue,
        )
        axes[2].set_title("Monetary Distribution (Total Spend)")
        axes[2].set_xlabel("Total Spend (GBP)")

        plt.tight_layout()
        plt.show()