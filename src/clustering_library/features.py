import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

logger = logging.getLogger(__name__)
def _gini(series: pd.Series) -> float:
    arr = series.dropna().to_numpy()
    n = len(arr)
    if n <= 1 or np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n

class FeatureEngineer:
    CANDIDATES: List[str] = [
        # Family 1 — Purchase Rhythm
        "Recency",
        "InterPurchaseCV",
        "ActiveSpanDays",
        "MonthlyOrderRate",
        # Family 2 — Spending Shape
        "AOV",
        "SpendGini",
        "PriceCV",
        # Family 3 — Basket Behavior
        "NewSKURate",
        "RepeatSKUFraction",
        "BasketSizeCV",
        # Family 4 — Volume & Bulk
        "BurstIndex",
        "BulkLineRate",
        "QuantityCV",
        "MedianBasketQty",
        # Family 5 — Product Concentration
        "SKU_HHI",
        "MaxSpendShare",
        # Family 6 — Customer Lifecycle
        "ReturnRate",
        "ReturnValueRate",
        "SpendAcceleration",
        # Family 7 — Seasonality
        "QuarterConcentration",
    ]

    BULK_THRESHOLD: int = 12
    NUM_METHODS: int = 8

    FEATURE_FAMILIES: Dict[str, List[str]] = {
        "Purchase Rhythm": [
            "Recency", "InterPurchaseCV", "ActiveSpanDays", "MonthlyOrderRate",
        ],
        "Spending Shape": [
            "AOV", "SpendGini", "PriceCV", "MaxSpendShare", "SpendAcceleration",
        ],
        "Basket Behaviour": [
            "NewSKURate", "RepeatSKUFraction", "BasketSizeCV",
        ],
        "Volume & Bulk": [
            "BurstIndex", "BulkLineRate", "QuantityCV", "MedianBasketQty", "SKU_HHI",
        ],
        "Customer Lifecycle": [
            "ReturnRate", "ReturnValueRate", "QuarterConcentration",
        ],
    }

    FORCE_KEEP: List[str] = [
        "Recency","MonthlyOrderRate","ReturnRate",
        "QuarterConcentration", "SKU_HHI",
    ]

    def __init__(
        self,
        data_path: str,
        raw_data_path: Optional[str] = None,
        corr_threshold: float = 0.85,
        force_keep: Optional[List[str]] = None,
    ) -> None:
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.corr_threshold = corr_threshold
        self.force_keep: List[str] = self.FORCE_KEEP if force_keep is None else force_keep
        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.customer_feature_candidates: Optional[pd.DataFrame] = None
        self.customer_features: Optional[pd.DataFrame] = None
        self.customer_features_transformed: Optional[pd.DataFrame] = None
        self.customer_features_scaled_unweighted: Optional[pd.DataFrame] = None
        self.customer_features_scaled: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.qt: Optional[QuantileTransformer] = None
        self.feature_customer: List[str] = []
        self.family_weights: Dict[str, float] = {}

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        logger.info("Cleaned dataset shape: %s", self.df.shape)

        if self.raw_data_path and os.path.exists(self.raw_data_path):
            self.raw_df = pd.read_csv(
                self.raw_data_path, encoding="ISO-8859-1", low_memory=False
            )
            self.raw_df["InvoiceDate"] = pd.to_datetime(self.raw_df["InvoiceDate"])
            logger.info("Raw dataset shape: %s", self.raw_df.shape)
        elif self.raw_data_path:
            logger.warning("raw_data_path not found (%s). ReturnRate will be 0.", self.raw_data_path)
        return self.df

    def _compute_purchase_rhythm(self, df: pd.DataFrame) -> pd.DataFrame:
        temp_df = df[["CustomerID","InvoiceDate"]].copy()
        ref_date = temp_df["InvoiceDate"].max() + pd.Timedelta(days = 1)

        temp_df = temp_df.drop_duplicates(subset=["CustomerID", "InvoiceDate"])
        temp_df = temp_df.sort_values(by = ['CustomerID',"InvoiceDate"])

        temp_df['days_since_last'] = temp_df.groupby('CustomerID')["InvoiceDate"].diff().dt.days
        
        rhythm_df = temp_df.groupby('CustomerID').agg(
        total_orders=('InvoiceDate', 'count'),
        first_purchase=('InvoiceDate', 'min'),
        last_purchase=('InvoiceDate', 'max'),
        avg_inter_days=('days_since_last', 'mean'),
        std_inter_days=('days_since_last', 'std')
    ).reset_index()

        rhythm_df['recency'] = (ref_date - rhythm_df['last_purchase']).dt.days
        rhythm_df['active_span'] = (rhythm_df['last_purchase'] - rhythm_df['first_purchase']).dt.days

        rhythm_df['inter_cv'] = np.where(
        (rhythm_df['avg_inter_days'] > 0) & (rhythm_df['std_inter_days'].notnull()),
        rhythm_df['std_inter_days'] / rhythm_df['avg_inter_days'],
        0.0
    )

        months_active = rhythm_df['active_span'] / 30
        rhythm_df['monthly_rate'] = np.where(
           months_active > 0,
           rhythm_df['total_orders'] / months_active,
           rhythm_df['total_orders']
    )
        rhythm_df['avg_inter_days'] = rhythm_df['avg_inter_days'].round(1)
        rhythm_df['inter_cv'] = rhythm_df['inter_cv'].round(2)
        rhythm_df['monthly_rate'] = rhythm_df['monthly_rate'].round(2)

    # Tra về đúng các cột cần thiết
        return rhythm_df[[
        'CustomerID', 
        'total_orders', 
        'recency', 
        'inter_cv', 
        'active_span', 
        'monthly_rate'
    ]]

    def _compute_spend_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
    
      CUST_COL = 'CustomerID'   
      AMOUNT_COL = 'TotalPrice' 
      PRICE_COL = 'UnitPrice'   
    
    # -------------------------------------------------------------
    # 1. GOM NHÓM & TÍNH CÁC CHỈ SỐ CƠ BẢN
    # -------------------------------------------------------------
      result = df.groupby(CUST_COL).agg(
        # AOV = Giá trị trung bình mỗi đơn hàng
         AOV=(AMOUNT_COL, 'mean'),
        
        # SpendGini = Mức độ bất bình đẳng/biến động trong tổng chi tiêu
         SpendGini=(AMOUNT_COL, _gini),
        
        # Biến trung gian để tính PriceCV
         _price_std=(PRICE_COL, 'std'),
         _price_mean=(PRICE_COL, 'mean')
    )
    
    
      result['PriceCV'] = np.where(
         result['_price_mean'] > 0,
         result['_price_std'] / result['_price_mean'],
        0.0
    )
    
    # -------------------------------------------------------------
    # 3. LÀM TRÒN & CLEANUP ĐẦU RA
    # -------------------------------------------------------------
      cols_to_round = ['AOV', 'SpendGini', 'PriceCV']
      result[cols_to_round] = result[cols_to_round].fillna(0).round(2)
    
      return result.reset_index()[[CUST_COL, 'AOV', 'SpendGini', 'PriceCV']]

    def _compute_basket_behavior(self,df: pd.DataFrame) -> pd.DataFrame:
    
    # TODO: SỬA TÊN CỘT Ở ĐÂY NẾU CẦN
    # -------------------------------------------------------------
      CUST_COL = "CustomerID"
      INV_COL  = "InvoiceNo"
      DATE_COL = "InvoiceDate"
      SKU_COL  = "StockCode"

    # 1. Chuẩn bị dữ liệu tập hợp SKU theo từng đơn hàng (đã sắp xếp theo thời gian)
      inv_sku = (
        df.groupby([CUST_COL, INV_COL])
        .agg(date=(DATE_COL, "min"), skus=(SKU_COL, lambda x: frozenset(x)))
        .reset_index()
        .sort_values([CUST_COL, "date"])
    )

    # 2. Tính NewSKURate
      def _new_sku_rate(group: pd.DataFrame) -> float:
        seen: set = set()
        rates = []
        for skus in group["skus"]:
            new = skus - seen
            rates.append(len(new) / len(skus))
            seen.update(skus)
        return float(np.mean(rates)) if rates else 1.0

      new_sku_rate = (
        inv_sku.groupby(CUST_COL, group_keys=False)
        .apply(_new_sku_rate, include_groups=False)
        .rename("NewSKURate")
    )

    # 3. Tính RepeatSKUFraction
      sku_inv_count = df.groupby([CUST_COL, SKU_COL])[INV_COL].nunique()
      repeat_frac = (
        (sku_inv_count > 1).groupby(level=CUST_COL).mean().rename("RepeatSKUFraction")
    )

    # 4. Tính BasketSizeCV
      basket_size = (
        df.groupby([CUST_COL, INV_COL])[SKU_COL]
        .nunique()
        .reset_index(name="n_skus")
    )
      bs_agg = basket_size.groupby(CUST_COL)["n_skus"].agg(["mean", "std"])
      basket_cv = (bs_agg["std"] / bs_agg["mean"]).fillna(0.0).rename("BasketSizeCV")

    # 5. Gộp các chỉ số trả về DataFrame
      result = pd.concat([new_sku_rate, repeat_frac, basket_cv], axis=1).reset_index()
      return result
    def _compute_volume_bulk(self, df: pd.DataFrame) -> pd.DataFrame:
        inv_spend = (
            df.groupby(["CustomerID","InvoiceNo"])["TotalPrice"]
            .sum()
            .reset_index(name = "InvoiceTotalPrice")

        )
        inv_agg = inv_spend.groupby("CustomerID")["InvoiceTotalPrice"].agg(["max","mean"])
        burst_idx = (inv_agg['max'] / inv_agg['mean']).rename("BurstIndex")

        df_bulk = df.assign(is_bulk = (df["Quantity"] > self.BULK_THRESHOLD).astype('float32'))
        burst_line_rate = df_bulk.groupby("CustomerID")["is_bulk"].mean().rename("BulkLineRate")

        quantity_metrics = df.groupby("CustomerID")['Quantity'].agg(["std","mean"])
        quantity_cv = (quantity_metrics['std'] / quantity_metrics['mean']).rename("QuantityCV")

        result = pd.concat([burst_idx, burst_line_rate,quantity_cv], axis = 1)
        return result

    def _compute_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
           sku_spend = df.groupby(["CustomerID", "StockCode"])["TotalPrice"].sum()
           cust_total = sku_spend.groupby(level = "CustomerID").transform("sum")
           sku_share = sku_spend / cust_total
           sku_hii = (sku_share ** 2).groupby(level = "CustomerID").sum().rename("SKU_HHI")
    
           inv_spend = (
              df.groupby(["CustomerID","InvoiceNo"])["TotalPrice"]
              .sum()
              .reset_index(name = 'inv_total')
           )
           inv_spend['cust_total'] = inv_spend.groupby("CustomerID")["inv_total"].transform("sum")
           max_spend_share = (
              (inv_spend["inv_total"] / inv_spend['cust_total'])
              .groupby(inv_spend["CustomerID"])
              .max()
              .rename("MaxSpendShare")
           )
           return pd.concat([sku_hii, max_spend_share], axis = 1)
    
    def _compute_lifecycle(self, df: pd.DataFrame,customer_ids: pd.Index) -> pd.DataFrame:
           
            if self.raw_df is not None:
                raw_uk = self.raw_df[
                    (self.raw_df["Country"] == "United Kingdom")
                    & (self.raw_df["CustomerID"].notna())
                ].copy()
                raw_uk["CustomerID"] = raw_uk["CustomerID"].astype(float).astype(int)
                raw_uk["InvoiceNo"] = raw_uk["InvoiceNo"].astype(str)
                raw_uk["is_cancelled"] = raw_uk["InvoiceNo"].str.startswith("C")
                raw_uk["AbsValue"] = (raw_uk["Quantity"] * raw_uk["UnitPrice"]).abs()
    
                inv_flags = (
                    raw_uk.groupby(["CustomerID", "InvoiceNo"])["is_cancelled"]
                    .first()
                    .reset_index()
                )
                agg_count = inv_flags.groupby("CustomerID")["is_cancelled"].agg(
                    n_cancelled="sum", n_total="count"
                )
                return_rate = (agg_count["n_cancelled"] / agg_count["n_total"]).rename("ReturnRate")
                return_rate = return_rate.reindex(customer_ids).fillna(0.0)
    
                cust_cancel_val = (
                    raw_uk[raw_uk["is_cancelled"]]
                    .groupby("CustomerID")["AbsValue"]
                    .sum()
                )
                cust_total_val = raw_uk.groupby("CustomerID")["AbsValue"].sum()
                return_value_rate = (cust_cancel_val / cust_total_val).rename("ReturnValueRate")
                return_value_rate = return_value_rate.reindex(customer_ids).fillna(0.0)
            else:
                logger.warning("raw_data_path not set — ReturnRate and ReturnValueRate = 0.")
                return_rate = pd.Series(0.0, index=customer_ids, name="ReturnRate")
                return_value_rate = pd.Series(0.0, index=customer_ids, name="ReturnValueRate")
    
            return pd.concat([return_rate, return_value_rate], axis=1)
    def _compute_spend_acceleration(
        self, df: pd.DataFrame, customer_ids: pd.Index
    ) -> pd.Series:
        """Spending trajectory: (last-3-month avg) / (first-3-month avg) - 1, clipped to [-3, 3].

        Customers with fewer than 3 distinct months of history get 0.0.
        """
        monthly = (
            df.assign(YearMonth=df["InvoiceDate"].dt.to_period("M"))
            .groupby(["CustomerID", "YearMonth"])["TotalPrice"]
            .sum()
            .reset_index(name="MonthlySpend")
        )

        def _accel(group: pd.DataFrame) -> float:
            spend = group.sort_values("YearMonth")["MonthlySpend"].values
            if len(spend) < 3:
                return 0.0
            first_avg = spend[:3].mean()
            last_avg = spend[-3:].mean()
            raw = (last_avg - first_avg) / (abs(first_avg) + 1e-9)
            return float(np.clip(raw, -3.0, 3.0))

        accel = (
            monthly.groupby("CustomerID")
            .apply(_accel, include_groups=False)
            .rename("SpendAcceleration")
        )
        return accel.reindex(customer_ids).fillna(0.0)

    def _compute_quarter_concentration(
        self, df: pd.DataFrame, customer_ids: pd.Index
    ) -> pd.Series:
        """Max quarter spend fraction — captures seasonal concentration.

        Values in [0.25, 1.0]: 0.25 = uniform across quarters, 1.0 = all spend
        in one quarter. Customers absent from df get 0.25 (no concentration).
        """
        q_df = df.copy()
        q_df["Quarter"] = q_df["InvoiceDate"].dt.quarter

        cust_total = q_df.groupby("CustomerID")["TotalPrice"].sum()
        cust_qmax = (
            q_df.groupby(["CustomerID", "Quarter"])["TotalPrice"]
            .sum()
            .groupby("CustomerID")
            .max()
        )

        conc = (cust_qmax / cust_total.clip(lower=1e-9)).rename("QuarterConcentration")
        conc = conc.clip(0.25, 1.0)
        return conc.reindex(customer_ids).fillna(0.25)
    def create_customer_features(self) -> pd.DataFrame:
        df = self.df

        logger.info(f"[1/{self.NUM_METHODS}] Purchase Rhythm...")
        rhythm = self._compute_purchase_rhythm(df)
        logger.info(f"[2/{self.NUM_METHODS}] Spending Shape...")
        spending = self._compute_spend_metrics(df)
        logger.info(f"[3/{self.NUM_METHODS}] Basket Behavior...")
        basket = self._compute_basket_behavior(df)
        logger.info(f"[4/{self.NUM_METHODS}] Volume & Bulk...")
        volume = self._compute_volume_bulk(df)
        

    # Đảm bảo CustomerID luôn nằm ở Index trước khi join
        def _ensure_index(d):
           return d.set_index("CustomerID") if "CustomerID" in d.columns else d

        rhythm = _ensure_index(rhythm)
        spending = _ensure_index(spending)
        basket = _ensure_index(basket)
        volume = _ensure_index(volume)

        base = (
         rhythm.join(spending, how="outer")
        .join(basket, how="outer")
        .join(volume, how="outer")
    )
        logger.info(f"[5/{self.NUM_METHODS}] Product Concentration...")
        concentration = self._compute_concentration(df)
        logger.info(f"[6/{self.NUM_METHODS}] Customer Lifecycle (ReturnRate + ReturnValueRate)...")
        lifecycle = self._compute_lifecycle(df, base.index)
        logger.info(f"[7/{self.NUM_METHODS}] Acceleration Concentration...")
        acceleration = self._compute_spend_acceleration(df, base.index)
        logger.info(f"[8/{self.NUM_METHODS}] Quarter Concentration...")
        quarter = self._compute_quarter_concentration(df, base.index)
        all_candidates = base.copy()

    # 1. Chỉ lấy cột khai báo trong CANDIDATES (Nếu rỗng/không khớp thì lấy hết các cột)
        valid_cols = [c for c in self.CANDIDATES if c in all_candidates.columns]
        if not valid_cols:
          logger.warning(
            "Không cột nào trong CANDIDATES khớp! Tự động lấy tất cả các cột hiện có."
        )
        valid_cols = list(all_candidates.columns)

        all_candidates = all_candidates[valid_cols]

    # 2. Xử lý các giá trị NaN phát sinh
        n_missing = all_candidates.isnull().sum().sum()
        if n_missing > 0:
          logger.warning(
            "%d NaN values — filling with column medians.", n_missing
        )
        all_candidates = all_candidates.fillna(all_candidates.median())

        logger.info(
        "Computed %d candidates for %d customers",
        len(valid_cols),
        len(all_candidates),
    )
        self.customer_feature_candidates = all_candidates.copy()

        provisional_qt = QuantileTransformer(
           output_distribution="normal",
           n_quantiles=min(500, len(all_candidates)),
           random_state=42,
        )
        provisional = provisional_qt.fit_transform(all_candidates)
        provisional = StandardScaler().fit_transform(provisional)
        provisional_df = pd.DataFrame(
           np.clip(provisional, -3.0, 3.0),
           columns = all_candidates.columns,
           index = all_candidates.index,
        )
    # 3. Vì chưa dùng _filter_by_correlation, giữ lại toàn bộ cột hợp lệ
        self.feature_customer = self._filter_by_correlaton(provisional_df)
        self.customer_features = all_candidates[
        self.feature_customer
    ].reset_index()

        logger.info(
        "Final feature set (%d features):", len(self.feature_customer)
    )
        for f in self.feature_customer:
           logger.info("    %s", f)

        return self.customer_features
    
    

    def _filter_by_correlaton(self, df: pd.DataFrame) -> List[str]:
       corr_matrix = df.corr(method = "pearson").abs()
       upper_tri = corr_matrix.where(
          np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)
       )
       to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > self.corr_threshold)]
       selected_features = [c for c in df.columns if c not in to_drop]

       dropped_count = len(to_drop)
       if (dropped_count) > 0:
          logger.info(
             "Dropped %d features due to the high correlation (> %.2f): %s",
             dropped_count,
             self.corr_threshold,
             to_drop,
          )
       return selected_features

    def transform_features(self) -> pd.DataFrame:
        indexed = self.customer_features.set_index("CustomerID")

        self.qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=500,
            random_state=42

        )
        transformed = self.qt.fit_transform(indexed.values)
        self.customer_features_transformed = pd.DataFrame(
            transformed,
            columns = self.feature_customer,
            index=indexed.index
        )
        logger.info("QuantileTransformer applied")
        return self.customer_features_transformed

    def _build_family_weights(self) -> Dict[str, float]:
        feature_to_family = {}
        for family, features in self.FEATURE_FAMILIES.items():
           for feature in features:
              feature_to_family[feature] = family

    # 2. Đếm số lượng feature trong từng nhóm (Bọc lót .get() để gán 'Other' nếu thiếu)
        feature_families = [
            feature_to_family.get(f, "Other") for f in self.feature_customer
    ]
        counts = pd.Series(feature_families).value_counts()

    # 3. Tính trọng số 1 / sqrt(k) cho từng feature
        weights = {}
        for feature in self.feature_customer:
            family = feature_to_family.get(feature, "Other")
        # k là số lượng cột thuộc nhóm đó
            k = counts[family]
            weights[feature] = float(1.0 / np.sqrt(k))

        return weights

    def scale_features(self) -> pd.DataFrame:
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.customer_features_transformed)

        scaled_df = pd.DataFrame(
            scaled,
            columns = self.feature_customer,
            index = self.customer_features_transformed.index

        )
        clipped = scaled_df.clip(lower=-3.0, upper = 3.0)
        self.customer_features_scaled_unweighted = clipped
        self.family_weights = self._build_family_weights()
        weighted= clipped.mul(pd.Series(self.family_weights), axis = 1)
        self.customer_features_scaled = weighted
        logger.info("Feature scaling complete")
        return weighted
    def save_pipeline(self, output_dir: str = "../models") -> None:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Run scale_features() first.")
        if self.qt is None:
            raise RuntimeError("QuantileTransformer not fitted. Run transform_features() first.")

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.qt,     f"{output_dir}/quantile_transformer.pkl")

        metadata: Dict = {
            "bulk_qty_threshold": self.BULK_QTY_THRESHOLD,
            "feature_order":      self.feature_customer,
            "n_candidates":       len(self.CANDIDATES),
            "corr_threshold":     self.corr_threshold,
            "correlation_space": "quantile_transformed_scaled_clipped",
            "post_scale_clip":    3.0,
            "family_weights":     self.family_weights,
            "feature_families":   self.FEATURE_FAMILIES,
        }
        with open(f"{output_dir}/pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Pipeline artifacts saved to: %s/", output_dir)

    def save_features(self, output_dir: str = "../data/processed") -> None:
        os.makedirs(output_dir, exist_ok=True)

        self.customer_feature_candidates.to_csv(
            f"{output_dir}/customer_feature_candidates.csv"
        )
        self.customer_features.set_index("CustomerID").to_csv(
            f"{output_dir}/customer_features.csv"
        )
        self.customer_features_transformed.to_csv(
            f"{output_dir}/customer_features_transformed.csv"
        )
        self.customer_features_scaled_unweighted.to_csv(
            f"{output_dir}/customer_features_scaled_unweighted.csv"
        )
        self.customer_features_scaled.to_csv(
            f"{output_dir}/customer_features_scaled.csv"
        )
        logger.info("All features saved to: %s", output_dir)

        models_dir = os.path.join(os.path.dirname(output_dir), "models")
        self.save_pipeline(models_dir)