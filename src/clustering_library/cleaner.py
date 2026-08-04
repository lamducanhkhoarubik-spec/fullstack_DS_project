import logging
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataCleaner:
    """Class chuyên đảm nhận nhiệm vụ đọc, làm sạch và lưu dữ liệu."""

    def __init__(self, file_path: Union[str, Path]):
        """Khởi tạo DataCleaner với đường dẫn file đầu vào."""
        self.file_path = Path(file_path)
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_uk: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Đọc dữ liệu từ file Excel vào DataFrame và chuẩn hóa CustomerID."""
        logging.info(f"Đang tải dữ liệu từ: {self.file_path}")
        if not self.file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {self.file_path}")

        # 1. Đọc file Excel
        self.df = pd.read_excel(self.file_path)

        # 2. Chuẩn hóa tên cột (loại bỏ khoảng trắng & đổi 'Customer ID' thành 'CustomerID')
        self.df.columns = self.df.columns.astype(str).str.strip()
        if "Customer ID" in self.df.columns:
            self.df = self.df.rename(columns={"Customer ID": "CustomerID"})

        # 3. Biến đổi cột CustomerID
        if "CustomerID" in self.df.columns:
            self.df["CustomerID"] = self.df["CustomerID"].apply(
                lambda x: str(int(x)).zfill(6) if pd.notna(x) else np.nan
            )
        else:
            logging.warning(f"Không tìm thấy cột 'CustomerID'! Các cột hiện có: {list(self.df.columns)}")

        # 4. Log thông tin dữ liệu
        logging.info("Dataset shape: %s", self.df.shape)
        logging.info("Number of records: %s rows", f"{len(self.df):,}")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Thực hiện lọc rác, loại bỏ đơn hủy và chọn thị trường UK."""
        if self.df.empty:
            raise ValueError("Chưa có dữ liệu! Hãy gọi load_data() trước.")

        logging.info("Đang tiến hành làm sạch dữ liệu...")

        # Tính tổng tiền
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Loại bỏ đơn hủy (InvoiceNo bắt đầu bằng 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # Lọc thị trường UK
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Bỏ dòng thiếu CustomerID và dòng có Quantity/UnitPrice <= 0
        self.df_uk = self.df_uk.dropna(subset=["CustomerID"])
        self.df_uk = self.df_uk[(self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)]

        logging.info("Làm sạch hoàn tất! Kích thước dữ liệu UK: %s", self.df_uk.shape)
        return self.df_uk

    def transform_datetime(self, column_name: str = "InvoiceDate") -> None:
        """Tạo các đặc trưng thời gian (DayOfWeek, HourOfDay) trên dữ liệu df_uk."""
        if self.df_uk.empty:
            raise ValueError("Chưa có dữ liệu df_uk! Hãy gọi clean_data() trước.")

        # Xử lý tên cột dính khoảng trắng
        if column_name not in self.df_uk.columns and "Invoice Date" in self.df_uk.columns:
            self.df_uk.rename(columns={"Invoice Date": column_name}, inplace=True)

        if column_name not in self.df_uk.columns:
            raise KeyError(f"Cột '{column_name}' không có trong DataFrame. Các cột hiện có: {list(self.df_uk.columns)}")

        # Đảm bảo ép kiểu datetime
        self.df_uk[column_name] = pd.to_datetime(self.df_uk[column_name], errors="coerce")

        # Tạo các đặc trưng thời gian trực tiếp lên self.df_uk
        self.df_uk["DayOfWeek"] = self.df_uk[column_name].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk[column_name].dt.hour

        logging.info("Đã tạo xong các đặc trưng thời gian (DayOfWeek, HourOfDay)!")

    def save_to_csv(self, output_path: Union[str, Path] = "data/processed/cleaned_data.csv", index: bool = False) -> Path:
        """Lưu DataFrame df_uk ra file CSV."""
        if self.df_uk.empty:
            raise ValueError("Không có dữ liệu df_uk để lưu!")

        out_path = Path(output_path)

        # Tự động tạo thư mục cha nếu chưa có (vd: data/processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Đang lưu dữ liệu đã làm sạch vào: {out_path}")
        self.df_uk.to_csv(out_path, index=index, encoding="utf-8-sig")
        logging.info("Lưu file thành công!")
        return out_path


if __name__ == "__main__":
    # Luồng chạy chuẩn chỉnh
    cleaner = DataCleaner(file_path="data/raw/Online Retail.xlsx")

    cleaner.load_data()
    cleaner.clean_data()
    cleaner.transform_datetime(column_name="InvoiceDate")
    cleaner.save_to_csv(output_path="data/processed/cleaned_data.csv")