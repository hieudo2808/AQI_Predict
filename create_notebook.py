import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown ô tiêu đề
header_md = """# 🚀 PM2.5 / US AQI Prediction - Đồ Án Nhập Môn KHDL
Notebook minh họa quy trình End-to-End từ khâu đọc dữ liệu, tiền xử lý, huấn luyện đa mô hình và cuối cùng là giải thích cỗ máy.
Hãy chạy từ trên xuống dưới (Run All) để thưởng thức Pipeline. """

# Import Python code
imports_code = """import os
import sys
# Bổ sung root dir vào system path để import trơn láng
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from src.config import TARGET, FEATURES
print("🚀 Đã load thành công config thư viện!")"""

# Data Processing
data_processing_md = """## 1. Nạp và Chuẩn hóa Dữ liệu (ETL)
Pipeline tự động tải file raw (nếu gọi fetch_open_meteo) và tiến hành resample, missing imputation."""

data_processing_code = """from src.etl.clean_standardize import run_clean_standardize
from src.etl.build_features import build_all_features

# Tải thẳng dữ liệu đã xử lý sạch sẽ (Cho tốc độ nhanh nhất tại Notebook)
processed_path = "../data/processed/dataset.csv"
if os.path.exists(processed_path):
    df_clean = pd.read_csv(processed_path, parse_dates=['date'])
    print(f"✅ Loaded: {df_clean.shape[0]} bản ghi từ {df_clean['date'].min().date()} tới {df_clean['date'].max().date()}")
else:
    print("Vui lòng chạy `python -m src.main` trước để tạo file dataset.csv hoàn chỉnh.")

# Chạy tạo Đặc Trưng (Features)
df_features = build_all_features(df_clean)
display(df_features.tail(3))
"""

# Huấn luyện (Modeling)
modeling_md = """## 2. Huấn Luyện Machine Learning (Modeling)
Tách tập Train / Test và đẩy dữ liệu qua các mô hình Cây Quyết Định cực mạnh (XGBoost/LightGBM)."""

modeling_code = """from src.modeling.train import prepare_data, run_cv_all_models, run_all_models, compare_models
from src.config import DEFAULT_HORIZON

# Thiết lập Horizon mặc định t+1
data_dict = prepare_data(df_features, FEATURES, horizon=DEFAULT_HORIZON)

# Chạy Baseline & ML Models (Không chạy CV trên Notebook để tiết kiệm thời gian)
results = run_all_models(data_dict)

comp_df = compare_models(results)
display(comp_df)
"""

# Giải thích SHAP
explain_md = """## 3. Explaining Output (Khả năng diễn giải hộp đen)
Ứng dụng SHAP (Game Theory API) để lột trần xem mô hình dựa vào đâu để dự đoán hạt ô nhiễm."""

explain_code = """from src.explainability.explain import plot_shap_summary

# Rút trích model đỉnh nhất từ list results
best_model_name = comp_df.index[0]
best_model_dict = next(item for item in results if item["name"] == best_model_name)
best_model = best_model_dict['model']

print(f"🔥 Triển khai SHAP trên ông vua thuật toán: {best_model_name}")
plot_shap_summary(best_model, data_dict['X_train'], save_path=None, show=True)
"""

finish_md = """## 🎉 Hết. 
Phần minh hoạ giao diện cảnh báo thời gian thực xin mời mở Terminal chạy nhánh:
`streamlit run src/ui/app_streamlit.py`
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(header_md),
    nbf.v4.new_code_cell(imports_code),
    nbf.v4.new_markdown_cell(data_processing_md),
    nbf.v4.new_code_cell(data_processing_code),
    nbf.v4.new_markdown_cell(modeling_md),
    nbf.v4.new_code_cell(modeling_code),
    nbf.v4.new_markdown_cell(explain_md),
    nbf.v4.new_code_cell(explain_code),
    nbf.v4.new_markdown_cell(finish_md),
]

import os
os.makedirs("notebooks", exist_ok=True)
with open('notebooks/02_end_to_end_demo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Đã tạo Notebook notebooks/02_end_to_end_demo.ipynb")
