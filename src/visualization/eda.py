"""
Module Phân tích Khám phá Dữ liệu (EDA) và Trực quan hóa cho chuỗi thời gian PM2.5 Hà Nội.
Bao gồm bộ 9 biểu đồ bắt buộc theo Kế hoạch Hoàn Chỉnh.
"""
import os
import warnings
import logging
import matplotlib
# Thiết lập backend Agg để không mở GUI popup (Non-interactive)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.config import TARGET, FIGURES_DIR, PLOT_STYLE, FONT_SIZE

logger = logging.getLogger(__name__)

# Thiết lập style vẽ đồ thị chung
sns.set_style(PLOT_STYLE)
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Helper: Đảm bảo DataFrame có DatetimeIndex."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ['date', 'datetime', 'time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    df.index.name = 'datetime'
    return df


def plot_01_ts_pm25(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """1. Line plot PM2.5 theo thời gian + Rolling Mean 24h & 168h."""
    df = _ensure_datetime_index(df)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Resample daily mean để tránh nhòe khi vẽ hourly raw data dài hạn
    df_daily = df[TARGET].resample('D').mean()
    
    ax.plot(df_daily.index, df_daily.values, color='#bdc3c7', alpha=0.6, label='PM2.5 trung bình ngày')
    ax.plot(df_daily.index, df_daily.rolling(7).mean(), color='#3498db', linewidth=2, label='Đường xu thế 7 ngày')
    ax.plot(df_daily.index, df_daily.rolling(30).mean(), color='#e74c3c', linewidth=2.5, label='Đường xu thế 30 ngày')

    # Ngưỡng nồng độ bụi tương ứng mức AQI EPA (35.5 µg/m³: Kém, 55.5 µg/m³: Xấu)
    ax.axhline(y=35.5, color='#ff7e00', linestyle='--', alpha=0.7, label='Ngưỡng Kém (>35.5 µg/m³)')
    ax.axhline(y=55.5, color='#ff0000', linestyle='--', alpha=0.7, label='Ngưỡng Xấu (>55.5 µg/m³)')

    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Nồng độ PM2.5 (µg/m³)')
    ax.set_title('01. Chuỗi thời gian PM2.5 tại Hà Nội kèm các đường trung bình trượt')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, '01_ts_pm25.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_02_decompose_stl(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """2. Phân tách chuỗi thời gian STL Decomposition (statsmodels)."""
    df = _ensure_datetime_index(df)
    series = df[TARGET].dropna()
    
    # period=24 vì dữ liệu theo giờ (hourly frequency), chu kỳ ngày là 24 giờ
    stl = STL(series, period=24, seasonal=13, robust=True)
    result = stl.fit()
    
    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle('02. STL Decomposition — Nồng độ PM2.5 tại Hà Nội (Hourly)', fontsize=14, y=1.02)
    
    save_path = os.path.join(save_dir, '02_decompose_stl.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_03_box_month(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """3. Boxplot PM2.5 theo tháng để minh họa tính mùa vụ."""
    df = _ensure_datetime_index(df)
    df_plot = df[[TARGET]].copy()
    df_plot['month'] = df_plot.index.month

    fig, ax = plt.subplots(figsize=(12, 6))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
        sns.boxplot(x='month', y=TARGET, hue='month', data=df_plot, ax=ax, palette='coolwarm', legend=False)
    
    ax.set_title('03. Phân phối nồng độ PM2.5 theo từng tháng trong năm (Tính mùa vụ)')
    ax.set_xlabel('Tháng trong năm')
    ax.set_ylabel('Nồng độ PM2.5 (µg/m³)')
    ax.axhline(y=35.5, color='#ff7e00', linestyle='--', alpha=0.5, label='Ngưỡng Kém (35.5)')
    ax.axhline(y=55.5, color='#ff0000', linestyle='--', alpha=0.5, label='Ngưỡng Xấu (55.5)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, '03_box_month.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_04_box_hour(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """4. Boxplot PM2.5 theo giờ trong ngày để phát hiện pattern sinh hoạt."""
    df = _ensure_datetime_index(df)
    df_plot = df[[TARGET]].copy()
    df_plot['hour'] = df_plot.index.hour

    fig, ax = plt.subplots(figsize=(14, 6))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
        sns.boxplot(x='hour', y=TARGET, hue='hour', data=df_plot, ax=ax, palette='husl', legend=False)
    
    ax.set_title('04. Phân phối nồng độ PM2.5 theo giờ trong ngày')
    ax.set_xlabel('Giờ trong ngày')
    ax.set_ylabel('Nồng độ PM2.5 (µg/m³)')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, '04_box_hour.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_05_heatmap_hour_day(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """5. Heatmap PM2.5 trung bình: Giờ × Ngày trong tuần."""
    df = _ensure_datetime_index(df)
    
    pivot = (
        df.groupby([df.index.hour, df.index.dayofweek])[TARGET]
        .mean()
        .unstack()
    )
    pivot.index.name = 'Giờ trong ngày'
    pivot.columns = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, ax=ax, annot=True, fmt='.0f',
                cbar_kws={'label': 'PM2.5 trung bình (µg/m³)'})
    
    ax.set_title('05. PM2.5 Trung Bình theo Giờ và Ngày trong tuần tại Hà Nội', fontsize=14, pad=15)
    ax.set_xlabel('Ngày trong tuần')
    
    save_path = os.path.join(save_dir, '05_heatmap_hour_day.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_06_acf_pacf(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """6. Vẽ đồ thị tự tương quan ACF và PACF (lags 72)."""
    df = _ensure_datetime_index(df)
    series = df[TARGET].dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(series, lags=72, ax=ax1, title='ACF — PM2.5 (lags 0–72h)')
    plot_pacf(series, lags=72, ax=ax2, title='PACF — PM2.5 (lags 0–72h)', method='ywm')
    plt.tight_layout()

    save_path = os.path.join(save_dir, '06_acf_pacf.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_07_corr_spearman(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """7. Heatmap tương quan Spearman giữa các chất ô nhiễm và khí tượng."""
    df = _ensure_datetime_index(df)
    cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'ozone', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'pressure_msl']
    cols_to_plot = [c for c in cols if c in df.columns]
    
    # Tính tương quan Spearman (phi tuyến tính)
    corr = df[cols_to_plot].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('07. Ma trận tương quan Spearman (Chất ô nhiễm & Khí tượng)')

    save_path = os.path.join(save_dir, '07_corr_spearman.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_08_missingness(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """8. Biểu đồ tỷ lệ Missing Values theo tháng của các cột chính."""
    df = _ensure_datetime_index(df)
    cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'ozone', 'temperature_2m']
    cols_to_check = [c for c in cols if c in df.columns]
    
    # Dùng pd.Grouper thay vì to_period để giữ timezone-aware DatetimeIndex
    df_missing = df[cols_to_check].isnull().groupby(pd.Grouper(freq='ME')).mean() * 100
    df_missing.index = df_missing.index.strftime('%Y-%m')

    fig, ax = plt.subplots(figsize=(12, 6))
    df_missing.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', edgecolor='white')
    
    ax.set_title('08. Tỷ lệ Missing Values tích lũy của các biến theo từng tháng')
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Tỷ lệ missing tích lũy (%)')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)

    save_path = os.path.join(save_dir, '08_missingness.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def plot_09_pred_vs_actual(y_true: pd.Series, y_pred: pd.Series, model_name: str, save_dir: str = FIGURES_DIR) -> None:
    """9. Biểu đồ So sánh Predicted vs Actual PM2.5 trên tập Test."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Chỉ vẽ 200 điểm cuối cùng để dễ quan sát chi tiết
    y_true_plot = y_true.tail(200)
    y_pred_plot = y_pred.tail(200)
    
    ax.plot(y_true_plot.index, y_true_plot.values, label='Giá trị thực tế (Actual)', color='#2c3e50', linewidth=2)
    ax.plot(y_true_plot.index, y_pred_plot.values, label=f'Dự đoán ({model_name})', color='#e74c3c', linewidth=1.8, linestyle='--')
    
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Nồng độ PM2.5 (µg/m³)')
    ax.set_title(f'09. So sánh dự báo vs thực tế PM2.5 ({model_name} - 200 giờ cuối)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=15)

    save_path = os.path.join(save_dir, '09_pred_vs_actual.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"💾 Đã lưu: {save_path}")


def generate_all_eda_plots(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """
    Tiện ích chạy sinh toàn bộ 8 biểu đồ EDA khám phá (ngoại trừ 09_pred_vs_actual cần kết quả mô hình).
    
    Parameters:
        df (pd.DataFrame): DataFrame đã processed sạch.
        save_dir (str): Thư mục lưu ảnh.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"🚀 Bắt đầu sinh toàn bộ biểu đồ EDA vào thư mục: {save_dir}")
    
    plot_01_ts_pm25(df, save_dir)
    plot_02_decompose_stl(df, save_dir)
    plot_03_box_month(df, save_dir)
    plot_04_box_hour(df, save_dir)
    plot_05_heatmap_hour_day(df, save_dir)
    plot_06_acf_pacf(df, save_dir)
    plot_07_corr_spearman(df, save_dir)
    plot_08_missingness(df, save_dir)
    
    logger.info("✅ Hoàn tất sinh 8/9 biểu đồ EDA!")
