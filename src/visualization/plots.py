"""
Module trực quan hóa dữ liệu US_AQI Hà Nội.
Tất cả hàm vẽ đều hỗ trợ:
  - plt.show()   → hiện popup khi chạy .py
  - plt.savefig() → lưu ảnh vào reports/figures/
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    TARGET, FONT_SIZE, PLOT_STYLE,
    FIGURES_DIR, MODEL_COLORS, MODEL_SHORT_NAMES,
)

# ─── Thiết lập style chung ───
sns.set_style(PLOT_STYLE)
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def _save_or_show(fig, save_path=None, show=True):
    """
    Helper: lưu ảnh và/hoặc hiện biểu đồ.

    Parameters:
        fig: matplotlib Figure.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có gọi plt.show() không.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'   💾 Đã lưu: {save_path}')
    if show:
        plt.show()
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
# 1. BIỂU ĐỒ EDA
# ═════════════════════════════════════════════════════════════

def plot_aqi_timeseries(df, save_path=None, show=True):
    """
    Biểu đồ US AQI theo thời gian với đường trung bình động 30 ngày
    và các ngưỡng mức cảnh báo.

    Parameters:
        df (pd.DataFrame): DataFrame có cột 'date' và TARGET.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có hiện biểu đồ popup không.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(df['date'], df[TARGET], alpha=0.3, color='#e74c3c', label='US AQI hàng ngày')
    ax.plot(df['date'], df[TARGET].rolling(30).mean(),
            color='#c0392b', linewidth=2, label='Trung bình 30 ngày')

    # Ngưỡng AQI
    ax.axhline(y=100, color='#ff7e00', linestyle='--', linewidth=1, alpha=0.7, label='Kém (>100)')
    ax.axhline(y=150, color='#ff0000', linestyle='--', linewidth=1, alpha=0.7, label='Xấu (>150)')

    ax.set_xlabel('Ngày')
    ax.set_ylabel('US AQI')
    ax.set_title('📈 Chỉ số US AQI tại Hà Nội theo thời gian')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path, show)


def plot_correlation_heatmap(df, features, save_path=None, show=True):
    """
    Heatmap tương quan giữa các features và US_AQI.

    Parameters:
        df (pd.DataFrame): DataFrame có các features.
        features (list): Danh sách tên features cần vẽ.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có hiện biểu đồ popup không.
    """
    cols_to_plot = features + [TARGET]
    corr = df[cols_to_plot].corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={'size': 7}
    )
    ax.set_title('📊 Ma trận tương quan giữa các đặc trưng')

    _save_or_show(fig, save_path, show)


def plot_aqi_distribution(df, save_path=None, show=True):
    """
    Phân phối US AQI (histogram + boxplot theo tháng).

    Parameters:
        df (pd.DataFrame): DataFrame có cột TARGET.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có hiện biểu đồ popup không.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df[TARGET], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    axes[0].axvline(df[TARGET].mean(), color='red', linestyle='--', label=f'Mean={df[TARGET].mean():.1f}')
    axes[0].axvline(df[TARGET].median(), color='green', linestyle='--', label=f'Median={df[TARGET].median():.1f}')
    # Ngưỡng AQI
    for bp, color in [(100, '#ff7e00'), (150, '#ff0000')]:
        axes[0].axvline(bp, color=color, linestyle=':', alpha=0.6)
    axes[0].set_title('Phân phối US AQI')
    axes[0].set_xlabel('US AQI')
    axes[0].set_ylabel('Tần suất')
    axes[0].legend()

    # Box plot theo tháng
    if 'month' in df.columns:
        df.boxplot(column=TARGET, by='month', ax=axes[1])
        axes[1].set_title('US AQI theo tháng (mùa vụ)')
        axes[1].set_xlabel('Tháng')
        axes[1].set_ylabel('US AQI')
        plt.suptitle('')
    else:
        sns.kdeplot(df[TARGET], ax=axes[1], fill=True, color='#e74c3c')
        axes[1].set_title('KDE - Phân phối US AQI')

    fig.tight_layout()
    _save_or_show(fig, save_path, show)


# ═════════════════════════════════════════════════════════════
# 2. BIỂU ĐỒ SO SÁNH MÔ HÌNH
# ═════════════════════════════════════════════════════════════

def plot_model_comparison(results, save_path=None, show=True):
    """
    So sánh 3 metrics (MAE, RMSE, R²) giữa các mô hình bằng bar chart.

    Parameters:
        results (list[dict]): Danh sách kết quả từ run_all_models.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có hiện biểu đồ popup không.
    """
    names = [r['name'] for r in results]
    n = len(names)
    # Tạo short names nếu vượt quá danh sách có sẵn
    short = MODEL_SHORT_NAMES[:n] if n <= len(MODEL_SHORT_NAMES) else [n[:8] for n in names]
    colors = MODEL_COLORS[:n] if n <= len(MODEL_COLORS) else plt.cm.Set2(np.linspace(0, 1, n))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, metric, title in [
        (axes[0], 'MAE', 'MAE ↓ (thấp hơn = tốt hơn)'),
        (axes[1], 'RMSE', 'RMSE ↓ (thấp hơn = tốt hơn)'),
        (axes[2], 'R2', 'R² ↑ (cao hơn = tốt hơn)'),
    ]:
        vals = [r[metric] for r in results]
        bars = ax.bar(short, vals, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)

        # Ghi giá trị lên bar
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}' if metric != 'R2' else f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    fig.suptitle('📊 So sánh hiệu suất các mô hình dự báo US AQI', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_actual_vs_predicted(results, dates_test, y_test, save_path=None, show=True):
    """
    Plot Actual vs Predicted cho mô hình tốt nhất và baseline.

    Parameters:
        results (list[dict]): Danh sách kết quả.
        dates_test (pd.Series): Ngày test.
        y_test (pd.Series): Giá trị thực.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có hiện biểu đồ popup không.
    """
    # Tìm mô hình tốt nhất theo R²
    best = max(results, key=lambda x: x['R2'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x_axis = dates_test if dates_test is not None else range(len(y_test))

    # Plot mô hình tốt nhất
    axes[0].plot(x_axis, y_test.values, label='Giá trị thực', color='#2c3e50', linewidth=1.5)
    axes[0].plot(x_axis, best['pred'], label=f'{best["name"]} (R²={best["R2"]:.4f})',
                 color='#e74c3c', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('US AQI')
    axes[0].set_title(f'🏆 Mô hình tốt nhất: {best["name"]}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot Seasonal Naive baseline (hoặc first result)
    baseline = next((r for r in results if 'Naive' in r['name']), results[0])
    axes[1].plot(x_axis, y_test.values, label='Giá trị thực', color='#2c3e50', linewidth=1.5)
    axes[1].plot(x_axis, baseline['pred'], label=f'{baseline["name"]} (R²={baseline["R2"]:.4f})',
                 color='#95a5a6', linewidth=1.5, alpha=0.8)
    axes[1].set_xlabel('Ngày')
    axes[1].set_ylabel('US AQI')
    axes[1].set_title(f'📉 Baseline: {baseline["name"]}')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('📈 Giá trị thực vs Dự đoán (Tập Test)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_feature_importance(model, feature_names, top_n=15, save_path=None, show=True):
    """
    Biểu đồ Feature Importance (cho tree-based models).

    Parameters:
        model: Mô hình đã train (cần có .feature_importances_).
        feature_names (list): Tên các features.
        top_n (int): Số features hiển thị.
        save_path (str, optional): Đường dẫn lưu ảnh.
        show (bool): Có hiện biểu đồ popup không.
    """
    if not hasattr(model, 'feature_importances_'):
        print('⚠️  Mô hình không hỗ trợ feature_importances_')
        return

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind='barh', color='#27ae60', edgecolor='white', ax=ax)
    ax.set_xlabel('Importance')
    ax.set_title(f'🏅 Top {top_n} Feature Importance (US AQI)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path, show)


# ═════════════════════════════════════════════════════════════
# 3. CHẠY TẤT CẢ BIỂU ĐỒ
# ═════════════════════════════════════════════════════════════

def run_all_plots(df, results, dates_test, y_test, features, save=True, show=True):
    """
    Hàm tiện lợi: vẽ tất cả biểu đồ quan trọng.

    Parameters:
        df (pd.DataFrame): DataFrame đã có features.
        results (list[dict]): Kết quả mô hình.
        dates_test: Ngày test.
        y_test: Giá trị thực.
        features (list): Danh sách features.
        save (bool): Có lưu ảnh không.
        show (bool): Có hiện biểu đồ popup không.
    """
    base = FIGURES_DIR if save else None

    print('\n📊 Đang vẽ biểu đồ...\n')

    plot_aqi_timeseries(
        df,
        save_path=os.path.join(base, 'us_aqi_timeseries.png') if base else None,
        show=show
    )

    plot_aqi_distribution(
        df,
        save_path=os.path.join(base, 'us_aqi_distribution.png') if base else None,
        show=show
    )

    plot_correlation_heatmap(
        df, features,
        save_path=os.path.join(base, 'correlation_heatmap.png') if base else None,
        show=show
    )

    plot_model_comparison(
        results,
        save_path=os.path.join(base, 'model_comparison.png') if base else None,
        show=show
    )

    plot_actual_vs_predicted(
        results, dates_test, y_test,
        save_path=os.path.join(base, 'actual_vs_predicted.png') if base else None,
        show=show
    )

    # Feature importance cho mô hình tốt nhất (chỉ tree-based)
    tree_models = [r for r in results if r.get('model') and hasattr(r['model'], 'feature_importances_')]
    if tree_models:
        best_tree = max(tree_models, key=lambda x: x['R2'])
        plot_feature_importance(
            best_tree['model'], features,
            save_path=os.path.join(base, 'feature_importance.png') if base else None,
            show=show
        )

    print('\n✅ Hoàn thành tất cả biểu đồ!')
