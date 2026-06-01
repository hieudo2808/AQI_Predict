"""
Module tạo khả năng diễn giải cho mô hình Machine Learning (Explainability).
Sử dụng thư viện SHAP và tính toán Permutation Importance.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance
from pathlib import Path
from src.config import FIGURES_DIR, PLOT_STYLE, SHAP_MAX_SAMPLES

def create_explain_dir():
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

def plot_permutation_importance(model, X_test, y_test, save_path=None, show=False):
    """
    Tính và vẽ biểu đồ Permutation Importance.
    Với Tree-based, đôi khi SHAP bị nhiễu do Feature mạnh làm che mờ. Permutation Importance
    trực tiếp xáo trộn dữ liệu test để xem mức giảm thiểu R2/RMSE của model độc lập.
    """
    create_explain_dir()
    print("📈 Đang tính toán Permutation Importance...")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_idx = result.importances_mean.argsort()[-15:] # Lấy top 15 tính năng quan trọng nhất

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False
    )
    ax.set_title("Permutation Importances (Test Set)")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"✅ Đã kết xuất Permutation Importance tới {save_path}")

def plot_shap_summary(model, X_train, save_path=None, show=False):
    """
    Sử dụng SHAP TreeExplainer trên dữ liệu XGBoost/LightGBM/RF để vẽ.
    Lưu ý: Không dùng cho Prophet/SARIMAX.
    """
    create_explain_dir()
    print("📈 Đang tạo SHAP Explainer (Sẽ tốn chút thời gian)...")
    try:
        # Nếu là mô hình wrapper của xgboost/LightGBM
        explainer = shap.TreeExplainer(model)
        # Chỉ giới hạn phân tích SHAP trên 1000 mẫu để tăng tốc
        X_sample = X_train.sample(min(SHAP_MAX_SAMPLES, len(X_train)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Summary (Feature contribution to PM2.5 Prediction)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Đã kết xuất SHAP Summary tới {save_path}")
        if show:
            plt.show()
        plt.close()
            
    except Exception as e:
        print(f"⚠️ SHAP Summary Error (Chỉ khả dụng trên Tree Models): {e}")

def plot_shap_waterfall(model, X_train, X_test, y_test, save_path=None, show=False):
    """
    Tạo SHAP waterfall plot cho episode có nồng độ thực tế cao nhất.
    """
    create_explain_dir()
    print("📈 Đang tạo SHAP Waterfall (Cho episode ô nhiễm cao nhất)...")
    try:
        # Tìm dòng có y_test cao nhất
        max_idx = y_test.idxmax()
        actual_val = y_test.loc[max_idx]
        print(f"🔥 Episode ô nhiễm cao nhất vào lúc: {max_idx} với thực tế PM2.5 = {actual_val:.2f} µg/m³")
        
        X_sample = X_test.loc[[max_idx]]
        
        # Tạo TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Nhận Explanation object
        explanation = explainer(X_sample)
        
        # Vẽ
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], show=False)
        plt.title(f'SHAP Waterfall (Episode {max_idx} - PM2.5 {actual_val:.1f} µg/m³)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Đã kết xuất SHAP Waterfall tới {save_path}")
        if show:
            plt.show()
        plt.close()
        
    except Exception as e:
        print(f"⚠️ SHAP Waterfall Error: {e}")

def run_explainability(best_model_name, best_model, X_train, X_test, y_test):
    """
    Entry point sinh biểu đồ giải thích mô hình từ `train.py`.
    Tham số: tên mô hình, instance sklearn, data test.
    """
    print('─' * 60)
    print(f'🕵️‍♀️ EXPLAINABILITY: Phân rã hộp đen mô hình {best_model_name}')
    print('─' * 60)
    
    model = best_model
    model_name = best_model_name
    
    is_tree = any(tree in model_name for tree in ['Random Forest', 'XGBoost', 'LightGBM'])
    
    if not is_tree:
        print(f"⚠️ Mô hình '{model_name}' không hỗ trợ tính SHAP (Chỉ khả dụng với Tree Based).")
        print("🔄 Đang thử load XGBoost model từ models/xgb_t24.json để vẽ SHAP...")
        import xgboost as xgb
        xgb_path = os.path.join("models", "xgb_t24.json")
        if os.path.exists(xgb_path):
            try:
                model = xgb.XGBRegressor()
                model.load_model(xgb_path)
                model_name = "XGBoost (Fallback)"
                is_tree = True
                print("✅ Tải thành công mô hình XGBoost fallback!")
            except Exception as e:
                print(f"❌ Không thể load mô hình XGBoost fallback: {e}")
        else:
            print(f"❌ Không tìm thấy file mô hình XGBoost tại {xgb_path}")

    if is_tree:
        plot_permutation_importance(
            model, X_test, y_test,
            save_path=os.path.join(FIGURES_DIR, "permutation_importance.png")
        )
        
        plot_shap_summary(
            model, X_train,
            save_path=os.path.join(FIGURES_DIR, "10_shap_summary.png")
        )
        
        plot_shap_waterfall(
            model, X_train, X_test, y_test,
            save_path=os.path.join(FIGURES_DIR, "11_shap_waterfall.png")
        )
    else:
        print(f"❌ Bỏ qua vẽ SHAP vì không có mô hình Tree-based khả dụng.")
    
    print('✅ Khâu bóc tách giải thích hoàn tất.')
