"""
App Streamlit cung cấp Giao diện Điều khiển (Dashboard) chất lượng không khí PM2.5 (US AQI).
Tích hợp Bản đồ Folium, Hệ thống Cảnh báo, và Explainability.
Command chạy: streamlit run src/ui/app_streamlit.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from streamlit_folium import folium_static

# Fix đường dẫn tuyệt đối cho import nếu chạy ngoài root dir
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ui.map_heatmap import create_aqi_heatmap
from src.config import TARGET, FIGURES_DIR, FEATURES, DEFAULT_HORIZON
from src.aqi_formula import pm25_to_aqi, aqi_to_level

# Config giao diện Streamlit
st.set_page_config(page_title="Hanoi AQI Live & Forecast Dashboard", page_icon="🌤️", layout="wide")

# ─── 1. HÀM LOAD DỮ LIỆU VÀ MÔ HÌNH ───
@st.cache_data
def load_data():
    """Nạp dữ liệu từ Parquet features/targets sạch tần suất Hourly."""
    path = "data/features/features_targets.parquet"
    if not os.path.exists(path):
        # Fallback nếu không có parquet thì đọc dataset processed
        path = "data/processed/aq_hourly_clean.parquet"
        if not os.path.exists(path):
            st.error(f"❌ Không tìm thấy dữ liệu tại data/features/ hoặc data/processed/. Vui lòng chạy pipeline trước.")
            return pd.DataFrame()
            
    df = pd.read_parquet(path)
    # Robustly flatten any DatetimeIndex (named 'datetime', 'date', or unnamed)
    if isinstance(df.index, pd.DatetimeIndex):
        index_col_name = df.index.name or 'datetime'
        df = df.reset_index().rename(columns={index_col_name: 'date'})
    elif 'date' not in df.columns and df.index.name:
        df = df.reset_index().rename(columns={df.index.name: 'date'})
    elif 'date' not in df.columns:
        df = df.reset_index()
        df.columns = ['date'] + list(df.columns[1:])

    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('Asia/Ho_Chi_Minh')
    return df

@st.cache_resource
def load_horizon_model(horizon):
    """Nạp mô hình XGBoost (hoặc SARIMA fallback) tương ứng cho horizon giờ."""
    path = f"models/xgb_t{horizon}.json"
    if os.path.exists(path):
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model, "XGBoost (JSON)"
        
    path_pkl = f"models/sarima_t{horizon}.pkl"
    if os.path.exists(path_pkl):
        return joblib.load(path_pkl), "SARIMA (PKL)"
        
    return None, None

df = load_data()

# ─── 2. HEADER DỰ ÁN ───
st.title("🛡️ Hệ Thống Live-Monitor & Dự Báo Ô Nhiễm PM2.5 Hà Nội")
st.markdown("Giám sát chất lượng không khí thời gian thực và cung cấp cảnh báo AI đa khung thời gian (Multi-Horizon).")

if df.empty:
    st.warning("⚠️ Dữ liệu đang được cập nhật, vui lòng chạy pipeline chính để khởi tạo.")
    st.stop()

# Lấy thông tin quan trắc mới nhất
latest_row = df.iloc[-1]
latest_date = latest_row['date'].strftime('%d-%m-%Y %H:%M')
latest_pm25 = latest_row['pm2_5']

# Quy đổi AQI thời gian thực
latest_aqi = pm25_to_aqi(latest_pm25)
latest_level, latest_color = aqi_to_level(latest_aqi)

# Thiết lập chữ trắng cho nền tối, chữ đen cho nền sáng
text_color = '#ffffff' if latest_color in ['#ff0000', '#8f3f97', '#7e0023'] else '#000000'

# Hiển thị Panel Cảnh báo chất lượng cao
st.markdown(f"""
<div style="background-color: {latest_color}; padding: 22px; border-radius: 12px; text-align: center; color: {text_color}; margin-bottom: 20px;">
    <p style="margin: 0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">Chỉ số US AQI Hiện Tại (Hà Nội)</p>
    <h1 style="margin: 5px 0; font-size: 3.8rem; color: {text_color};">{latest_aqi}</h1>
    <h2 style="margin: 0; font-weight: bold; color: {text_color};">{latest_level}</h2>
    <p style="margin: 8px 0 0 0; font-size: 1.15rem;">Nồng độ bụi mịn PM2.5 thực tế đo được: <b>{latest_pm25:.1f} µg/m³</b> (Cập nhật lúc {latest_date})</p>
</div>
""", unsafe_allow_html=True)

# ─── 3. KHỐI KHUYẾN NGHỊ SỨC KHỎE CHI TIẾT ───
advice_dict = {
    'Good': "🌱 Chất lượng không khí tuyệt vời và an toàn cho mọi người. Bạn nên tham gia các hoạt động thể dục thể thao ngoài trời.",
    'Moderate': "💡 Chất lượng không khí ở mức chấp nhận được. Tuy nhiên, người cực kỳ nhạy cảm với ô nhiễm (hen suyễn, hô hấp yếu) nên hạn chế hoạt động ngoài trời kéo dài nếu cảm thấy mệt mỏi.",
    'Unhealthy for Sensitive': "⚠️ Nhóm nhạy cảm (trẻ em, người già, phụ nữ mang thai và người có bệnh tim mạch/hô hấp) nên hạn chế hoạt động mạnh ngoài trời kéo dài. Nên đeo khẩu trang lọc bụi khi đi ra đường.",
    'Unhealthy': "🚨 Mức độ ô nhiễm có hại cho sức khỏe cộng đồng. Tất cả mọi người nên hạn chế hoạt động thể lực ngoài trời quá lâu. Bắt buộc đeo khẩu trang N95 khi di chuyển bên ngoài.",
    'Very Unhealthy': "☣️ Cảnh báo nghiêm trọng về sức khỏe. Tất cả mọi người nên tránh hoạt động ngoài trời hoàn toàn. Đóng các cửa sổ trong phòng và sử dụng máy lọc không khí trong nhà.",
    'Hazardous': "💀 Tình trạng khẩn cấp toàn diện. Tránh ra ngoài tuyệt đối. Mọi người nên ở trong nhà kín, bật máy lọc không khí và chuẩn bị các biện pháp y tế hỗ trợ nếu cần thiết."
}
st.info(advice_dict.get(latest_level, "Không có khuyến nghị cụ thể cho mức này."))

# ─── 4. METRICS BOARD ───
st.markdown("### 📊 Chỉ Số Đo Lường Nhanh")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Thời Gian Cập Nhật", latest_date)

# Tính chênh lệch so với 24 giờ trước
if len(df) > 24:
    prev_pm25 = df.iloc[-25]['pm2_5']
    prev_aqi = pm25_to_aqi(prev_pm25)
    delta_aqi = latest_aqi - prev_aqi
    col2.metric("Chỉ số US AQI", f"{latest_aqi}", delta=f"{delta_aqi:+.0f} so với 24h trước", delta_color="inverse")
else:
    col2.metric("Chỉ số US AQI", f"{latest_aqi}")

col3.metric("Bụi mịn PM2.5", f"{latest_pm25:.1f} µg/m³")

# Kiểm tra sự hiện diện của các mô hình
model_status = "Sẵn sàng (3 Horizons)" if os.path.exists("models/xgb_t24.json") else "Chưa huấn luyện"
col4.metric("Hệ Thống Trí Tuệ Nhân Tạo (AI)", model_status)

st.markdown("---")

# ─── 5. MULTI-HORIZON FORECAST ENGINE (AI DỰ BÁO) ───
st.subheader("🔮 Trạm Dự Báo Đa Khung Thời Gian (AI Forecast Engine)")
st.caption("Chọn horizon để nạp mô hình XGBoost/SARIMA tương ứng và dự báo nồng độ bụi mịn PM2.5 tiếp theo.")

horizon_choice = st.radio(
    "Lựa chọn khoảng thời gian dự báo tương lai:",
    options=[1, 24, 72],
    format_func=lambda x: "Dự báo ngắn hạn: 1 giờ tới (t+1h)" if x == 1 else (
        "Dự báo trung hạn: 24 giờ tới / ngày mai (t+24h)" if x == 24 else "Dự báo dài hạn: 72 giờ tới / 3 ngày tới (t+72h)"
    ),
    horizontal=True
)

model, model_type = load_horizon_model(horizon_choice)

if model is not None:
    # Lấy dòng mới nhất có đầy đủ features để suy diễn
    last_features = df.iloc[[-1]][FEATURES]
    pm25_pred = model.predict(last_features)[0]
    
    # Quy đổi dự báo sang AQI
    aqi_pred = pm25_to_aqi(pm25_pred)
    pred_level, pred_color = aqi_to_level(aqi_pred)
    pred_text_color = '#ffffff' if pred_color in ['#ff0000', '#8f3f97', '#7e0023'] else '#000000'
    
    # Tính thời điểm dự kiến xảy ra
    pred_time = (df['date'].iloc[-1] + pd.Timedelta(hours=horizon_choice)).strftime('%d-%m-%Y %H:%M')
    
    # Hiển thị Card dự báo cao cấp
    st.markdown(f"""
    <div style="border: 2px solid {pred_color}; padding: 18px; border-radius: 10px; background-color: #f8f9fa; margin-top: 10px;">
        <h4 style="margin: 0; color: #333;">🔮 KẾT QUẢ DỰ BÁO CỦA MÔ HÌNH {model_type.upper()}</h4>
        <p style="margin: 5px 0; font-size: 1.1rem;">Dự đoán cho thời điểm: <b>{pred_time}</b> (t+{horizon_choice}h)</p>
        <div style="display: flex; gap: 20px; align-items: center; margin-top: 10px;">
            <div style="padding: 10px 20px; border-radius: 6px; background-color: {pred_color}; color: {pred_text_color}; text-align: center; min-width: 140px;">
                <span style="font-size: 0.85rem; display: block; font-weight: bold; text-transform: uppercase;">DỰ BÁO US AQI</span>
                <b style="font-size: 1.8rem;">{aqi_pred}</b>
            </div>
            <div>
                <p style="margin: 0; font-size: 1.15rem;">Mức độ cảnh báo: <b style="color: {pred_color}; text-transform: uppercase;">{pred_level}</b></p>
                <p style="margin: 2px 0 0 0; font-size: 1.15rem;">Nồng độ PM2.5 dự kiến: <b style="color: {pred_color};">{pm25_pred:.1f} µg/m³</b></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning(f"⚠️ Không tìm thấy file mô hình tại models/xgb_t{horizon_choice}.json hoặc sarima_t{horizon_choice}.pkl. Vui lòng chạy huấn luyện trước.")

st.markdown("---")

# ─── 6. BẢN ĐỒ VÀ GIẢI THÍCH (SHAP) ───
c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader("📍 Bản Đồ Phân Bố Nhiệt Bụi Mịn (Hà Nội)")
    st.caption("Mô phỏng bức tranh chỉ số US AQI loang tỏa dưới dạng bản đồ nhiệt quanh khu vực thủ đô.")
    
    # Tạo bản đồ Folium
    heatmap_engine = create_aqi_heatmap(latest_aqi)
    folium_static(heatmap_engine, width=700, height=500)

with c2:
    st.subheader("🕵️‍♀️ Bóc Tách Não Bộ Mô Hình AI (Explainability)")
    st.caption("Ứng dụng lý thuyết trò chơi SHAP để xem các đặc trưng đóng góp như thế nào vào kết quả dự báo.")
    
    shap_tabs = st.tabs(["📊 SHAP Summary (Tổng thể)", "💧 SHAP Waterfall (Episode cực đoan)"])
    
    with shap_tabs[0]:
        shap_sum_path = os.path.join(FIGURES_DIR, "10_shap_summary.png")
        if os.path.exists(shap_sum_path):
            st.image(shap_sum_path, use_column_width=True, caption="Biểu đồ SHAP Summary top 15 features.")
        else:
            st.info("Chưa có biểu đồ SHAP Summary. Hãy chạy huấn luyện và sinh mô hình trước.")
            
    with shap_tabs[1]:
        shap_wat_path = os.path.join(FIGURES_DIR, "11_shap_waterfall.png")
        if os.path.exists(shap_wat_path):
            st.image(shap_wat_path, use_column_width=True, caption="Giải thích dòng dữ liệu ô nhiễm cực đoan nhất trong tập test.")
        else:
            st.info("Chưa có biểu đồ SHAP Waterfall.")

st.markdown("---")

# ─── 7. ĐÁNH GIÁ MÔ HÌNH VÀ IMPORTANCE ───
st.subheader("📊 Đánh Giá Hiệu Năng & Tầm Quan Trọng Của Biến")
st.caption("Rà soát mức độ tin cậy và sự đóng góp của các biến độc lập.")

colA, colB = st.columns(2)

with colA:
    st.markdown("##### 📈 Permutation Importance (Test Set)")
    perm_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
    if os.path.exists(perm_path):
        st.image(perm_path, use_column_width=True, caption="Mức độ giảm hiệu năng mô hình khi xáo trộn từng biến.")
    else:
        st.info("Chưa sinh biểu đồ Permutation Importance.")

with colB:
    st.markdown(f"##### 🎯 Ma Trận Nhầm Lẫn (Confusion Matrix - Horizon t+{horizon_choice}h)")
    cm_path = os.path.join(FIGURES_DIR, f"cm_xgboost_t{horizon_choice}h.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True, caption=f"Confusion Matrix phân loại AQI ở Horizon t+{horizon_choice}h.")
    else:
        st.info(f"Chưa có Confusion Matrix cho Horizon t+{horizon_choice}h. Vui lòng chạy đánh giá.")
