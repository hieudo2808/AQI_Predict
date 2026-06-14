"""
App Streamlit cung cấp Giao diện Điều khiển (Dashboard) chất lượng không khí PM2.5 (US AQI).
Tích hợp Bản đồ Folium, Hệ thống Cảnh báo, và Explainability.
Command chạy: streamlit run src/ui/app_streamlit.py
"""
import os
# Fix đường dẫn tuyệt đối cho import nếu chạy ngoài root dir
import sys

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ui.map_heatmap import create_aqi_heatmap
from src.config import FIGURES_DIR, WEATHER_FEATURES
from src.aqi_formula import pm25_to_aqi, aqi_to_level
from src.modeling.champion import (
    DEFAULT_MANIFEST_PATH,
    apply_future_weather_if_allowed,
    build_feature_frame,
    load_model_for_horizon,
)
from src.pipelines.prediction_pipeline import check_data_freshness, StaleDataError
from src.data.ingest import fetch_weather_forecast

# Config giao diện Streamlit
st.set_page_config(page_title="Hanoi AQI Live & Forecast Dashboard", layout="wide")

# ─── 1. HÀM LOAD DỮ LIỆU VÀ MÔ HÌNH ───
@st.cache_data
def load_data(file_mtime):
    """Nạp dữ liệu từ Parquet features/targets sạch tần suất Hourly."""
    path = "data/features/features_targets.parquet"
    if not os.path.exists(path):
        # Fallback nếu không có parquet thì đọc dataset processed
        path = "data/processed/aq_hourly_clean.parquet"
        if not os.path.exists(path):
            st.error(f"Không tìm thấy dữ liệu tại data/features/ hoặc data/processed/. Vui lòng chạy pipeline trước.")
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

# Lấy mốc thời gian sửa đổi file để tự động clear cache khi file thay đổi
path_to_watch = "data/features/features_targets.parquet"
if not os.path.exists(path_to_watch):
    path_to_watch = "data/processed/aq_hourly_clean.parquet"
file_mtime = os.path.getmtime(path_to_watch) if os.path.exists(path_to_watch) else 0

@st.cache_resource
def load_horizon_model(horizon):
    """Nạp champion model; fallback sang legacy XGBoost/SARIMA nếu chưa export champion."""
    return load_model_for_horizon(horizon)

df = load_data(file_mtime)

# ─── 2. HEADER DỰ ÁN ───
st.title("Hệ Thống Live-Monitor & Dự Báo Ô Nhiễm PM2.5 Hà Nội")
st.markdown("Giám sát chất lượng không khí thời gian thực và cung cấp cảnh báo AI đa khung thời gian (Multi-Horizon).")

try:
    check_data_freshness(max_age_hours=6)
except StaleDataError as e:
    st.error(f"**CẢNH BÁO DỮ LIỆU CŨ:** {e}")

if df.empty:
    st.warning("Dữ liệu đang được cập nhật, vui lòng chạy pipeline chính để khởi tạo.")
    st.stop()

# --- TÁCH LUỒNG 1: DỮ LIỆU THỰC TẾ (SENSOR DATA) ---
if 'pm2_5_missing_flag' in df.columns:
    actual_data_df = df[df['pm2_5_missing_flag'] == 0]
else:
    actual_data_df = df

if not actual_data_df.empty:
    latest_actual_row = actual_data_df.iloc[-1]
else:
    latest_actual_row = df.iloc[-1]

latest_date = latest_actual_row['date']
latest_date_str = latest_date.strftime('%d-%m-%Y %H:%M')
latest_pm25 = latest_actual_row['pm2_5']

# Tính toán độ trễ (Lag)
current_time = pd.Timestamp.now(tz='Asia/Ho_Chi_Minh')
lag_hours = max(0, int((current_time - latest_date).total_seconds() / 3600))

lag_warning = ""
if lag_hours > 0:
    lag_warning = f"<span style='color: inherit; font-weight: bold;'>⚠️ Trạm đo PurpleAir đang bị trễ {lag_hours} giờ</span>"

# Quy đổi AQI thời gian thực
latest_aqi = pm25_to_aqi(latest_pm25)
latest_level, latest_color = aqi_to_level(latest_aqi)

# Thiết lập chữ trắng cho nền tối, chữ đen cho nền sáng
text_color = '#ffffff' if latest_color in ['#ff0000', '#8f3f97', '#7e0023'] else '#000000'

# --- TÁCH LUỒNG 2: DỰ BÁO TỨC THỜI (NOWCAST) ---
nowcast_bundle = load_horizon_model(1)
nowcast_model = nowcast_bundle.model
nowcast_type = nowcast_bundle.label
nowcast_pm25 = None
nowcast_time_str = "Không xác định"

if nowcast_model is not None and len(df) >= 2:
    current_hour_tz = pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').replace(minute=0, second=0, microsecond=0)
    prev_hour_tz = current_hour_tz - pd.Timedelta(hours=1)
    
    matching_rows = df[df['date'] == prev_hour_tz]
    if not matching_rows.empty:
        nowcast_base_row = matching_rows.iloc[[-1]].copy()
    else:
        nowcast_base_row = df.iloc[[-1]].copy()
        current_hour_tz = df.iloc[-1]['date'] + pd.Timedelta(hours=1)

    nowcast_features = build_feature_frame(df, nowcast_bundle.feature_columns, base_row=nowcast_base_row)
    if nowcast_bundle.uses_future_weather_forecast:
        current_hour_naive = current_hour_tz.tz_localize(None)
        try:
            forecast_df = fetch_weather_forecast()
            current_weather = forecast_df[forecast_df['datetime'] == current_hour_naive]
            if not current_weather.empty:
                nowcast_features = apply_future_weather_if_allowed(
                    nowcast_features,
                    current_weather.iloc[0],
                    nowcast_bundle.uses_future_weather_forecast,
                )
        except Exception:
            pass
    
    nowcast_pm25 = nowcast_model.predict(nowcast_features)[0]
    nowcast_aqi = pm25_to_aqi(nowcast_pm25)
    nowcast_level, nowcast_color = aqi_to_level(nowcast_aqi)
    nowcast_text_color = '#ffffff' if nowcast_color in ['#ff0000', '#8f3f97', '#7e0023'] else '#000000'
    nowcast_time_str = current_hour_tz.strftime('%d-%m-%Y %H:%M')

# --- HIỂN THỊ CỘT ĐÔI ---
col_actual, col_nowcast = st.columns(2)

with col_actual:
    st.markdown(f"""
    <div style="background-color: {latest_color}; padding: 22px; border-radius: 12px; text-align: center; color: {text_color}; margin-bottom: 20px; min-height: 260px;">
        <p style="margin: 0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">Chỉ số US AQI Thực tế (Trạm đo)</p>
        <h1 style="margin: 5px 0; font-size: 3.8rem; color: {text_color};">{latest_aqi}</h1>
        <h2 style="margin: 0; font-weight: bold; color: {text_color};">{latest_level}</h2>
        <p style="margin: 8px 0 0 0; font-size: 1.15rem;">PM2.5: <b>{latest_pm25:.1f} µg/m³</b> ({latest_date_str})</p>
        <p style="margin: 5px 0 0 0; font-size: 1rem;">{lag_warning}</p>
    </div>
    """, unsafe_allow_html=True)

with col_nowcast:
    if nowcast_pm25 is not None:
        st.markdown(f"""
        <div style="background-color: {nowcast_color}; padding: 22px; border-radius: 12px; text-align: center; color: {nowcast_text_color}; margin-bottom: 20px; min-height: 260px;">
            <p style="margin: 0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">AI Nowcast (Dự Báo Tức Thời)</p>
            <h1 style="margin: 5px 0; font-size: 3.8rem; color: {nowcast_text_color};">{nowcast_aqi}</h1>
            <h2 style="margin: 0; font-weight: bold; color: {nowcast_text_color};">{nowcast_level}</h2>
            <p style="margin: 8px 0 0 0; font-size: 1.15rem;">PM2.5: <b>{nowcast_pm25:.1f} µg/m³</b> ({nowcast_time_str})</p>
            <p style="margin: 5px 0 0 0; font-size: 1rem; color: inherit;">Ước lượng bằng {nowcast_type}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Không thể hiển thị Nowcast do thiếu model hoặc dữ liệu.")

# ─── 2.5 KIỂM ĐỊNH MÔ HÌNH TRỰC QUAN (REAL-TIME VALIDATION) ───
if nowcast_model is not None and 'pm2_5_missing_flag' in df.columns:
    st.markdown("### Kiểm Định Độ Chính Xác (Nowcast Validation)")
    st.caption("Đối chiếu kết quả dự báo của AI Nowcast với số liệu thực tế đo được tại trạm trong 72 giờ qua.")
    
    # Lấy 72 dòng dữ liệu thực tế gần nhất (bỏ qua những dòng ffill)
    valid_df = df[df['pm2_5_missing_flag'] == 0].tail(73).copy() # Cần 73 để shift 1
    
    if len(valid_df) > 1:
        # Tính toán input features t-1
        if all(col in valid_df.columns for col in nowcast_bundle.feature_columns):
            eval_features = valid_df[nowcast_bundle.feature_columns].shift(1).copy()

            if nowcast_bundle.uses_future_weather_forecast:
                for col in WEATHER_FEATURES:
                    if col in eval_features.columns:
                        eval_features[col] = valid_df[col]

            valid_df = valid_df.iloc[1:].copy()
            eval_features = eval_features.iloc[1:].copy()

            preds = nowcast_model.predict(eval_features)

            chart_index = valid_df['date'].dt.tz_localize(None)
            chart_df = pd.DataFrame({
                'Actual PM2.5 (Thực tế)': valid_df['pm2_5'].values,
                'Predicted Nowcast (AI)': preds
            }, index=chart_index)

            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(chart_df['Actual PM2.5 (Thực tế)'], chart_df['Predicted Nowcast (AI)'])

            st.markdown(f"**Sai số trung bình (MAE) trong {len(chart_df)} giờ qua:** <b style='color: #d35400;'>{mae:.2f} µg/m³</b>", unsafe_allow_html=True)
            st.line_chart(chart_df)
        else:
            st.info("Nowcast validation chỉ hiển thị cho champion dùng feature table trực tiếp.")
        
st.markdown("---")

# ─── 3. KHỐI KHUYẾN NGHỊ SỨC KHỎE CHI TIẾT ───
advice_dict = {
    'Good': "Chất lượng không khí tuyệt vời và an toàn cho mọi người. Bạn nên tham gia các hoạt động thể dục thể thao ngoài trời.",
    'Moderate': "Chất lượng không khí ở mức chấp nhận được. Tuy nhiên, người cực kỳ nhạy cảm với ô nhiễm (hen suyễn, hô hấp yếu) nên hạn chế hoạt động ngoài trời kéo dài nếu cảm thấy mệt mỏi.",
    'Unhealthy for Sensitive': "Nhóm nhạy cảm (trẻ em, người già, phụ nữ mang thai và người có bệnh tim mạch/hô hấp) nên hạn chế hoạt động mạnh ngoài trời kéo dài. Nên đeo khẩu trang lọc bụi khi đi ra đường.",
    'Unhealthy': "Mức độ ô nhiễm có hại cho sức khỏe cộng đồng. Tất cả mọi người nên hạn chế hoạt động thể lực ngoài trời quá lâu. Bắt buộc đeo khẩu trang N95 khi di chuyển bên ngoài.",
    'Very Unhealthy': "Cảnh báo nghiêm trọng về sức khỏe. Tất cả mọi người nên tránh hoạt động ngoài trời hoàn toàn. Đóng các cửa sổ trong phòng và sử dụng máy lọc không khí trong nhà.",
    'Hazardous': "Tình trạng khẩn cấp toàn diện. Tránh ra ngoài tuyệt đối. Mọi người nên ở trong nhà kín, bật máy lọc không khí và chuẩn bị các biện pháp y tế hỗ trợ nếu cần thiết."
}
st.info(advice_dict.get(latest_level, "Không có khuyến nghị cụ thể cho mức này."))

# ─── 4. METRICS BOARD ───
st.markdown("### Chỉ Số Đo Lường Nhanh")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Thời Gian Cập Nhật", latest_date_str)

# Tính chênh lệch so với 24 giờ trước
if len(df) > 24:
    prev_pm25 = df.iloc[-25]['pm2_5']
    prev_aqi = pm25_to_aqi(prev_pm25)
    delta_aqi = latest_aqi - prev_aqi
    col2.metric("Chỉ số US AQI", f"{latest_aqi}", delta=f"{delta_aqi:+.0f} so với 24h trước", delta_color="inverse")
else:
    col2.metric("Chỉ số US AQI", f"{latest_aqi}")

col3.metric("Bụi mịn PM2.5", f"{latest_pm25:.1f} µg/m³")

if os.path.exists(DEFAULT_MANIFEST_PATH):
    model_status = "Champion registry"
elif os.path.exists("models/xgb_t24.json"):
    model_status = "Legacy XGBoost"
else:
    model_status = "Chưa huấn luyện"
col4.metric("Hệ Thống Trí Tuệ Nhân Tạo (AI)", model_status)

st.markdown("---")

# ─── 5. MULTI-HORIZON FORECAST ENGINE (AI DỰ BÁO) ───
st.subheader("Trạm Dự Báo Tương Lai (AI Forecast Engine)")
st.caption("Chọn horizon để nạp champion model; nếu chưa export champion thì tự fallback sang mô hình legacy.")

horizon_choice = st.radio(
    "Lựa chọn khoảng thời gian dự báo (tính từ hiện tại):",
    options=[1, 24, 48, 72],
    format_func=lambda x: "Tiếp theo: 1 giờ tới (t+1h)" if x == 1 else (
        "Ngày mai: 24 giờ tới (t+24h)" if x == 24 else (
            "2 Ngày tới: 48 giờ tới (t+48h)" if x == 48 else "3 Ngày tới: 72 giờ tới (t+72h)"
        )
    ),
    horizontal=True
)

model_bundle = load_horizon_model(horizon_choice)
model = model_bundle.model
model_type = model_bundle.label

if model is not None:
    last_features = build_feature_frame(df, model_bundle.feature_columns)
    
    # Tính thời điểm dự kiến xảy ra
    target_time = df['date'].iloc[-1] + pd.Timedelta(hours=horizon_choice)
    target_time_naive = target_time.tz_localize(None)
    
    if model_bundle.uses_future_weather_forecast:
        try:
            forecast_df = fetch_weather_forecast()
            future_weather = forecast_df[forecast_df['datetime'] == target_time_naive]

            if not future_weather.empty:
                last_features = apply_future_weather_if_allowed(
                    last_features,
                    future_weather.iloc[0],
                    model_bundle.uses_future_weather_forecast,
                )
            else:
                st.warning(f"Không tìm thấy dự báo thời tiết cho thời điểm {target_time.strftime('%d-%m-%Y %H:%M')}. Dùng giá trị hiện tại làm fallback.")
        except Exception as e:
            st.warning(f"Lỗi khi lấy dự báo thời tiết Open-Meteo ({e}). Dùng giá trị hiện tại làm fallback.")

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
        <h4 style="margin: 0; color: #333;">KẾT QUẢ DỰ BÁO TƯƠNG LAI ({model_type.upper()})</h4>
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
    st.warning(f"Không tìm thấy champion hoặc legacy model cho horizon t+{horizon_choice}h. Vui lòng chạy benchmark/export champion trước.")

st.markdown("---")

# ─── 6. BẢN ĐỒ VÀ GIẢI THÍCH (SHAP) ───
c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader("Bản Đồ Phân Bố Nhiệt Bụi Mịn (Hà Nội)")
    st.caption("Mô phỏng bức tranh chỉ số US AQI loang tỏa dưới dạng bản đồ nhiệt quanh khu vực thủ đô.")
    
    # Tạo bản đồ Folium
    heatmap_engine = create_aqi_heatmap(latest_aqi)
    st_folium(heatmap_engine, width=700, height=500)

with c2:
    st.subheader("Bóc Tách Khả Năng Diễn Giải Mô Hình (Explainability)")
    st.caption("Ứng dụng lý thuyết trò chơi SHAP để xem các đặc trưng đóng góp như thế nào vào kết quả dự báo.")
    
    shap_tabs = st.tabs(["SHAP Summary (Tổng thể)", "SHAP Waterfall (Episode cực đoan)"])
    
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
st.subheader("Đánh Giá Hiệu Năng & Tầm Quan Trọng Của Biến")
st.caption("Rà soát mức độ tin cậy và sự đóng góp của các biến độc lập.")

colA, colB = st.columns(2)

with colA:
    st.markdown("##### Permutation Importance (Test Set)")
    perm_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
    if os.path.exists(perm_path):
        st.image(perm_path, use_column_width=True, caption="Mức độ giảm hiệu năng mô hình khi xáo trộn từng biến.")
    else:
        st.info("Chưa sinh biểu đồ Permutation Importance.")

with colB:
    st.markdown(f"##### Ma Trận Nhầm Lẫn Legacy XGBoost (Horizon t+{horizon_choice}h)")
    cm_path = os.path.join(FIGURES_DIR, f"cm_xgboost_t{horizon_choice}h.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True, caption=f"Confusion Matrix legacy XGBoost ở Horizon t+{horizon_choice}h.")
    else:
        st.info(f"Chưa có Confusion Matrix cho Horizon t+{horizon_choice}h. Vui lòng chạy đánh giá.")
