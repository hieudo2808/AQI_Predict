"""
Module Folium tạo bản đồ Heatmap thể hiện cường độ ô nhiễm AQI quanh Hà Nội.
Vì bộ dữ liệu Open-Meteo hiện tại chỉ kéo 1 điểm duy nhất của không gian thành phố,
ta sẽ sinh các trạm vệ tinh (fake stations) lân cận tâm Hà Nội để tạo hiệu ứng Heatmap.
"""
import folium
from folium.plugins import HeatMap
import numpy as np

# Tọa độ cơ sở
HANOI_LAT = 21.0285
HANOI_LON = 105.8542

def create_aqi_heatmap(center_aqi):
    """
    Sinh bản đồ Heatmap tại Hà Nội dựa trên mức AQI hiện tại.
    """
    # Bắt đầu map tại Hà Nội
    m = folium.Map(location=[HANOI_LAT, HANOI_LON], zoom_start=12, tiles="cartodbpositron")
    
    # Tính gradient phân bổ độ dốc (AQI càng cao thì tán ô nhiễm càng rộng)
    data = [[HANOI_LAT, HANOI_LON, float(center_aqi)]]

    # Lan tỏa random xung quanh (Radius khoàng 0.05 ~ 5km)
    np.random.seed(42)
    for _ in range(15):
        lat_offset = np.random.uniform(-0.06, 0.06)
        lon_offset = np.random.uniform(-0.06, 0.06)
        # Điểm vệ tinh sẽ lây dao động từ 70% đến 110% của tâm
        noise_aqi = center_aqi * np.random.uniform(0.7, 1.1) 
        data.append([HANOI_LAT + lat_offset, HANOI_LON + lon_offset, float(noise_aqi)])
        
    # Tạo plugin heatmap
    # Radius điều chỉnh độ lớn của mỗi luồng loang khói
    HeatMap(data, radius=25, blur=15, gradient={
        0.2: 'green',
        0.5: 'yellow',
        0.75: 'orange',
        0.9: 'red',
        1.0: 'purple'
    }).add_to(m)

    # Đánh dấu tâm
    folium.Marker(
        location=[HANOI_LAT, HANOI_LON],
        tooltip="Trung tâm đo lường (HN)",
        icon=folium.Icon(color="red" if center_aqi > 150 else ("orange" if center_aqi > 100 else "green"), icon="info-sign")
    ).add_to(m)
    
    return m
