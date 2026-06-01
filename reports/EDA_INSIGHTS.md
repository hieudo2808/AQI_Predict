# KẾT QUẢ EXPLORATORY DATA ANALYSIS (EDA) VÀ INSIGHT MÙA VỤ PM2.5

## 1. Thống kê mùa vụ (Seasonality Insights)

- Trung bình nồng độ PM2.5 vào **Mùa Đông**: 44.83 µg/m³
- Trung bình nồng độ PM2.5 vào **Mùa Hè**: 38.7 µg/m³
- Nồng độ bụi PM2.5 vào Mùa Đông cao gấp **1.2 lần** so với Mùa Hè.
  Điều này khẳng định luận điểm ô nhiễm không khí tại Hà Nội có tính chu kỳ mùa vụ rất mạnh, phù hợp với thời tiết nghịch nhiệt (temperature inversion) vào mùa đông cản trở khả năng khuếch tán chất ô nhiễm.

## 2. Các biểu đồ đã xuất (Trong `reports/figures/`):

- `01_ts_pm25.png`: Chuỗi thời gian PM2.5 cho thấy xu hướng dài hạn tăng vọt vào các tháng mùa đông.
- `02_decompose_stl.png`: Phân tách STL chỉ ra rõ ràng tính chu kỳ ngày (daily) và xu thế dài hạn.
- `03_box_month.png`: Boxplot cho thấy phân phối PM2.5 cao nhất vào các tháng 11, 12, 1, 2.
- `04_box_hour.png`: Boxplot theo giờ cho thấy đỉnh ô nhiễm xuất hiện vào buổi sáng (7h-9h) và tối (19h-22h) khớp với giờ cao điểm.
- `05_heatmap_hour_day.png`: Heatmap Giờ × Ngày minh họa rõ traffic pattern gây ô nhiễm.
- `06_acf_pacf.png`: ACF/PACF gợi ý tự tương quan rất mạnh ở các lag gần và chu kỳ 24h.
- `07_corr_spearman.png`: Tương quan Spearman cho thấy PM2.5 tương quan rất cao với PM10, CO, NO2 và tương quan nghịch với nhiệt độ, tốc độ gió.
- `08_missingness.png`: Tỷ lệ missing values của các cột dữ liệu theo từng tháng cực kỳ thấp.
