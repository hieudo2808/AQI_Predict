"""
Hàm chuyển đổi PM2.5 ↔ US_AQI theo chuẩn EPA/AirNow.
"""

AQI_LEVELS = [
    (50, 'Good', '#00E400'),
    (100, 'Moderate', '#FFFF00'),
    (150, 'Unhealthy for Sensitive', '#FF7E00'),
    (200, 'Unhealthy', '#FF0000'),
    (300, 'Very Unhealthy', '#8F3F97'),
    (500, 'Hazardous', '#7E0023'),
]


def pm25_to_aqi(c: float) -> int:
    """
    Chuyển đổi nồng độ PM2.5 (µg/m³) sang chỉ số US_AQI.
    
    Theo công thức của Cơ quan Bảo vệ Môi trường Hoa Kỳ (US EPA).
    """
    bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    c_r = round(c, 1)
    if c_r < 0:
        return 0
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= c_r <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (c_r - c_lo) + i_lo)
    return 500


def aqi_to_level(aqi: int) -> tuple[str, str]:
    """
    Trả về (nhãn, mã màu) dựa trên chỉ số US_AQI.
    """
    if aqi < 0:
        return 'Good', '#00E400'
    for threshold, label, color in AQI_LEVELS:
        if aqi <= threshold:
            return label, color
    return 'Hazardous', '#7E0023'
