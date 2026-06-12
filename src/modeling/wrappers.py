"""
Wrappers cho mô hình Time Series (Prophet, SARIMAX) để tương thích chuẩn API của Scikit-Learn.
Cho phép dùng chung trong pipeline `train_evaluate` và `run_cv` vốn dĩ chỉ truyền (X, y).
"""

import pandas as pd
import warnings

# Lưu ý: prophet và statsmodels được import LAZY (bên trong .fit) để việc nạp
# SARIMAXWrapper không kéo theo prophet, và ngược lại. Nhờ vậy benchmark có thể
# dùng SARIMAX kể cả khi môi trường chưa cài prophet.

warnings.filterwarnings('ignore')

class ProphetWrapper:
    """
    Wrapper cho Prophet.
    Prophet yêu cầu DataFrame có cột 'ds' (ngày) và 'y' (target).
    Các biến ngoại sinh (X) phải được khai báo add_regressor().
    """
    def __init__(self, use_exog=True, **kwargs):
        self.use_exog = use_exog
        self.kwargs = kwargs
        self.model = None
        self.regressors = []

    def fit(self, X, y):
        # Thiết lập cột 'ds', mô phỏng date bằng index nếu X không có 'date'
        # Tuy nhiên, pipeline PM25_NMKHDL đã truyền Pandas DataFrame/Series
        df = pd.DataFrame()
        
        # Thử lấy 'date' nếu có trong X
        if 'date' in X.columns:
            df['ds'] = pd.to_datetime(X['date']).dt.tz_localize(None)
        else:
            # Tạo date dummy nếu không có dữ liệu thật (đủ để chạy Prophet)
            start = pd.Timestamp('2020-01-01')
            df['ds'] = pd.date_range(start=start, periods=len(y), freq='D')
            
        df['y'] = y.values
        
        # Dùng các feature liên quan (nếu use_exog) - bỏ time encoded
        if self.use_exog:
            self.regressors = [col for col in X.columns if col != 'date']
            for col in self.regressors:
                df[col] = X[col].values

        from prophet import Prophet
        self.model = Prophet(**self.kwargs)

        if self.use_exog:
            for col in self.regressors:
                self.model.add_regressor(col)
                
        self.model.fit(df)
        return self

    def predict(self, X):
        df = pd.DataFrame()
        if 'date' in X.columns:
            df['ds'] = pd.to_datetime(X['date']).dt.tz_localize(None)
        else:
            # Nếu ko có, dự báo tiếp nối
            last_date = getattr(self.model.history, 'ds').max() if len(self.model.history) > 0 else pd.Timestamp('2020-01-01')
            df['ds'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(X), freq='D')
            
        if self.use_exog:
            for col in self.regressors:
                df[col] = X[col].values
                
        forecast = self.model.predict(df)
        return forecast['yhat'].values


class SARIMAXWrapper:
    """
    Wrapper cho SARIMAX (Statsmodels).
    (endog = y, exog = X).
    Để tránh train quá chậm, mặc định tham số sẽ tinh gọn.
    """
    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), use_exog=True, **kwargs):
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.model_res = None
        
    def fit(self, X, y):
        # Trích xuất dữ liệu mảng
        exog = X.drop(columns=['date']).values if 'date' in X.columns else X.values
        
        if not self.use_exog:
            exog = None
            
        # Thường xuyên có cảnh báo về hội tụ, ignore nó cho sạch log
        import warnings
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.filterwarnings("ignore")

        model = SARIMAX(endog=y.values,
                        exog=exog,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
                        
        self.model_res = model.fit(disp=False)
        return self

    def predict(self, X):
        steps = len(X)
        exog = X.drop(columns=['date']).values if 'date' in X.columns else X.values
        
        if not self.use_exog:
            exog = None
            
        pred = self.model_res.forecast(steps=steps, exog=exog)
        return pred
