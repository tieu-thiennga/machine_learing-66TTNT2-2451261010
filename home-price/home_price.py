import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
file_name = 'data.csv'
df = pd.read_csv(file_name)

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

features = [
    "Year",
    "Month",
    "CA-Los Angeles",
    "CA-San Diego",
    "CA-San Francisco",
    "DC-Washington",
    "FL-Miami",
    "IL-Chicago",
]

target = "National-US"

data = df[features + [target]].copy()
data = data.dropna()

x = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"Hệ số xác định R2 Score: {r2:.4f}")
print(f"MAE (Sai số tuyệt đối trung bình): {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

weights = pd.DataFrame({"Đặc trưng": features, "Hệ số (Weight)": model.coef_})
print("\n=== MỨC ĐỘ ẢNH HƯỞNG CỦA CÁC ĐẶC TRƯNG ===")
print(weights)
comparison = pd.DataFrame(
    {"Thực tế": y_test.tail(5).values, "Mô hình dự báo": y_pred[-5:]}
)
print("\n=== SO SÁNH 5 KẾT QUẢ CUỐI CÙNG ===")
print(comparison)