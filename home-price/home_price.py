import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data.csv')

df["Diện tích"] = (
    df["Diện tích"]
    .str.replace(" m²", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

df["Số tầng"] = pd.to_numeric(
    df["Số tầng"],
    errors="coerce"
)

df["Số phòng ngủ"] = (
    df["Số phòng ngủ"]
    .str.replace(" phòng", "", regex=False)
)
df["Số phòng ngủ"] = pd.to_numeric(
    df["Số phòng ngủ"],
    errors="coerce"
)
df["Giá/m2"] = (
    df["Giá/m2"]
    .str.replace(" triệu/m²", "", regex=False)
    .str.replace(" đ/m²", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["Giá/m2"] = pd.to_numeric(
    df["Giá/m2"],
    errors="coerce"
)
data = df[
    ["Diện tích", "Số tầng", "Số phòng ngủ", "Giá/m2"]
].dropna()

# xử lý dữ liệu đặt biến tìm hàm 
X = data[
    ["Diện tích", "Số tầng", "Số phòng ngủ"]
]

y = data["Giá/m2"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print("Sai số tuyệt đối trung bình (MAE):", mae)
print("RMSE:", rmse)
print("R2:", r2)


print("b = ", model.intercept_)

print("hệ số :")
for feature, weight in zip(X.columns, model.coef_):
    print(feature, "=", weight)