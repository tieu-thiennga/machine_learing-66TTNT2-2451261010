import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def clean_num(val):
  if pd.isna(val):
    return np.nan
  val = str(val).replace(',', '.')
  m = re.search(r'[-+]?\d*\.\d+|\d+', val)
  return float(m.group()) if m else np.nan


df = pd.read_csv('data.csv')
df['area'] = df['Diện tích'].apply(clean_num)
df['price_m2'] = df['Giá/m2'].apply(clean_num)

data = df[['area', 'price_m2']].dropna()
data = data[
    (data['area'] >= 25)
    & (data['area'] <= 120)
    & (data['price_m2'] >= 30)
    & (data['price_m2'] <= 220)
]

sample_data = data.sample(n=50, random_state=42)
X = sample_data[['area']]
y = sample_data['price_m2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=30, test_size=20, random_state=42
)

degree = 18
model = make_pipeline(
    PolynomialFeatures(degree=degree), StandardScaler(), LinearRegression()
)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_error = mean_squared_error(y_train, train_pred)
test_error = mean_squared_error(y_test, test_pred)

print('=' * 50)
print(f'ĐÁNH GIÁ MÔ HÌNH HỒI QUY ĐA THỨC BẬC {degree}:')
print(f' - Train Error (MSE): {train_error:.2f}')
print(f' - Test Error (MSE):  {test_error:.2f}')
print('=' * 50)

plt.figure(figsize=(9, 6))

x_min, x_max = X['area'].min() - 2, X['area'].max() + 2
x_plot = pd.DataFrame({'area': np.linspace(x_min, x_max, 500)})
y_plot = model.predict(x_plot)

plt.scatter(
    X_train,
    y_train,
    color='red',
    edgecolors='black',
    s=55,
    label='Training',
    zorder=5,
)
plt.scatter(
    X_test,
    y_test,
    color='gold',
    marker='s',
    edgecolors='black',
    s=55,
    label='Test',
    zorder=5,
)

plt.plot(
    x_plot,
    y_plot,
    color='blue',
    linewidth=2.2,
    label=f'đường hồi quy',
)

# Giới hạn khung hình quan sát
plt.ylim(0, 300)
plt.xlim(20, 125)
plt.title(
    f'overfitting (Degree = {degree}: Overfitting)',
    fontsize=13,
    fontweight='bold',
)
plt.xlabel('Diện tích (m²)', fontsize=11)
plt.ylabel('Giá (triệu/m²)', fontsize=11)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()