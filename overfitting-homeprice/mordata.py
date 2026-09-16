import re
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

# Chia tập Test cố định (20% dữ liệu) để đánh giá khách quan
X_train_full, X_test, y_train_full, y_test = train_test_split(
    data[['area']], data['price_m2'], test_size=0.2, random_state=42
)

# Thử nghiệm tăng dần số lượng mẫu huấn luyện (N) với cùng mô hình bậc 8
sample_sizes = [30, 60, 120, 250, len(X_train_full)]
degree = 8

print(
    '{:<18} | {:<18} | {:<18}'.format(
        'So luong mau (N)', 'Train MSE', 'Test MSE'
    )
)
print('-' * 60)

for n in sample_sizes:
  X_sub = X_train_full.iloc[:n]
  y_sub = y_train_full.iloc[:n]

  model = make_pipeline(
      PolynomialFeatures(degree=degree), StandardScaler(), LinearRegression()
  )
  model.fit(X_sub, y_sub)

  tr_err = mean_squared_error(y_sub, model.predict(X_sub))
  te_err = mean_squared_error(y_test, model.predict(X_test))

  print('{:<18} | {:<18.2f} | {:<18.2f}'.format(n, tr_err, te_err))