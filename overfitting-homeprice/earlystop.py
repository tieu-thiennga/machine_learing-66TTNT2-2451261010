import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

sample_data = data.sample(n=60, random_state=2024)
X = sample_data[['area']]
y = sample_data['price_m2']

test_size = 20
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=2024
)

val_size = 10
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=2024
)

poly = PolynomialFeatures(degree=8, include_bias=False)
scaler = StandardScaler()

X_tr = scaler.fit_transform(poly.fit_transform(X_train))
X_va = scaler.transform(poly.transform(X_val))
X_te = scaler.transform(poly.transform(X_test))

N_tr = len(X_tr)
X_tr_b = np.c_[np.ones(N_tr), X_tr]
X_va_b = np.c_[np.ones(len(X_va)), X_va]
X_te_b = np.c_[np.ones(len(X_te)), X_te]

np.random.seed(42)
w = np.zeros(X_tr_b.shape[1])
learning_rate = 0.05
epochs = 1500

prev_val_err = float('inf')
best_epoch = None

print(
    f"{'Epoch':<10} | {'Train Error (MSE)':<20} | {'Val Error (MSE)':<20} |"
    f" {'Test Error (MSE)':<20}"
)
print('-' * 75)

for ep in range(1, epochs + 1):
  pred_tr = X_tr_b @ w
  grad = (2 / N_tr) * (X_tr_b.T @ (pred_tr - y_train.values))
  w = w - learning_rate * grad

  tr_err = mean_squared_error(y_train, pred_tr)
  va_err = mean_squared_error(y_val, X_va_b @ w)
  te_err = mean_squared_error(y_test, X_te_b @ w)

  if ep % 50 == 0 or ep == 1 or ep in [152, 153, 154]:
    print(f'{ep:<10} | {tr_err:<20.2f} | {va_err:<20.2f} | {te_err:<20.2f}')

  if va_err > prev_val_err:
    best_epoch = ep - 1
    print('-' * 75)
    print(
        f'=> DỪNG SỚM TẠI EPOCH {best_epoch} (Do Val Error bắt đầu tăng ở'
        f' Epoch {ep})'
    )
    break
  else:
    prev_val_err = va_err