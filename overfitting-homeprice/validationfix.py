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

sample_data = data.sample(n=60, random_state=42)
X = sample_data[['area']]
y = sample_data['price_m2']

test_size = 20
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

val_size = 10
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_size, random_state=42
)

print(
    f"{'Bậc (d)':<10} | {'Train Error (Z)':<18} | {'Validation Error (Y)':<22} |"
    f" {'Test Error':<18}"
)
print('-' * 75)

val_errors = []

for d in range(1, 6):
  model = make_pipeline(
      PolynomialFeatures(degree=d), StandardScaler(), LinearRegression()
  )
  model.fit(X_train, y_train)

  tr_err = mean_squared_error(y_train, model.predict(X_train))
  va_err = mean_squared_error(y_val, model.predict(X_val))
  te_err = mean_squared_error(y_test, model.predict(X_test))

  val_errors.append(va_err)
  print(f'{d:<10} | {tr_err:<18.2f} | {va_err:<22.2f} | {te_err:<18.2f}')

best_degree = np.argmin(val_errors) + 1
print('-' * 75)
print(f'=> BẬC TỐI ƯU ĐƯỢC CHỌN: d = {best_degree}')