import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
val_errors = []

print(
    '{:<10} | {:<18} | {:<18} | {:<18}'.format(
        'Alpha', 'Train Error (Z)', 'Val Error (Y)', 'Test Error'
    )
)
print('-' * 72)

for a in alphas:
  model = make_pipeline(
      PolynomialFeatures(degree=8), StandardScaler(), Ridge(alpha=a)
  )
  model.fit(X_train, y_train)

  tr_err = mean_squared_error(y_train, model.predict(X_train))
  va_err = mean_squared_error(y_val, model.predict(X_val))
  te_err = mean_squared_error(y_test, model.predict(X_test))

  val_errors.append(va_err)
  print(
      '{:<10} | {:<18.2f} | {:<18.2f} | {:<18.2f}'.format(
          a, tr_err, va_err, te_err
      )
  )

best_alpha = alphas[np.argmin(val_errors)]
print('-' * 72)
print('=> HE SO PHAT TOI UU: alpha =', best_alpha)