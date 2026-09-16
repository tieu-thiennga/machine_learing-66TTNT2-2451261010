import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
degrees = list(range(1, 9))
cv_val_scores = []
train_scores = []
test_scores = []

print(
    '{:<10} | {:<18} | {:<22} | {:<18}'.format(
        'Bac (d)', 'Train MSE (TB)', 'CV Val MSE (5-Fold)', 'Test MSE'
    )
)
print('-' * 75)

for d in degrees:
  fold_tr_errs = []
  fold_va_errs = []

  for tr_idx, va_idx in kf.split(X_train_val):
    X_tr, X_va = X_train_val.iloc[tr_idx], X_train_val.iloc[va_idx]
    y_tr, y_va = y_train_val.iloc[tr_idx], y_train_val.iloc[va_idx]

    model = make_pipeline(
        PolynomialFeatures(degree=d), StandardScaler(), LinearRegression()
    )
    model.fit(X_tr, y_tr)

    fold_tr_errs.append(mean_squared_error(y_tr, model.predict(X_tr)))
    fold_va_errs.append(mean_squared_error(y_va, model.predict(X_va)))

  mean_tr_err = np.mean(fold_tr_errs)
  mean_va_err = np.mean(fold_va_errs)
  train_scores.append(mean_tr_err)
  cv_val_scores.append(mean_va_err)

  full_model = make_pipeline(
      PolynomialFeatures(degree=d), StandardScaler(), LinearRegression()
  )
  full_model.fit(X_train_val, y_train_val)
  te_err = mean_squared_error(y_test, full_model.predict(X_test))
  test_scores.append(te_err)

  print(
      '{:<10} | {:<18.2f} | {:<22.2f} | {:<18.2f}'.format(
          d, mean_tr_err, mean_va_err, te_err
      )
  )

best_degree = degrees[np.argmin(cv_val_scores)]
print('-' * 75)
print('=> BAC TOI UU (CV 5-FOLD): d =', best_degree)