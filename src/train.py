"""Обучение модели цены аренды. Запуск: python src/train.py

Параметры читаются из params.json, метрики пишутся в metrics.json,
обученная модель сохраняется в models/model.joblib (в Git не хранится).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
params = json.loads((ROOT / "params.json").read_text(encoding="utf-8"))

df = pd.read_csv(ROOT / "data" / "astana_rent_clean.csv")
num_cols = ["rooms", "area", "floor", "floors", "year"]
cat_cols = ["district", "furniture"]
X = df[num_cols + cat_cols]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"])

cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"))
bin_col = params["bin_feature"]
if params["n_bins"] > 0:
    # выбранный числовой признак заменяем на интервалы (кодируем one-hot)
    rest = [c for c in num_cols if c != bin_col]
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    binned = make_pipeline(SimpleImputer(strategy="median"),
                           KBinsDiscretizer(n_bins=params["n_bins"], encode="onehot",
                                            strategy="uniform"))
    prep = make_column_transformer((num, rest), (binned, [bin_col]), (cat, cat_cols))
else:
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    prep = make_column_transformer((num, num_cols), (cat, cat_cols))

regressors = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=params["alpha"]),
    "lasso": Lasso(alpha=params["alpha"], max_iter=50000),
}
model = make_pipeline(prep, regressors[params["model"]])

# Выбор эксперимента — только по кросс-валидации на обучающей выборке
cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "cv_r2": round(float(cv_r2), 4),
    "test_r2": round(float(r2_score(y_test, pred)), 4),
    "test_rmse": round(float(root_mean_squared_error(y_test, pred))),
    "test_mae": round(float(mean_absolute_error(y_test, pred))),
}
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "model.joblib")
(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(params["model"], metrics)
