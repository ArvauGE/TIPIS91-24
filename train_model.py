import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('train.csv')

# Разделение данных на признаки и целевую переменную
X = data.drop('price_range', axis=1)
y = data['price_range']

# Разделение на обучающую и тестовую выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Обучение модели
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Предсказание и оценка модели
y_val_pred_rf = rf_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_val_pred_rf))

# Сохранение модели
joblib.dump(rf_model, 'phone_price_model.pkl')
