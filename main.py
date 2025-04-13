import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.impute import SimpleImputer
import os

# Đọc dữ liệu
df = pd.read_csv("data/gym_dataset.csv")
print("5 dòng đầu tiên:\n", df.head())

# Kiểm tra dữ liệu thiếu và thông tin tổng quan
print("\nSố giá trị thiếu:\n", df.isnull().sum())
print("\nThông tin dataset:\n", df.info())

# Chuẩn hóa tên cột
df.columns = [col.strip().lower()
                .replace(" ", "_")
                .replace("(kg)", "__kg")
                .replace("(m)", "__m")
                .replace("(hours)", "__hours")
                .replace("(liters)", "__liters")
                .replace("(days/week)", "__days_per_week") for col in df.columns]

# Chuyển đổi categorical (Gender)
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# One-hot encoding cho các cột categorical khác
df = pd.get_dummies(df, columns=['workout_type', 'experience_level'], drop_first=True)

# Tách features và target
X = df.drop('calories_burned', axis=1)
y = df['calories_burned']

# Chia train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Xử lý NaN (nếu có)
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

# Huấn luyện mô hình Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
joblib.dump(ridge, "models/ridge_model.pkl")

# Huấn luyện mô hình Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/rf_model.pkl")

# Đánh giá mô hình
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    print(f"🔍 {model_name}")
    print("MAE:", mean_absolute_error(y, y_pred))
    print("MSE:", mean_squared_error(y, y_pred))
    print("R²:", r2_score(y, y_pred))
    print("-" * 30)

evaluate_model(ridge, X_test_scaled, y_test, "Ridge Regression")
evaluate_model(rf, X_test, y_test, "Random Forest Regression")
