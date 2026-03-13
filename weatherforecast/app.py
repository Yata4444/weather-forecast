import streamlit as st
import pandas as pd
import requests
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Прогноз опадів")

st.header("1. Збір історичних даних")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Широта (Харків)", value=50.00, format="%.2f")
    lon = st.number_input("Довгота (Харків)", value=36.23, format="%.2f")
with col2:
    start_date = st.date_input("Початкова дата", datetime.date(2023, 8, 1))
    end_date = st.date_input("Кінцева дата", datetime.date(2024, 2, 1))

if st.button("Отримати історичні дані"):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,windspeed_10m_max,precipitation_sum&timezone=auto"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        if "daily" in data:
            df = pd.DataFrame(data['daily'])
            df.to_csv('weather_daily.csv', index=False)
            st.success("Історичні дані зібрано та збережено")
            st.dataframe(df.head())
        else:
            st.error("Помилка структури даних")
    else:
        st.error("Помилка з'єднання з API")

st.header("2. Навчання моделей")

if st.button("Навчити та обрати найкращу модель"):
    try:
        df = pd.read_csv('weather_daily.csv')
        
        df['target'] = (df['precipitation_sum'] > 0).astype(int)
        
        df = df.drop(columns=['time', 'precipitation_sum'], errors='ignore')
        df = df.dropna()
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "Логістична регресія": LogisticRegression(random_state=42),
            "Випадковий ліс": RandomForestClassifier(random_state=42),
            "Метод найближчих сусідів": KNeighborsClassifier()
        }
        
        best_name = ""
        best_acc = -1
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            st.write(f"Точність '{name}': {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_model = model
                
        st.success(f"Найкраща модель: {best_name} (Точність: {best_acc:.4f})")
        
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(X.columns.tolist(), 'features.pkl')
        st.info("Модель та стандартизатор збережено у файли (.pkl)")
        
    except FileNotFoundError:
        st.error("Спочатку виконайте крок 1")

st.header("3. Прогноз на наступні 3 дні")

if st.button("Зробити прогноз"):
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,windspeed_10m_max&forecast_days=3&timezone=auto"
        res = requests.get(url)
        
        if res.status_code == 200:
            data = res.json()
            forecast_df = pd.DataFrame(data['daily'])
            dates = forecast_df['time'].tolist()
            
            X_forecast = forecast_df[features]
            X_forecast_scaled = scaler.transform(X_forecast)
            
            probabilities = model.predict_proba(X_forecast_scaled)[:, 1]
            
            for i in range(len(dates)):
                prob = probabilities[i]
                if prob < 0.33:
                    level = "Низький"
                elif prob < 0.66:
                    level = "Середній"
                else:
                    level = "Високий"
                    
                st.write(f"Дата: {dates[i]} | Ймовірність опадів: {prob*100:.1f}% | Рівень: {level}")
        else:
            st.error("Помилка завантаження прогнозу")
            
    except FileNotFoundError:
        st.error("Спочатку виконайте крок 2 для створення та збереження моделі")