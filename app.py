import pandas as pd
import numpy as np
import streamlit as st
import joblib
model = joblib.load('C:/Users/User/PycharmProjects/PythonProject/phone_price_model.pkl')

# Загрузка модели
model = joblib.load('phone_price_model.pkl')

# Все столбцы, используемые для предсказания
all_columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 
               'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 
               'three_g', 'touch_screen', 'wifi']

# Заголовок приложения
st.title('Прогноз ценовой категории на телефоны')

# Ввод данных пользователем
battery_power = st.number_input('Мощность батареи', min_value=0, max_value=2000, value=1000, step=10)
blue = st.selectbox('Bluetooth', [0, 1])
clock_speed = st.number_input('Тактовая частота', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
dual_sim = st.selectbox('Две SIM-карты', [0, 1])
fc = st.number_input('Фронтальная камера (МП)', min_value=0, max_value=20, value=5, step=1)
four_g = st.selectbox('Поддержка 4G', [0, 1])
int_memory = st.number_input('Внутренняя память (ГБ)', min_value=0, max_value=128, value=32, step=1)
m_dep = st.number_input('Глубина (см)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
mobile_wt = st.number_input('Вес телефона (г)', min_value=0, max_value=250, value=150, step=1)
n_cores = st.number_input('Количество ядер процессора', min_value=1, max_value=8, value=4, step=1)
pc = st.number_input('Основная камера (МП)', min_value=0, max_value=20, value=12, step=1)
px_height = st.number_input('Высота экрана (пиксели)', min_value=0, max_value=2000, value=1000, step=10)
px_width = st.number_input('Ширина экрана (пиксели)', min_value=0, max_value=2000, value=1000, step=10)
ram = st.number_input('Оперативная память (МБ)', min_value=0, max_value=4000, value=2000, step=100)
sc_h = st.number_input('Высота экрана (см)', min_value=0, max_value=20, value=10, step=1)
sc_w = st.number_input('Ширина экрана (см)', min_value=0, max_value=20, value=5, step=1)
talk_time = st.number_input('Время разговора (ч)', min_value=0, max_value=24, value=12, step=1)
three_g = st.selectbox('Поддержка 3G', [0, 1])
touch_screen = st.selectbox('Сенсорный экран', [0, 1])
wifi = st.selectbox('Поддержка WiFi', [0, 1])

# Сбор введенных данных в DataFrame
input_data = {
    'battery_power': battery_power,
    'blue': blue,
    'clock_speed': clock_speed,
    'dual_sim': dual_sim,
    'fc': fc,
    'four_g': four_g,
    'int_memory': int_memory,
    'm_dep': m_dep,
    'mobile_wt': mobile_wt,
    'n_cores': n_cores,
    'pc': pc,
    'px_height': px_height,
    'px_width': px_width,
    'ram': ram,
    'sc_h': sc_h,
    'sc_w': sc_w,
    'talk_time': talk_time,
    'three_g': three_g,
    'touch_screen': touch_screen,
    'wifi': wifi
}

input_df = pd.DataFrame([input_data], columns=all_columns).fillna(0)

# Кнопка для предсказания цены
if st.button('Предсказать ценовую категорию'):
    prediction = model.predict(input_df)
    st.success(f'Предполагаемая ценовая категория: {prediction[0]}')
streamlit run app.py
