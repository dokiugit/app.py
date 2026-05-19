import pandas as pd
import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('trained_rf (1).pkl', 'rb'))
scaler = pickle.load(open('scaler (1).pkl', 'rb'))
st.title('Prediksi Hujan Besok')
st.write('Masukkan data cuaca hari ini untuk memprediksi apakah akan hujan besok atau tidak.')

# form untuk input data
temp_max = st.number_input("Suhu Maksimum Hari Ini (°C)", value=32.0)
temp_min = st.number_input("Suhu Minimum Hari Ini (°C)", value=25.0)
precipitation = st.number_input("Curah Hujan Hari Ini (mm)", value=0.0)
wind_speed = st.number_input("Kecepatan Angin Maksimum (km/jam)", value=12.0)
humidity = st.slider("Kecepatan Kelembapan Maksimum Hari Ini (%)", 0, 100, 85)

# Ketika tombol ditekan
if st.button("Prediksi Cuaca Besok"):
    # Susun data input menjadi bentuk dataframe/array
    input_data = np.array([[temp_max, temp_min, precipitation, wind_speed, humidity]])
    
    # Skalasi input data
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"Sedia Payung! Besok diprediksi **HUJAN** dengan probabilitas {prediction_proba*100:.1f}%.")
    else:
        st.success(f"Aman! Besok diprediksi **TIDAK HUJAN** (Cerah/Berawan) dengan probabilitas {(1-prediction_proba)*100:.1f}%.")