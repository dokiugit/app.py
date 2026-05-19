import pandas as pd
import numpy as np
import streamlit as st
import pickle
import requests
from datetime import datetime

# Set page config with meta tag for email verification
st.set_page_config(
    page_title="Prediksi Hujan Besok",
    page_icon="🌧️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom HTML with meta tag
st.markdown("""
<meta name="dicoding:email" content="dhokhiymusofa@gmail.com">
""", unsafe_allow_html=True)

# Load model and scaler
try:
    model = pickle.load(open('trained_rf.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model files not found! Please train the model first by running: python train_model.py")
    st.stop()

# Title and description
st.title('🌧️ Prediksi Hujan Besok')
st.write('Masukkan data cuaca hari ini untuk memprediksi apakah akan hujan besok atau tidak.')

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.write("""
    **Model:** Random Forest Classifier
    
    **Fitur yang digunakan:**
    - Suhu maksimum
    - Suhu minimum
    - Curah hujan
    - Kecepatan angin
    - Kelembapan relatif
    
    **Akurasi:** 83%
    """)
    
    st.markdown("---")
    st.write("**Lokasi:** Jakarta, Indonesia")
    st.write(f"**Tanggal Prediksi:** {datetime.now().strftime('%d %B %Y')}")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    temp_max = st.number_input(
        "🌡️ Suhu Maksimum Hari Ini (°C)", 
        value=32.0,
        min_value=0.0,
        max_value=50.0,
        step=0.1
    )
    precipitation = st.number_input(
        "💧 Curah Hujan Hari Ini (mm)", 
        value=0.0,
        min_value=0.0,
        max_value=200.0,
        step=0.1
    )
    humidity = st.slider(
        "💨 Kelembapan Maksimum Hari Ini (%)", 
        0, 100, 85
    )

with col2:
    temp_min = st.number_input(
        "🌡️ Suhu Minimum Hari Ini (°C)", 
        value=25.0,
        min_value=0.0,
        max_value=50.0,
        step=0.1
    )
    wind_speed = st.number_input(
        "💨 Kecepatan Angin Maksimum (km/jam)", 
        value=12.0,
        min_value=0.0,
        max_value=100.0,
        step=0.1
    )

# Separator
st.markdown("---")

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔮 Prediksi Cuaca Besok", use_container_width=True)

if predict_button:
    try:
        # Arrange input data
        input_data = np.array([[temp_max, temp_min, precipitation, wind_speed, humidity]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        
        # Display results
        if prediction[0] == 1:
            st.error(f"☔ **HUJAN BESOK!**")
            st.markdown(f"""
            ### Rekomendasi: Sedia Payung!
            
            **Probabilitas Hujan:** {prediction_proba[1]*100:.1f}%
            
            **Probabilitas Tidak Hujan:** {prediction_proba[0]*100:.1f}%
            
            ---
            **Tips:**
            - Bawa payung saat keluar
            - Gunakan jaket tahan air
            - Hati-hati di jalan yang licin
            """)
        else:
            st.success(f"☀️ **TIDAK HUJAN BESOK!**")
            st.markdown(f"""
            ### Rekomendasi: Aman untuk Aktivitas Outdoor
            
            **Probabilitas Tidak Hujan:** {prediction_proba[0]*100:.1f}%
            
            **Probabilitas Hujan:** {prediction_proba[1]*100:.1f}%
            
            ---
            **Tips:**
            - Cuaca cerah/berawan
            - Cocok untuk aktivitas outdoor
            - Jangan lupa tabir surya
            """)
        
        # Show input summary
        with st.expander("📊 Detail Input Data"):
            summary_data = {
                "Parameter": ["Suhu Maksimum", "Suhu Minimum", "Curah Hujan", "Kecepatan Angin", "Kelembapan"],
                "Nilai": [f"{temp_max}°C", f"{temp_min}°C", f"{precipitation}mm", f"{wind_speed}km/jam", f"{humidity}%"]
            }
            st.table(summary_data)
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam prediksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><small>🎓 Aplikasi Prediksi Hujan menggunakan Machine Learning | Random Forest Classifier</small></p>
    <p><small>📍 Data: Jakarta, Indonesia | 📈 Akurasi: 83%</small></p>
</div>
""", unsafe_allow_html=True)
