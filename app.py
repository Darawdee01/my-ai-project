import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np

with open('model_ml.pkl', 'rb') as f:
    model_ml = pickle.load(f)
model_nn = tf.keras.models.load_model('model_nn.h5')

st.sidebar.title("📌 เมนู")
page = st.sidebar.radio("เลือกหน้า", ["🏠 หน้าหลัก", "📘 อธิบายโมเดล", "🧪 ทดสอบทำนายผล"])

if page == "🏠 หน้าหลัก":
    st.title("Project: AI Prediction System")
    st.write("ชุดข้อมูลที่ใช้: Titanic Dataset (รอดหรือไม่รอด)")
    st.dataframe(pd.read_csv('titanic_data.csv'))

elif page == "📘 อธิบายโมเดล":
    st.header("รายละเอียดการพัฒนา")
    st.write("**1. Machine Learning:** ใช้ Ensemble (LR, RF, SVM)")
    st.write("**2. Neural Network:** ใช้โครงสร้าง Simple Dense Layer")

elif page == "🧪 ทดสอบทำนายผล":
    pclass = st.selectbox("ชั้นที่นั่ง", [1, 2, 3])
    age = st.number_input("อายุ", value=25)
    fare = st.number_input("ราคาตั๋ว", value=50.0)
    
    if st.button("ทำนาย"):
        input_data = np.array([[pclass, age, fare]])
        res_ml = model_ml.predict(input_data)[0]
        res_nn = (model_nn.predict(input_data) > 0.5).astype(int)[0][0]
        st.success(f"ผลจาก ML: {'รอด' if res_ml==1 else 'ไม่รอด'}")
        st.success(f"ผลจาก NN: {'รอด' if res_nn==1 else 'ไม่รอด'}")
