import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier # ใช้ตัวนี้แทน TensorFlow

# --- โหลดโมเดล (ใช้ไฟล์ .pkl ทั้งคู่เพื่อความเบา) ---
try:
    with open('model_ml.pkl', 'rb') as f:
        model_ml = pickle.load(f)
    with open('model_nn.pkl', 'rb') as f:
        model_nn = pickle.load(f)
    df_sample = pd.read_csv('titanic_data.csv')
except:
    st.error("ไม่พบไฟล์โมเดลหรือ Dataset ใน GitHub")

# --- เมนู ---
st.sidebar.title("📌 เมนู")
page = st.sidebar.radio("เลือกหน้า", ["🏠 หน้าหลัก", "📘 อธิบายโมเดล", "🧪 ทดสอบทำนายผล"])

if page == "🏠 หน้าหลัก":
    st.title("Project: AI Prediction System")
    st.write("ข้อมูลผู้โดยสาร Titanic")
    st.dataframe(df_sample.head())

elif page == "📘 อธิบายโมเดล":
    st.header("รายละเอียดโมเดล")
    st.write("1. **Machine Learning:** ใช้ Ensemble (LR, RF, SVM)")
    st.write("2. **Neural Network:** ใช้ Multi-layer Perceptron (MLP)")

elif page == "🧪 ทดสอบทำนายผล":
    st.subheader("ลองใส่ข้อมูลเพื่อพยากรณ์")
    pclass = st.selectbox("ชั้นที่นั่ง", [1, 2, 3])
    age = st.number_input("อายุ", value=25)
    fare = st.number_input("ราคาตั๋ว", value=50.0)
    
    if st.button("ทำนายผล"):
        input_data = np.array([[pclass, age, fare]])
        res_ml = model_ml.predict(input_data)[0]
        res_nn = model_nn.predict(input_data)[0]
        
        col1, col2 = st.columns(2)
        col1.metric("ผลจาก ML Ensemble", "รอด" if res_ml==1 else "ไม่รอด")
        col2.metric("ผลจาก Neural Network", "รอด" if res_nn==1 else "ไม่รอด")
