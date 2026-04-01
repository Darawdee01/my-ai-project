import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- การตั้งค่าเบื้องต้น ---
st.set_page_config(page_title="AI Model Dashboard", layout="wide")

# ฟังก์ชันโหลดข้อมูลและโมเดล (ใส่ try-except เพื่อกันเว็บพังถ้าไฟล์ไม่ครบ)
def load_assets():
    try:
        with open('model_ml.pkl', 'rb') as f:
            m_ml = pickle.load(f)
        with open('model_nn.pkl', 'rb') as f:
            m_nn = pickle.load(f)
        df = pd.read_csv('titanic_data.csv')
        return m_ml, m_nn, df
    except FileNotFoundError:
        return None, None, None

model_ml, model_nn, df_titanic = load_assets()

# --- ส่วนของ Sidebar (เมนู 4 หน้าหลักตามโจทย์) ---
st.sidebar.title("🔍 เมนูระบบ AI")
page = st.sidebar.radio("เลือกหน้าเว็บ:", [
    "1. รายละเอียดโมเดล ML", 
    "2. รายละเอียดโมเดล Neural Network", 
    "3. ทดสอบโมเดล ML (Ensemble)", 
    "4. ทดสอบโมเดล Neural Network"
])

# ตรวจสอบว่าโหลดไฟล์สำเร็จไหม
if model_ml is None:
    st.error("❌ ไม่พบไฟล์โมเดล (.pkl) หรือ Dataset ใน GitHub กรุณาตรวจสอบชื่อไฟล์")
    st.stop()

# --- หน้าที่ 1: อธิบายโมเดล ML (โจทย์ข้อ 4a) ---
if page == "1. รายละเอียดโมเดล ML":
    st.title("📘 รายละเอียดการพัฒนา Machine Learning")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 ข้อมูลที่ใช้ (Dataset)")
        st.write("- **ชื่อ:** Titanic Survival Dataset")
        st.write("- **ที่มา:** Kaggle (ข้อมูลผู้โดยสารเรือไททานิค)")
        st.write("- **Feature:** Pclass (ชั้นที่นั่ง), Age (อายุ), Fare (ราคาตั๋ว)")
    with col2:
        st.subheader("⚙️ การเตรียมข้อมูล")
        st.write("1. จัดการค่าว่าง (Missing Values) โดยการเติมค่าเฉลี่ยในคอลัมน์ Age")
        st.write("2. เลือกเฉพาะ Feature ที่มีผลต่อการรอดชีวิต")

    st.subheader("🧠 ทฤษฎีอัลกอริทึม (Ensemble Method)")
    st.info("ใช้เทคนิค **Voting Classifier** ซึ่งเป็นการรวมพลังของ 3 โมเดล ได้แก่ Logistic Regression, Random Forest และ SVM เพื่อเพิ่มความแม่นยำในการทำนายผล")

# --- หน้าที่ 2: อธิบายโมเดล Neural Network (โจทย์ข้อ 4a) ---
elif page == "2. รายละเอียดโมเดล Neural Network":
    st.title("📙 รายละเอียดการพัฒนา Neural Network")
    st.markdown("---")
    st.subheader("🧠 โครงสร้างโมเดล (Architecture)")
    st.write("ใช้อัลกอริทึม **Multi-layer Perceptron (MLP)** ซึ่งเป็นโครงสร้างพื้นฐานของ Artificial Neural Network")
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg", width=400)
    
    st.markdown("""
    - **Input Layer:** รับค่า 3 ปัจจัย (ชั้นที่นั่ง, อายุ, ราคาตั๋ว)
    - **Hidden Layers:** มีการตั้งค่า Hidden Layer ขนาด (10, 5) เพื่อประมวลผลความสัมพันธ์ที่ซับซ้อน
    - **Output Layer:** ใช้ Sigmoid Function เพื่อทำนายโอกาสการรอดชีวิต (0 ถึง 1)
    """)

# --- หน้าที่ 3: ทดสอบโมเดล ML (โจทย์ข้อ 4b) ---
elif page == "3. ทดสอบโมเดล ML (Ensemble)":
    st.title("🧪 ทดสอบการทำงาน: Machine Learning")
    st.write("กรอกข้อมูลด้านล่างเพื่อทำนายการรอดชีวิตด้วยโมเดล Ensemble")
    
    pclass = st.selectbox("ชั้นที่นั่ง (1=ดีที่สุด, 3=ประหยัด)", [1, 2, 3])
    age = st.slider("อายุ", 1, 100, 25)
    fare = st.number_input("ราคาตั๋ว (Fare)", value=30.0)
    
    if st.button("ทำนายผลด้วย ML"):
        input_data = np.array([[pclass, age, fare]])
        prediction = model_ml.predict(input_data)[0]
        if prediction == 1:
            st.success("✅ ผลพยากรณ์: มีโอกาสรอดชีวิตสูง")
        else:
            st.error("❌ ผลพยากรณ์: มีโอกาสไม่รอดชีวิต")

# --- หน้าที่ 4: ทดสอบโมเดล Neural Network (โจทย์ข้อ 4b) ---
elif page == "4. ทดสอบโมเดล Neural Network":
    st.title("🔬 ทดสอบการทำงาน: Neural Network")
    st.write("กรอกข้อมูลด้านล่างเพื่อทำนายผลด้วยระบบประสาทเทียม (ANN)")
    
    pclass_nn = st.radio("ชั้นที่นั่ง", [1, 2, 3], horizontal=True)
    age_nn = st.number_input("กรอกอายุ", value=25)
    fare_nn = st.slider("ราคาตั๋ว", 0, 500, 50)
    
    if st.button("ประมวลผลด้วย AI"):
        input_data = np.array([[pclass_nn, age_nn, fare_nn]])
        prediction_nn = model_nn.predict(input_data)[0]
        if prediction_nn == 1:
            st.success("⭐ AI วิเคราะห์ว่า: รอดชีวิต")
        else:
            st.warning("⚠️ AI วิเคราะห์ว่า: ไม่รอดชีวิต")
