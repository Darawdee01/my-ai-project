import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Model Dashboard", layout="wide")

# Function to load models and data
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

# --- Sidebar Navigation (4 Main Pages) ---
st.sidebar.title("🔍 AI System Menu")
page = st.sidebar.radio("Select Page:", [
    "1. ML Model Details", 
    "2. Neural Network Details", 
    "3. Test ML Model (Ensemble)", 
    "4. Test Neural Network"
])

# Error check if files are missing
if model_ml is None:
    st.error("❌ Model files (.pkl) or Dataset not found in GitHub. Please check file names.")
    st.stop()

# --- Page 1: ML Model Explanation (Requirement 4a) ---
if page == "1. ML Model Details":
    st.title("📘 Machine Learning Development Details")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Dataset Information")
        st.write("- **Name:** Titanic Survival Dataset")
        st.write("- **Source:** Kaggle (Titanic Passenger Data)")
        st.write("- **Features:** Pclass (Ticket Class), Age, Fare")
    with col2:
        st.subheader("⚙️ Data Pre-processing")
        st.write("1. Handled Missing Values: Filled missing Age data with the mean value.")
        st.write("2. Feature Selection: Selected key factors impacting survival probability.")

    st.subheader("🧠 Algorithm Theory (Ensemble Method)")
    st.info("""
    This system uses a **Voting Classifier** technique, which combines the power of 3 models: 
    Logistic Regression, Random Forest, and SVM to improve prediction accuracy.
    """)

# --- Page 2: Neural Network Explanation (Requirement 4a) ---
elif page == "2. Neural Network Details":
    st.title("📙 Neural Network Development Details")
    st.markdown("---")
    st.subheader("🧠 Model Architecture")
    st.write("The algorithm used is a **Multi-layer Perceptron (MLP)**, a fundamental structure of Artificial Neural Networks.")
    
    # Diagram link remains same
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg", width=400)
    
    st.markdown("""
    - **Input Layer:** Accepts 3 inputs (Pclass, Age, Fare).
    - **Hidden Layers:** Configured with hidden layers of size (10, 5) to process complex relationships.
    - **Output Layer:** Uses a Sigmoid/Softmax activation to predict the survival class (0 or 1).
    """)

# --- Page 3: ML Model Testing (Requirement 4b) ---
elif page == "3. Test ML Model (Ensemble)":
    st.title("🧪 Model Testing: Machine Learning")
    st.write("Input data below to predict survival using the Ensemble model.")
    
    pclass = st.selectbox("Passenger Class (1=First, 2=Second, 3=Third)", [1, 2, 3])
    age = st.slider("Age", 1, 100, 25)
    fare = st.number_input("Ticket Fare", value=30.0)
    
    if st.button("Predict with ML"):
        input_data = np.array([[pclass, age, fare]])
        prediction = model_ml.predict(input_data)[0]
        if prediction == 1:
            st.success("✅ Prediction: Likely to Survive")
        else:
            st.error("❌ Prediction: Unlikely to Survive")

# --- Page 4: Neural Network Testing (Requirement 4b) ---
elif page == "4. Test Neural Network":
    st.title("🔬 Model Testing: Neural Network")
    st.write("Input data below to predict results using the Artificial Neural Network (ANN).")
    
    pclass_nn = st.radio("Passenger Class", [1, 2, 3], horizontal=True)
    age_nn = st.number_input("Enter Age", value=25)
    fare_nn = st.slider("Ticket Fare Range", 0, 500, 50)
    
    if st.button("Process with AI"):
        input_data = np.array([[pclass_nn, age_nn, fare_nn]])
        prediction_nn = model_nn.predict(input_data)[0]
        if prediction_nn == 1:
            st.success("⭐ AI Analysis: Survived")
        else:
            st.warning("⚠️ AI Analysis: Not Survived")
