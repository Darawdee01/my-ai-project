import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Development Report", layout="wide")

def load_assets():
    try:
        with open('model_ml.pkl', 'rb') as f:
            m_ml = pickle.load(f)
        with open('model_nn.pkl', 'rb') as f:
            m_nn = pickle.load(f)
        df = pd.read_csv('titanic_data.csv')
        return m_ml, m_nn, df
    except:
        return None, None, None

model_ml, model_nn, df_titanic = load_assets()

# --- Sidebar ---
st.sidebar.title("📑 Project Menu")
page = st.sidebar.radio("Navigate to:", [
    "ML Model: Development & Theory", 
    "Neural Network: Development & Theory", 
    "Test: Machine Learning", 
    "Test: Neural Network"
])

if model_ml is None:
    st.error("Error: Required files (models/dataset) are missing in the repository.")
    st.stop()

# --- Page 1: Machine Learning Detail ---
if page == "1. ML Model: Development & Theory":
    st.title("📘 Machine Learning Development Report")
    st.markdown("---")
    
    st.header("1. Data Pre-processing")
    st.write("""
    - **Data Cleaning:** Handled missing values in the 'Age' column by imputing the mean value.
    - **Feature Selection:** Selected 'Pclass', 'Age', and 'Fare' as primary predictors.
    - **Data Scaling:** Normalized numerical inputs for better model performance.
    """)

    st.header("2. Algorithm Theory: Ensemble Learning")
    st.info("**Voting Classifier (Ensemble):** Combines Logistic Regression, Random Forest, and SVM.")

    st.header("3. Development Steps")
    st.write("`Step 1` Data Splitting into Training and Testing sets.")
    st.write("`Step 2` Hyperparameter tuning for individual models.")
    st.write("`Step 3` Implementing the Voting Classifier.")

    st.header("4. References & Data Validation")
    st.write("- **Primary Source:** Titanic Dataset from Kaggle (https://www.kaggle.com/c/titanic)")
    st.write("- **Methodology:** Scikit-learn Documentation (https://scikit-learn.org/)")
    # ปรับเป็นข้อความปกติครับ
    st.write("**Data Validation Process:** Collect the obtained data, summarize the findings, and validate them using **Google Gemini**.")

# --- Page 2: Neural Network Detail ---
elif page == "2. Neural Network: Development & Theory":
    st.title("📙 Neural Network Development Report")
    st.markdown("---")
    
    st.header("1. Data Pre-processing")
    st.write("- **Standardization:** Input features were scaled (0 to 1) for faster convergence.")

    st.header("2. Algorithm Theory: Artificial Neural Network (ANN)")
    st.write("""
    Utilizes **Multi-layer Perceptron (MLP)**:
    - **Input Layer:** 3 Neurons.
    - **Hidden Layers:** 10 and 5 neurons with ReLU activation.
    - **Output Layer:** 1 Neuron with Sigmoid activation.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg", caption="ANN Architecture", width=500)

    st.header("3. Development Steps")
    st.write("`Step 1` Defining Sequential architecture.")
    st.write("`Step 2` Compiling with Adam Optimizer and Binary Cross-Entropy.")

    st.header("4. References & Data Validation")
    st.write("- **Primary Source:** Deep Learning Theory via TensorFlow/Keras Documentation")
    # ปรับเป็นข้อความปกติครับ
    st.write("**Data Validation Process:** Collect the obtained data, summarize the findings, and validate them using **Google Gemini**.")
   
# --- Pages 3 & 4 (Testing Pages) ---
elif page == "3. Test: Machine Learning":
    st.title("🧪 Test: Ensemble ML")
    pclass = st.selectbox("Pclass", [1, 2, 3])
    age = st.slider("Age", 1, 100, 25)
    fare = st.number_input("Fare", value=30.0)
    if st.button("Predict"):
        res = model_ml.predict([[pclass, age, fare]])[0]
        st.success("Result: Survived" if res==1 else "Result: Not Survived")

elif page == "4. Test: Neural Network":
    st.title("🔬 Test: Neural Network")
    pclass_nn = st.radio("Pclass", [1, 2, 3], horizontal=True)
    age_nn = st.number_input("Age", value=25)
    fare_nn = st.slider("Fare", 0, 500, 50)
    if st.button("Run AI Analysis"):
        res_nn = model_nn.predict([[pclass_nn, age_nn, fare_nn]])[0]
        st.info("AI Prediction: Survived" if res_nn==1 else "AI Prediction: Not Survived")

