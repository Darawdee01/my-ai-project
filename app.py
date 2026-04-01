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
    - **Data Cleaning:** Handled missing values in the 'Age' column by imputing the mean value to maintain dataset size.
    - **Feature Selection:** Selected 'Pclass', 'Age', and 'Fare' as primary predictors based on historical correlation with survival rates.
    - **Data Scaling:** Normalized numerical inputs to ensure equal weight during model training.
    """)

    st.header("2. Algorithm Theory: Ensemble Learning")
    st.info("**Voting Classifier (Ensemble):** This model combines multiple individual classifiers to make a final decision.")
    st.write("""
    We used a **Soft Voting** approach integrating three distinct algorithms:
    - **Logistic Regression:** Provides a baseline probability for binary classification.
    - **Random Forest:** A collection of decision trees that prevents overfitting.
    - **SVM (Support Vector Machine):** Optimizes the decision boundary for better separation between 'Survived' and 'Not Survived'.
    """)

    st.header("3. Development Steps")
    st.write("`Step 1` Data Splitting into Training and Testing sets.")
    st.write("`Step 2` Hyperparameter tuning for individual models.")
    st.write("`Step 3` Implementing the Voting Classifier to aggregate results.")

    st.header("4. References")
    st.write("- **Scikit-learn Documentation:** https://scikit-learn.org/")
    st.write("- **Titanic Dataset:** https://www.kaggle.com/c/titanic")
    st.write("- **Data Validation Process:** Collect the obtained data, summarize the findings, and validate them using **Google Gemini**.")

# --- Page 2: Neural Network Detail ---
elif page == "2. Neural Network: Development & Theory":
    st.title("📙 Neural Network Development Report")
    st.markdown("---")
    
    st.header("1. Data Pre-processing")
    st.write("""
    - **Encoding:** Categorical data like 'Pclass' was treated as numerical input.
    - **Standardization:** Input features were scaled to a small range (0 to 1) to help the Neural Network converge faster during Gradient Descent.
    """)

    st.header("2. Algorithm Theory: Artificial Neural Network (ANN)")
    st.write("""
    The model utilizes a **Multi-layer Perceptron (MLP)** architecture, which mimics the human brain's processing:
    - **Input Layer:** 3 Neurons corresponding to our features.
    - **Hidden Layers:** Two layers with 10 and 5 neurons respectively, using the **ReLU** activation function.
    - **Output Layer:** 1 Neuron with a **Sigmoid** activation function.
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg", caption="ANN Architecture", width=500)

    st.header("3. Development Steps")
    st.write("`Step 1` Defining Sequential architecture.")
    st.write("`Step 2` Compiling with Adam Optimizer and Binary Cross-Entropy.")

    st.header("4. References & Data Validation")
    st.write("- **Primary Source:** Deep Learning Theory via TensorFlow/Keras Documentation")
    st.write("- **Data Validation Process:** Collect the obtained data, summarize the findings, and validate them using **Google Gemini**.")
   
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
