
import streamlit as st
import numpy as np
import joblib

# ===== MODEL PATHS =====
MODEL_PATH = "heart_disease_rf_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===== TITLE =====
st.title("❤️ Heart Disease Risk Assessment AI Agent")

st.write("Enter patient details below")

# ===== INPUTS =====
age = st.number_input("Age", 1, 100, 50)

sex = st.selectbox("Sex", ["Male", "Female"])


cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])

trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Cholesterol", 100, 600, 200)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

restecg = st.selectbox("Rest ECG", [0, 1, 2])

thalach = st.number_input(
    "Maximum Heart Rate",
    50,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

slope = st.selectbox("Slope", [0, 1, 2])

ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4])

thal = st.selectbox("Thal", [0, 1, 2, 3])

# ===== ENCODING =====
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ===== PREDICTION =====
if st.button("Predict Heart Disease Risk"):

    input_data = np.array([
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    risk = probability[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.write(f"Risk Probability: {risk:.2f}%")

    # ===== AI RECOMMENDATIONS =====

    st.subheader("🤖 AI Health Assistant")

    if prediction[0] == 1:

        st.warning("""
        Possible Risk Factors:
        • High Blood Pressure
        • High Cholesterol
        • Cardiac Stress Indicators

        Recommendations:
        • Consult a Cardiologist
        • Reduce Salt Intake
        • Exercise Regularly
        • Monitor BP and Cholesterol
        """)

    else:

        st.success("""
        Recommendations:
        • Maintain Healthy Diet
        • Continue Physical Activity
        • Attend Regular Checkups
        """)

# ===== CHATBOT =====

st.subheader("💬 Ask the AI Health Agent")

question = st.text_input("Ask a heart health question")

if question:

    q = question.lower()

    if "cholesterol" in q:
        st.write(
            "High cholesterol can increase heart disease risk. Healthy diet and exercise can help reduce it."
        )

    elif "blood pressure" in q:
        st.write(
            "Keeping blood pressure under control helps reduce strain on the heart."
        )

    elif "exercise" in q:
        st.write(
            "Regular exercise strengthens the heart and improves cardiovascular health."
        )

    elif "heart disease" in q:
        st.write(
            "Heart disease includes conditions affecting the heart and blood vessels."
        )

    else:
        st.write(
            "Please consult a healthcare professional for personalized medical advice."
        )
