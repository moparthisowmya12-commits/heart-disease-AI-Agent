# ❤️ Heart Disease Risk Assessment AI Agent

## Project Overview

The Heart Disease Risk Assessment AI Agent is a machine learning-based healthcare application developed to predict the risk of heart disease based on patient health parameters.

The system uses a trained Random Forest Classifier model to analyze patient data and predict whether the patient is at risk of heart disease. Along with prediction, the application also provides health recommendations and an AI-based health assistant for user queries.

---

## Features

* Predicts heart disease risk using Machine Learning
* Uses Random Forest Classifier for classification
* Provides risk probability percentage
* Gives AI-based health recommendations
* Interactive user interface using Streamlit
* Includes AI chatbot for heart health-related questions

---

## Technologies Used

* Python
* Machine Learning
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## Dataset Features

The model uses the following input parameters:

* Age
* Sex
* Chest Pain Type (cp)
* Resting Blood Pressure (trestbps)
* Cholesterol (chol)
* Fasting Blood Sugar (fbs)
* Rest ECG (restecg)
* Maximum Heart Rate (thalach)
* Exercise Induced Angina (exang)
* Oldpeak
* Slope
* Major Vessels (ca)
* Thal

---

## Machine Learning Workflow

1. Load dataset
2. Data preprocessing
3. Train-test split
4. Feature scaling using StandardScaler
5. Train Random Forest Classifier
6. Evaluate model performance
7. Save trained model and scaler
8. Deploy using Streamlit

---

## Installation

Clone the repository:

```bash
git clone <repository-link>
cd heart-disease-ai-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## Project Structure

```bash
heart-disease-ai-agent/
│
├── app.py
├── requirements.txt
├── heart_disease_rf_model.pkl
├── scaler.pkl
├── README.md
```

---

## Model Performance

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Future Enhancements

* Improve model accuracy using advanced ML models
* Add emergency healthcare alerts
* Store patient assessment records
* Integrate real-time hospital recommendations

---

## Conclusion

This project demonstrates how Artificial Intelligence and Machine Learning can help in early detection of heart disease and assist healthcare professionals in decision-making.

Early prediction can help reduce risks and improve patient care.
