import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler_heart.pkl')
expected_columns = joblib.load('columns.pkl')




st.title("Heart Disease Prediction")

st.markdown("provide following details to predict heart disease:")

age = st.slider("Age", 18, 120, 25)
sex = st.selectbox("Sex", ['M','F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)

cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])

resting_bst_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])



if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': chest_pain,
        'Resting Blood Pressure': resting_bp,
        'Cholesterol': cholesterol,
        'Fasting Blood Sugar > 120 mg/dl': fasting_bs,
        'Resting ECG': resting_bst_ecg,
        'Max Heart Rate': max_hr,
        'Exercise Induced Angina': exercise_angina,
        'Oldpeak (ST Depression)': oldpeak,
        'ST Slope'+ st_slope  : 1
    }
    
    
    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    
    if prediction == 1:
        st.error(" high risk of Heart Disease.  ")
    else:
        st.success(" low risk of Heart Disease.  ")