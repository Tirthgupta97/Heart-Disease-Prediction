import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle

with open('tree_model', 'rb') as f1:
    model = pickle.load(f1)
with open('scaler_pkl', 'rb') as f2:
    scaler = pickle.load(f2)
   
st.title('Heart Disease Prediction')
st.write('Please provide the following information to predict the heart disease')

# Collect the inputs

age = st.number_input('Age', min_value=1, max_value=120, step=1)

gender = st.radio('Gender', ('Male', 'Female'))
if gender == "Male":
    gender = 1
else:
    gender = 0

cp = st.selectbox('Chest Pain Type', ('Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'))
if cp == "Typical Angina":
    cp = 0
elif cp == "Atypical Angina":
    cp = 1
elif cp == "Non-Anginal Pain":
    cp = 2
else:
    cp = 3

bp = st.number_input('Resting Blood Pressure (mm/Hg)', min_value=90, max_value=200, step=1)

cholestoral = st.number_input('Cholestoral (mm/dl)', min_value=120, max_value=570, step=1)

blood_sugar = st.number_input('Fasting Blood Sugar (mg/dl)', min_value=0, max_value=300, step=1)
if blood_sugar > 120:
    blood_sugar = 1
else:
    blood_sugar = 0

electro_result = st.selectbox('Resting Electrocardiographic Result', (
    'Normal',
    'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 
    'showing probable or definite left ventricular hypertrophy by Estes\' criteria'))
if electro_result == "Normal":
    electro_result = 0
elif electro_result == "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)":
    electro_result = 1
else:
    electro_result = 2

max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=70, max_value=210, step=1)

exercise_angina = st.radio('Exercise Induced Angina', ('No', 'Yes'))
if exercise_angina == "No":
    exercise_angina = 0
else:
    exercise_angina = 1

oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, step=0.1)

slope = st.selectbox('Slope', ('Up-Sloping', 'Flat', 'Down-Sloping'))
if slope == "Up-Sloping":
    slope = 0
elif slope == "Flat":
    slope = 1
else:
    slope = 2

vessels = st.selectbox('No.of Major Vessels', (0, 1, 2, 3))

thal = st.selectbox('Thal Rate', ('Normal', 'Fixed Defect', 'Reversable Defect'))
if thal == "Normal":
    thal = 1
elif thal == "Fixed Defect":
    thal = 2
else:
    thal = 3

input_data = [[age, gender, cp, bp, cholestoral, blood_sugar, electro_result, max_heart_rate, exercise_angina, oldpeak, slope, vessels, thal]]
input_data_scaled = scaler.transform(input_data)


if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    if prediction == 1:
        st.error('Prediction : Patient has Heart Disease :(')
    else:
        st.success('Prediction : Patient has No Heart Disease :) ')
