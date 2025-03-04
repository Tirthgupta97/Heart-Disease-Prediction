import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    with open('tree_model', 'rb') as f1:
        model = pickle.load(f1)
    with open('scaler_pkl', 'rb') as f2:
        scaler = pickle.load(f2)
    return model, scaler

model, scaler = load_models()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1E88E5;
        font-size: 3rem !important;
        text-align: center;
    }
    .stSubheader {
        color: #424242;
        text-align: center;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title('🫀 Heart Disease Prediction')
st.subheader('Enter patient information for heart disease prediction')
st.markdown("---")

# Create three columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Personal Information")
    age = st.number_input('Age', min_value=1, max_value=120, step=1, help="Patient's age in years")
    gender = st.radio('Gender', ('Male', 'Female'), help="Patient's gender")
    gender = 1 if gender == "Male" else 0

with col2:
    st.markdown("### Clinical Measurements")
    bp = st.number_input('Resting Blood Pressure (mm/Hg)', min_value=90, max_value=200, step=1)
    cholestoral = st.number_input('Cholesterol (mm/dl)', min_value=120, max_value=570, step=1)
    blood_sugar = st.number_input('Fasting Blood Sugar (mg/dl)', min_value=0, max_value=300, step=1)
    blood_sugar = 1 if blood_sugar > 120 else 0
    max_heart_rate = st.number_input('Maximum Heart Rate', min_value=70, max_value=210, step=1)

with col3:
    st.markdown("### Diagnostic Information")
    cp = st.selectbox('Chest Pain Type', 
                     ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'),
                     help="Type of chest pain experienced by patient")
    cp = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}[cp]
    
    exercise_angina = st.radio('Exercise Induced Angina', ('No', 'Yes'))
    exercise_angina = 1 if exercise_angina == "Yes" else 0

# Create two columns for remaining fields
col4, col5 = st.columns(2)

with col4:
    st.markdown("### ECG Results")
    electro_result = st.selectbox('Resting ECG Result', 
                                ('Normal',
                                 'ST-T Wave Abnormality',
                                 'Left Ventricular Hypertrophy'))
    electro_result = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[electro_result]
    
    oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, step=0.1)
    
    slope = st.selectbox('ST Slope', ('Up-Sloping', 'Flat', 'Down-Sloping'))
    slope = {'Up-Sloping': 0, 'Flat': 1, 'Down-Sloping': 2}[slope]

with col5:
    st.markdown("### Additional Tests")
    vessels = st.selectbox('Number of Major Vessels', (0, 1, 2, 3),
                         help="Number of major vessels colored by flourosopy")
    
    thal = st.selectbox('Thalassemia Test Result', 
                       ('Normal', 'Fixed Defect', 'Reversable Defect'))
    thal = {'Normal': 1, 'Fixed Defect': 2, 'Reversable Defect': 3}[thal]

st.markdown("---")

# Center the predict button
col1, col2, col3 = st.columns([1,1,1])
with col2:
    predict_button = st.button('Predict', use_container_width=True)

if predict_button:
    input_data = [[age, gender, cp, bp, cholestoral, blood_sugar, 
                   electro_result, max_heart_rate, exercise_angina, 
                   oldpeak, slope, vessels, thal]]
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.error('📊 Prediction: High Risk of Heart Disease')
        st.markdown("""
            ⚠️ **Please note**: This prediction suggests a higher risk of heart disease. 
            It is recommended to:
            - Consult a healthcare professional
            - Schedule a thorough medical examination
            - Review your lifestyle and diet
            """)
    else:
        st.success('📊 Prediction: Low Risk of Heart Disease')
        st.markdown("""
            ✅ **Good news**: The prediction suggests a lower risk of heart disease. 
            Remember to:
            - Maintain a healthy lifestyle
            - Regular check-ups
            - Continue any prescribed medications
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>This tool is for educational purposes only and should not be used as a substitute for professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)
