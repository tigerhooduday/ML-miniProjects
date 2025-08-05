import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('models/final_model.pkl')

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Minimal CSS styling
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: :times new roman, serif;
            background-color: #000000ff;
            color: #ffffff;
        }
        .header {
            font-size: 24px;
            font-weight: 600;
            color: #ffffffff;
            margin-bottom: 20px;
        }
        .subheader {
            font-size: 16px;
            color: #e9e9e9ff;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #34444eff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            width: 100%;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #3a7ae4;
        }
        .stNumberInput input, .stSlider {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .result-box {
            background-color: #00b16ff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Project Information")
    st.markdown("""
    This app predicts **house prices** using a trained Machine Learning model.
    
    Adjust the parameters and click **Predict** to see the result.
    
    **Tech Stack**:
    - Streamlit
    - Scikit-learn
    - Joblib
    """)

# Main content
st.markdown('<div class="header">🏠 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter the house details below to get a price estimate</div>', unsafe_allow_html=True)

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("Overall Quality (1-10 scale)", 1, 10, 5)
    GrLivArea = st.number_input("Living Area (sq ft)", min_value=0, value=1500)
    GarageCars = st.slider("Garage Size (car capacity)", 0, 5, 2)
    TotalBathrooms = st.number_input("Total Bathrooms", min_value=0.0, step=0.5, value=2.0)

with col2:
    TotalSF = st.number_input("Total Area (sq ft)", min_value=0, value=2000)
    HouseAge = st.number_input("House Age (years)", min_value=0, value=20)
    GarageAge = st.number_input("Garage Age (years)", min_value=0, value=15)
    YearsSinceRemodel = st.number_input("Years Since Remodel", min_value=0, value=10)

# Prediction button and result
if st.button("Predict Price", key="predict"):
    input_data = np.array([[OverallQual, GrLivArea, GarageCars, TotalBathrooms,
                          TotalSF, HouseAge, GarageAge, YearsSinceRemodel]])
    prediction = model.predict(input_data)[0]
    
    with st.container():
        st.markdown("### Prediction Result")
        st.markdown(f"""
        <div class="result-box">
            <p style="font-size: 20px; margin-bottom: 0;">Estimated House Price:</p>
            <p style="font-size: 32px; font-weight: 600; color: #8efa6dff; margin-top: 0;">${prediction:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)