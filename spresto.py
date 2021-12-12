import streamlit as st
import streamlit_analytics
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

streamlit_analytics.start_tracking()

from PIL import Image
image = Image.open('spresto.png')
st.image(image,use_column_width=False)

st.write("""
# Gestational Diabetes Mellitus (GDM) 
This app predicts maternal GDM outcome early during preconception care.
""")

st.sidebar.header('Patient Input Features')

# Collects patient input features into dataframe
def patient_input_features():
    pcv1_hba1c_ifcc = st.sidebar.number_input('HbA1c (mmol/mol)', 0.0,50.0,35.0)
    pcv1_sbp = st.sidebar.number_input('Systolic Blood Pressure (mmHg)', 50.0,250.0,120.0)
    pcv1_dbp = st.sidebar.number_input('Diastolic Blood Pressure (mmHg)', 50.0,250.0,100.0)
    pcv1_map = (1/3)*pcv1_sbp + (2/3)*pcv1_dbp
    fasting_insulin_pcv1 = st.sidebar.number_input('Fasting Insulin (mU/L)', 0.0,50.0,15.0)
    fasting_TG_pcv1 = st.sidebar.number_input('Triglycerides (mmol/L)', 0.0,5.0,1.0)
    fasting_HDL_pcv1 = st.sidebar.number_input('High Density Lipoprotein Cholesterol (mmol/L)', 0.0,5.0,1.0)
    fasting_TGHDL_pcv1 = fasting_TG_pcv1/fasting_HDL_pcv1
    data = {'pcv1_hba1c_ifcc': pcv1_hba1c_ifcc,
            'pcv1_map': pcv1_map,
            'fasting_insulin_pcv1': fasting_insulin_pcv1,
            'fasting_TGHDL_pcv1': fasting_TGHDL_pcv1}
    features = pd.DataFrame(data, index=[0])
    return features
df = patient_input_features()

df = df[:1] # Selects only the first row (the patient input data)

# Displays the patient input features
st.subheader('Patient Input features :')

st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('3.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of GDM'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of GDM'] = 'Yes'
st.write(df1)

#prediction_proba = load_clf.predict_proba(df)
#st.subheader('Prediction Probability in % :')
#st.write(prediction_proba * 100)

streamlit_analytics.stop_tracking()