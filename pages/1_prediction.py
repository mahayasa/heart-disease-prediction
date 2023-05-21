import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier
#import os


#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pages/crd.json'

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🩺",
)

st.title('Heart Disease Prediction')

form = st.form(key='form_prediction')
age = form.slider('How old are you?', 20, 90, 25)

sex = form.selectbox(
    'What is your gender?',
    ('Male', 'Female'))


cp = form.selectbox(
    'Chest pain experienced?',
    ('typical angina', 'atypical angina','non-anginal pain','asymptomatic'))

bd = form.slider('Resting blood pressure ?', 90, 200, 100)

chol = form.slider('Your cholesterol measurement', 120, 600, 126)

fbs = form.selectbox(
    'Fasting Blood Sugar more than 120 mg/dl?',
    ('Yes', 'No'))

rer = form.selectbox(
    'resting electrocardiographic results?',
    ('Normal', 'having ST-T wave abnormality','showing probable or definite left ventricular hypertrophy'))

hr = form.slider('Maximum Heart rate ?', 70, 250, 80)

ex = form.selectbox(
    'Exercise included angina?',
    ('Yes', 'No'))

std = form.number_input('ST depression induced by exercise relative to rest')

sl = form.selectbox(
    'slope of the peak exercise ST segment',
    ('upsloping', 'flat','downsloping'))

vs = form.selectbox(
    'The number of major vessels',
    ('0', '1','2','3'))

th = form.selectbox(
    'Thalassemia',
    ('Normal', 'fixed defect','reverasble defect'))

submit_button = form.form_submit_button(label='Predict!',type='primary',use_container_width=1)

if submit_button:
    with st.spinner('Wait for it...'):
        time.sleep(2)
    if(cp=="typical angina"):
         cp_input=0
    if(cp=="atypical angina"):
         cp_input=1
    if(cp=="non-anginal pain"):
         cp_input=2
    if(cp=="asymptomatic"):
         cp_input=3
    if(sex=="Male"):
        sex_input=1
    if(sex=="Female"):
        sex_input=0
    if(fbs=="No"):
        fbs_input=0
    if(fbs=="Yes"):
        fbs_input=1
    if(ex=="No"):
        ex_input=0
    if(ex=="Yes"):
        ex_input=1
    if(sl=="upsloping"):
        sl_input=0
    if(sl=="flat"):
        sl_input=1
    if(sl=="downsloping"):
        sl_input=2
    if(th=="Normal"):
        th_input=0
    if(th=="fixed defect"):
        th_input=1
    if(th=="reverasble defect"):
        th_input=2
    if(rer=="Normal"):
        rer_input=0
    if(rer=="having ST-T wave abnormality"):
        rer_input=1
    if(rer=="showing probable or definite left ventricular hypertrophy"):
        rer_input=2

    instances = [{
        "age":int(age),
        "sex": int(sex_input),
        "cp": int(cp_input),
        "trestbps":int(bd),
        "chol": int(chol),
        "fbs": int(fbs_input),
        "restecg":int(rer_input),
        "thalach": int(hr),
        "exang": int(ex_input),
        "oldpeak": float(std),
        "slope": int(sl_input),
        "ca": int(vs),
        "thal" : int(th_input)
    }]

    input_dic = {
        "age":int(age),
        "sex": sex,
        "chest pain experienced": cp,
        "resting blood pressure":int(bd),
        "cholesterol measurement": int(chol),
        "fasting blood sugar > 120 mg/dl": fbs,
        "resting electrocardiographic measurement":rer,
        "maximum heart rate achieved": int(hr),
        "Exercise induced angina": ex,
        "ST depression induced by exercise relative to rest": float(std),
        "peak exercise ST segment": sl,
        "number of major vessels": vs,
        "thalassemia" : th
    }


    # Importing the dataset
    dataset=pd.read_csv('pages/model/heart.csv')

    X=dataset.drop(['target'],axis=1)
    y=dataset["target"]

    #Training XGBoost
    classifier = XGBClassifier(colsample_bytree=0.6, max_depth = 5, gamma=1.5, min_child_weight=5)
    clf=classifier.fit(X, y)

    input = pd.DataFrame(instances)
    y_pred = clf.predict(input)
    result=y_pred[0]


    with st.expander("Your Condition"):
        st.write(input_dic)
    if(result==0):
        st.success('Your condition is not indicated for Heart Disease', icon="✅")
    else:
        st.error('Your condition indicated for Heart Disease', icon="🚨")
    st.write("How much you can trust this result? learn more [here](/Machine)")