import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from google.cloud import aiplatform

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

bd = form.slider('Resting blood pressure ?', 90, 120, 100)

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
         cp="0"
    if(cp=="atypical angina"):
         cp="1"
    if(cp=="non-anginal pain"):
         cp="2"
    if(cp=="asymptomatic"):
         cp="3"
    if(sex=="Male"):
        sex="1"
    if(sex=="Female"):
        sex="0"
    if(fbs=="No"):
        fbs="0"
    if(fbs=="Yes"):
        fbs="1"
    if(ex=="No"):
        ex="0"
    if(ex=="Yes"):
        ex="1"
    if(sl=="upsloping"):
        sl="0"
    if(sl=="flat"):
        sl="1"
    if(sl=="downsloping"):
        sl="2"
    if(th=="Normal"):
        th="0"
    if(th=="fixed defect"):
        th="1"
    if(th=="reverasble defect"):
        th="2"
    if(rer=="Normal"):
        rer="0"
    if(rer=="having ST-T wave abnormality"):
        rer="1"
    if(rer=="showing probable or definite left ventricular hypertrophy"):
        rer="2"
    instances = [{
        "age": str(age),
        "ca": str(vs),
        "chol": str(chol),
        "cp": cp,
        "exang":ex,
        "fbs": fbs,
        "oldpeak": str(std),
        "restecg":rer,
        "sex": sex,
        "slope": sl,
        "thal": th,
        "thalach": str(hr),
        "trestbps": str(bd)
    }]

    # Set your project and region
    project = "666115095520"
    region = "us-central1"

    # Initialize the client
    client = aiplatform.gapic.JobServiceClient(client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"})

    # Set the display name and job spec
    display_name = "heart-diseases"

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        # for more info on the instance schema, please use get_model_sample.py
        # and look at the yaml found in instance_schema_uri

    endpoint = client.endpoint_path(
            project=project, location=region, endpoint="6305017488086138880"
        )
    response = client.predict(
            endpoint=endpoint, instances=instances
        )
        # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
            dict(prediction)

    pred_pos=prediction["scores"][0]
    pred_neg=prediction["scores"][1]

    pred_pos=round(pred_pos*100,2)
    pred_neg=round(pred_neg*100,2)

    print_pred_pos=str(pred_pos)+" %"
    print_pred_neg=str(pred_neg)+" %"

    if(pred_neg>pred_pos):
        st.success('Your condition is not indicated for Heart Disease', icon="✅")
    else:
        st.error('Your condition indicated for Heart Disease', icon="🚨")
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Heart Disiease indication', 'Not Indicated'
    sizes = [pred_pos, pred_neg]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    col1, col2= st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.metric(label="Heart Disiease indication", value=print_pred_pos)
        st.metric(label="Not Indicated", value=print_pred_neg)
        st.caption('How much you can trust this result? learn more [here](/model)')