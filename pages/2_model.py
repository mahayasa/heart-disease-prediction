import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Machine Learning Explanation",
    page_icon="🤖",
)

st.write("# Machine Learning Explanation")

st.markdown(
    """
    Heart disease prediction in this application is using AutoML model from google cloud services, with 1026 row of dataset. You can check the source of dataset on [kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv), and for more data explanation detail you can check this [link](https://www.kaggle.com/code/tentotheminus9/what-causes-heart-disease-explaining-the-model/notebook)
    ### How was the model evaluation?
    - F1-Score : 0.7787611
    - Precision :77.9%
    - Recall : 77.9%
    - AUC ROC : 0.894
"""
)

with st.expander("See explanation"):
    st.write("**Precision** is percentage of predictions that were correct (positive). The higher the precision, the fewer false positives predicted.")
    st.write("**Recall** is percentage of all ground truth items that were successfully predicted by the model. The higher the recall, the fewer false negatives, or the fewer predictions missed.")
    st.write("**F1-score** is harmonic mean of precision and recall. F1 measurement is use for a balance between precision and recall and there's an uneven class distribution")
    st.write("**AUC ROC** is harmonic mean of precision and recall. F1 measurement is use for a balance between precision and recall and there's an uneven class distribution")

st.write("### Confusion Matrix")
image = Image.open('pages/image/cfm.png')
st.image(image)

st.write("### How much you can trust this model?")
st.write("Not 100% to trust, this model only used minimum data training from old case, and achived 77.9% in precision and recall, if you have symptoms in heart disease consult to cardiologist immediately")

