import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to prototype of heart disease prediction! ❤")

st.markdown(
    """
    This application only a prototype for learn machine learning.Machine learning model from this app use [AutoML](https://cloud.google.com/automl?hl=id) from google cloud services
    ### How to use the app?
    - Go to [prediction](/prediction)
    - Learn about model [evaluation](/model)

"""
)