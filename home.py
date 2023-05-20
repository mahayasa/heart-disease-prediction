import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to prototype of heart disease prediction! ❤")

st.markdown(
    """
    This application only a prototype for learn machine learning. Machine learning model from this app using [XGBoost](https://xgboost.ai/about), XGBoost (Extreme Gradient Boosting) is an optimized and scalable gradient boosting framework for machine learning. It is based on the gradient boosting algorithm, which is an ensemble method that combines multiple weak prediction models (typically decision trees) to create a more accurate and powerful model.
    ### How to use the app?
    - Go to [prediction](/prediction)
    - Learn about model [evaluation](/Machine)
    ### Open source!
    - check the app code on [github](https://github.com/mahayasa/heart-disease-prediction)

"""
)