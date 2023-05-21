import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
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

with st.expander("Code snapshot"):
    code = '''
    from xgboost import XGBClassifier
    
    # Receiving input from user
    instances = [{
    "age":int(age),
    "sex": int(sex),
    "cp": int(cp),
    "trestbps":int(bd),
    "chol": int(chol),
    "fbs": int(fbs),
    "restecg":int(rer),
    "thalach": int(hr),
    "exang": int(ex),
    "oldpeak": float(std),
    "slope": int(sl),
    "ca": int(vs),
    "thal" : int(th)
    }]

    # Importing the dataset
    dataset=pd.read_csv('pages/model/heart.csv')

    # Divide data by feature and target
    X=dataset.drop(['target'],axis=1)
    y=dataset["target"]

    # Training XGBoost
    classifier = XGBClassifier(colsample_bytree=0.6, max_depth = 5, gamma=1.5, min_child_weight=5)
    clf=classifier.fit(X, y)

    # Make prediction
    input = pd.DataFrame(instances)
    y_pred = clf.predict(input)'''
    st.code(code, language='python')