import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import sklearn.metrics as mt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from numpy import mean
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

st.set_page_config(
    page_title="Machine Learning Explanation",
    page_icon="🤖",
)

st.write("# Machine Learning Explanation")

dataset=pd.read_csv('pages/model/heart.csv')


#filling missing value with mean
#dataset['TotalCharges'].replace(to_replace = 0, value = dataset['TotalCharges'].mean(), inplace=True)
X=dataset.drop(['target'],axis=1)
y=dataset["target"]

#kfold cross validation
cv = KFold(n_splits=5, random_state=1, shuffle=True)

#Training XGBoost
#classifier = XGBClassifier(eta=0.3, max_depth = 4, gamma=0, min_child_weight=1)
classifier = XGBClassifier(colsample_bytree=0.6, max_depth = 5, gamma=1.5, min_child_weight=5)
clf=classifier.fit(X, y)

score=cross_val_score(classifier, X, y, scoring='f1', cv=cv, n_jobs=-1)
pr=cross_val_score(classifier, X, y, scoring='precision', cv=cv, n_jobs=-1)
rc=cross_val_score(classifier, X, y, scoring='recall', cv=cv, n_jobs=-1)
auc1=cross_val_score(classifier, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)



st.markdown(
    """
    Heart disease prediction in this application is using [XGBoost](https://xgboost.ai/about) model, with 1025 row of dataset. You can check the source of dataset on [kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv), and for more data explanation detail you can check this [link](https://www.kaggle.com/code/tentotheminus9/what-causes-heart-disease-explaining-the-model/notebook)
    ### How was the model evaluation?
"""
)

st.write('- Precision: %.3f' % (mean(pr)))
st.write('- Recall: %.3f' % (mean(rc)))
st.write('- F1 score: %.3f' % (mean(score)))
st.write('- AUC ROC: %.3f' % (mean(auc1)))
with st.expander("See explanation"):
    st.write("**Precision** is percentage of predictions that were correct (positive). The higher the precision, the fewer false positives predicted.")
    st.write("**Recall** is percentage of all ground truth items that were successfully predicted by the model. The higher the recall, the fewer false negatives, or the fewer predictions missed.")
    st.write("**F1-score** is harmonic mean of precision and recall. F1 measurement is use for a balance between precision and recall and there's an uneven class distribution")
    st.write("**AUC ROC** stands for Area Under the Receiver Operating Characteristic curve. The classifier assigns each instance a probability or a score of belonging to the positive class. The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.")

st.write("### Confusion Matrix")
y_pred = cross_val_predict(classifier, X, y, cv=cv)

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
fig=plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)

st.write("### How much you can trust this model?")
st.write("Not 100% to trust, this model only used minimum data training from old case, not using preprocessing and another optimization method, altough the model achived 97% in F-1 Score. If you have symptoms in heart disease consult to cardiologist immediately.")

