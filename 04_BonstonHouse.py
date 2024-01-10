import streamlit as st
import pandas as pd
# import numpy as np
import shap
# from sklearn import datasets 
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle

# loading model
with open("bonston_model.pkl","rb") as f:
    model = pickle.load(f)
df = pd.read_csv("04_BonstonHousingData.csv")

st.title("Bonston House Price Predictor")
st.write("This web app precits bonston house price based on inputed parameter on sidebar with accuracy of 85%")

#  CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO       B  LSTAT  MEDV
used_feature = ['CRIM', 'INDUS', 'RM', 'AGE', 'DIS', 'B', 'LSTAT']


st.sidebar.header("User Input Parameter")
features={}
features["CRIM"] = st.sidebar.slider("CRIM",float(df["CRIM"].min()), float(df["CRIM"].max()),value=0.1,step=0.1)
features["INDUS"] = st.sidebar.slider("INDUS", \
                          float(df["INDUS"].min()), \
                          float(df["INDUS"].max()), \
                          value=0.1, \
                          step=0.1 \
                          )
# nox = st.sidebar.slider("NOX", float(df["NOX"].min()), float(df["NOX"].max()), value=0.1, step=0.1)
features["RM"] = st.sidebar.slider("RM", float(df["RM"].min()), float(df["RM"].max()), value=0.1, step=0.1)
features["AGE"] = st.sidebar.slider("AGE",float(df["AGE"].min()),float(df["AGE"].max()),value=0.1, step=0.1)
features["DIS"] = st.sidebar.slider("DIS", float(df["DIS"].min()), float(df["DIS"].max()), value=0.1, step=0.1)
features["B"]= st.sidebar.slider("B", float(df["B"].min()), float(df["B"].max()), value=0.1, step=0.1)
features["LSTAT"] = st.sidebar.slider("LSTAT", float(df["LSTAT"].min()), float(df["LSTAT"].max()), value=0.1,step=0.1)

st.write("Parameters Given")
parameter_df = pd.DataFrame(features,index=[0])
st.dataframe(parameter_df)

if st.button("Predict"):
    result = model.predict(parameter_df)
    st.write(f"**Prediction: {result[0]}**")
    st.write("---")

    # explaining model prediction using shap values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(parameter_df)

    st.header("Feature Importance")
    fig = plt.figure()
    plt.title("Feature importance based on SHAP values")
    shap.summary_plot(shap_values)
    st.pyplot(fig,bbox_inches='tight')
    st.write("---")

    fig=plt.figure()
    plt.title("Feature importance based on SHAP values (Bar)")
    shap.summary_plot(shap_values,parameter_df,plot_type="bar")
    st.pyplot(fig,bbox_inches="tight")