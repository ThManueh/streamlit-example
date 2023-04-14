import shap
import sklearn
import streamlit as st
import streamlit.components.v1 as components
import xgboost
import pandas as pd
import numpy as np
@st.cache
def load_data():
    return shap.datasets.adult()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.title("SHAP in Streamlit")

# train XGBoost model
X,y = load_data()

trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")



trainData = trainData.drop('Id', axis=1)
df_num = trainData.select_dtypes(include=['float64', 'int64'])
df_num = df_num[np.isfinite(df_num).all(1)]
X_train = df_num.drop("SalePrice", axis=1)
y_train = df_num["SalePrice"]


model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)


testData = testData.drop('Id', axis=1)
X_test = testData.select_dtypes(include=['float64', 'int64'])
X_test = X_test[np.isfinite(X_test).all(1)]


explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.summary_plot(shap_values, plot_type='violin'))

# visualize the training set predictions


