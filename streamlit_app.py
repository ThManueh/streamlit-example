import streamlit as st
from streamlit_shap import st_shap
import shap

from sklearn.model_selection import train_test_split
import xgboost

import numpy as np
import pandas as pd


@st.experimental_memo
def load_data():
    return shap.datasets.adult()

@st.experimental_memo
def load_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)
    params = {
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    model = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
    return model

st.title("SHAP in Streamlit")

# train XGBoost model
X,y = load_data()
X_display,y_display = shap.datasets.adult(display=True)

model = load_model(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

st_shap(shap.plots.waterfall(shap_values[0]), height=300)
st_shap(shap.plots.beeswarm(shap_values), height=300)


# import shap
# import sklearn
# import streamlit as st
# import streamlit.components.v1 as components
# import xgboost
# import pandas as pd
# from streamlit_shap import st_shap
# import numpy as np
# @st.cache
# def load_data():
#     return shap.datasets.adult()

# st.title("SHAP in Streamlit")

# # train XGBoost model
# X,y = load_data()

# trainData = pd.read_csv("train.csv")
# testData = pd.read_csv("test.csv")



# trainData = trainData.drop('Id', axis=1)
# df_num = trainData.select_dtypes(include=['float64', 'int64'])
# df_num = df_num[np.isfinite(df_num).all(1)]
# X_train = df_num.drop("SalePrice", axis=1)
# y_train = df_num["SalePrice"]


# model = sklearn.linear_model.LinearRegression()
# model.fit(X_train, y_train)


# testData = testData.drop('Id', axis=1)
# X_test = testData.select_dtypes(include=['float64', 'int64'])
# X_test = X_test[np.isfinite(X_test).all(1)]


# explainer = shap.Explainer(model.predict, X_test)
# shap_values = explainer(X_test)
# prin(shap_values)
# st_shap(shap.plots.beeswarm(shap_values), height=800)



