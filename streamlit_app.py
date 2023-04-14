import streamlit as st
from streamlit_shap import st_shap
import shap

from sklearn.model_selection import train_test_split
import xgboost

import numpy as np
import pandas as pd

trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

trainData = trainData.drop('Id', axis=1)
df_num = trainData.select_dtypes(include=['float64', 'int64'])
df_num = df_num[np.isfinite(df_num).all(1)]

X_train = df_num.drop("SalePrice", axis=1)
y_train = df_num["SalePrice"]

# print(X_train)
# print(X_train.shape)

model = LinearRegression()
model.fit(X_train, y_train)

testData = testData.drop('Id', axis=1)


print(testData)
X_test = testData.select_dtypes(include=['float64', 'int64'])
X_test = X_test[np.isfinite(X_test).all(1)]



# X_test = X_test.drop('Alley')
# X_test = X_test.drop('FireplaceQu')
# X_test = X_test.drop('MasVnrType')
# X_test = X_test.drop('PoolQC')

#print(X_test)


explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)
#print(shap_values)
#shap.waterfall_plot(shap_values)
#print(model.predict(X_test))
#shap.plots.beeswarm(shap_values)

st_shap(shap.plots.waterfall(shap_values[0]), height=300)
st_shap(shap.plots.beeswarm(shap_values), height=400)


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



