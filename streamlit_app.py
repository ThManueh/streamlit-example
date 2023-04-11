import pandas as pd
import shap
import sklearn
import numpy as np

trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

trainData = trainData.drop('Id', axis=1)
df_num = trainData.select_dtypes(include=['float64', 'int64'])
df_num = df_num[np.isfinite(df_num).all(1)]

X_train = df_num.drop("SalePrice", axis=1)
y_train = df_num["SalePrice"]

# print(X_train)
# print(X_train.shape)

model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)

testData = testData.drop('Id', axis=1)
X_test = testData.select_dtypes(include=['float64', 'int64'])
X_test = X_test[np.isfinite(X_test).all(1)]

# print(X_test)


explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)
print(shap_values)
#shap.waterfall_plot(shap_values)

#shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, plot_type='violin')



# shap.summary_plot(shap_values, plot_type='violin')
#
#
# mydict = [
#   {'MSSubClass': 20,
#    'LotFrontage': 80,
#    'LotArea':11150,
#    'OverallQual': 4,
#    'OverallCond': 5,
#    'YearBuilt': 1967,
#    'YearRemodAdd': 1968,
#    'MasVnrArea': 1978,
#    'BsmtFinSF1': 456,
#    'BsmtFinSF2': 124,
#    'BsmtUnfSF': 124,
#    'TotalBsmtSF': 883,
#    '1stFlrSF': 789,
#    '2ndFlrSF': 0,
#    'LowQualFinSF': 0,
#    'GrLivArea':839,
#    'BsmtFullBath':0,
#   'BsmtHalfBath':0,
#   'BsmtFullBath':0,
#   'FullBath': 1,
#   'HalfBath': 1,
#    'BedroomAbvGr': 2,
#    'KitchenAbvGr': 1,
#    'TotRmsAbvGrd': 5,
#    'Fireplaces': 0,
#    'GarageYrBlt': 1967,
#    'GarageCars': 1,
#    'GarageArea': 789,
#    'WoodDeckSF': 140,
#    'OpenPorchSF': 0,
#
#    'EnclosedPorch': 0,
#    '3SsnPorch': 0,
#    'ScreenPorch': 120,
#    'PoolArea': 0,
#
#    'MiscVal': 0,
#    'MoSold': 6,
#    'YrSold': 2010,
#    }]
#
#
# df = pd.DataFrame(mydict)
# print(df.shape)
# print(df)
#
# #X_test = df;


# print(X_test)
# print(model.predict(X_test))

