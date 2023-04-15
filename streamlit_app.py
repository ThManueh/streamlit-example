import pandas as pd
import shap
import sklearn
import numpy as np
import streamlit as st
import lightgbm as lgb
import optuna
import streamlit.components.v1 as components
from streamlit_shap import st_shap


@st.cache
def bind_socket():
    def objective(trial):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': trial.suggest_int("num_leaves", 20, 3000, step=20),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
            'bagging_fraction': trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
            'bagging_freq': trial.suggest_categorical("bagging_freq", [1]),
            'verbose': 0
        }
        trainData = pd.read_csv("train.csv")
        trainData = trainData.drop('Id', axis=1)
        df_num = trainData.select_dtypes(include=['float64', 'int64'])
        df_num = df_num[np.isfinite(df_num).all(1)]

        X_train = df_num.drop("SalePrice", axis=1)
        y_train = df_num["SalePrice"]

        lgb_train = lgb.Dataset(X_train, y_train)
        gbm = lgb.train(params=params, train_set=lgb_train)

        y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)

        error = sklearn.metrics.mean_squared_error(y_train, y_pred)

        return error

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1);

    # print("The Best value was: ", study.best_value);

    return study.best_params, study.best_value


best_params, best_value = bind_socket()

MSSubClass = st.sidebar.slider('Identifies the type of dwelling involved in the sale', 20, 190)
LotFrontage = st.sidebar.slider('Linear feet of street connected to property', 20, 500)
LotArea = st.sidebar.slider('Lot size in square feet', 50, 250000)
OverallQual = st.sidebar.slider('Rates the overall material and finish of the house', 1, 10)
OverallCond = st.sidebar.slider('Rates the overall condition of the house', 1, 10)
LotFrontage = st.sidebar.slider('Linear feet of street connected to property', 20, 330)
OverallQual = st.sidebar.slider('Rates the overall material and finish of the house', 0, 10)
OverallCond = st.sidebar.slider('Rates the overall condition of the house', 0, 10)
YearBuilt = st.sidebar.slider('Year Built. Remodel date (same as construction date if no remodeling or additions)',1872, 2010)
YearRemodAdd = st.sidebar.slider('Remodel date (same as construction date if no remodeling or additions)', 1950, 2010)
MasVnrArea = st.sidebar.slider('Masonry veneer area in square feet', 0, 2000)
BsmtFinSF1 = st.sidebar.slider('Type 1 finished square feet', 0, 6000)
BsmtFinSF2 = st.sidebar.slider('Type 2 finished square feet', 0, 6000)
BsmtUnfSF = st.sidebar.slider('Total square feet of basement area', 0, 2336)
TotalBsmtSF = st.sidebar.slider('Total square feet of basement area', 0, 6110)
A1stFlrSF = st.sidebar.slider('First floor square feet', 0, 2065)
A2ndFlrSF = st.sidebar.slider('Second floor square feet', 0, 2065)
LowQualFinSF = st.sidebar.slider('Low quality finished square feet (all floors)', 0, 10)
GrLivArea = st.sidebar.slider('Above grade (ground) living area square feet', 10, 5000)
BsmtHalfBath = st.sidebar.slider('Basement half bathrooms', 0, 10)
BsmtFullBath = st.sidebar.slider('Basement full bathrooms', 0, 10)
FullBath = st.sidebar.slider('Full bathrooms above grade', 0, 10)
HalfBath = st.sidebar.slider('Half baths above grade', 0, 10)
BedroomAbvGr = st.sidebar.slider('Bedrooms above grade (does NOT include basement bedrooms)', 0, 10)
KitchenAbvGr = st.sidebar.slider('Kitchens above grade', 0, 10)
TotRmsAbvGrd = st.sidebar.slider('Total rooms above grade (does not include bathrooms)', 0, 10)
Fireplaces = st.sidebar.slider('Number of fireplaces', 0, 10)
GarageYrBlt = st.sidebar.slider('Year garage was built', 1900, 2010)
GarageCars = st.sidebar.slider('Size of garage in car capacity', 0, 20)
GarageArea = st.sidebar.slider('Size of garage in square feet', 0, 600)
WoodDeckSF = st.sidebar.slider('Wood deck area in square feet', 0, 600)
OpenPorchSF = st.sidebar.slider('Open porch area in square feet', 0, 600)
EnclosedPorch = st.sidebar.slider(' Enclosed porch area in square feet', 0, 600)
S3snPorch = st.sidebar.slider('Three season porch area in square feet', 0, 600)
ScreenPorch = st.sidebar.slider('Screen porch area in square feet', 0, 600)
PoolArea = st.sidebar.slider('Pool area in square feet', 0, 1000)
MiscVal = st.sidebar.slider('Value of miscellaneous feature', 0, 1000)
MoSold = st.sidebar.slider('In which year was month it was sold?', 1, 12)
YrSold = st.sidebar.slider('In which year was it sold?', 2006, 2010)

mydict = [
    {'MSSubClass': MSSubClass,
     'LotFrontage': LotFrontage,
     'LotArea': LotArea,
     'OverallQual': OverallQual,
     'OverallCond': OverallCond,
     'YearBuilt': YearBuilt,
     'YearRemodAdd': YearRemodAdd,
     'MasVnrArea': MasVnrArea,
     'BsmtFinSF1': BsmtFinSF1,
     'BsmtFinSF2': BsmtFinSF2,
     'BsmtUnfSF': BsmtUnfSF,
     'TotalBsmtSF': TotalBsmtSF,
     '1stFlrSF': A1stFlrSF,
     '2ndFlrSF': A2ndFlrSF,
     'LowQualFinSF': LowQualFinSF,
     'GrLivArea': GrLivArea,
     'BsmtFullBath': BsmtFullBath,
     'BsmtHalfBath': BsmtHalfBath,
     'BsmtFullBath': BsmtFullBath,
     'FullBath': FullBath,
     'HalfBath': HalfBath,
     'BedroomAbvGr': BedroomAbvGr,
     'KitchenAbvGr': KitchenAbvGr,
     'TotRmsAbvGrd': TotRmsAbvGrd,
     'Fireplaces': Fireplaces,
     'GarageYrBlt': GarageYrBlt,
     'GarageCars': GarageCars,
     'GarageArea': GarageArea,
     'WoodDeckSF': WoodDeckSF,
     'OpenPorchSF': OpenPorchSF,
     'EnclosedPorch': EnclosedPorch,
     '3SsnPorch': S3snPorch,
     'ScreenPorch': ScreenPorch,
     'PoolArea': PoolArea,
     'MiscVal': MiscVal,
     'MoSold': MoSold,
     'YrSold': YrSold,
     }]


@st.cache
def train():
    # print('Minimum mean squared error: ' + str(study.best_value))
    print('Best parameter: ' + str(best_params))

    trainData = pd.read_csv("train.csv")
    testData = pd.read_csv("test.csv")

    trainData = trainData.drop('Id', axis=1)
    df_num = trainData.select_dtypes(include=['float64', 'int64'])
    df_num = df_num[np.isfinite(df_num).all(1)]

    X_train = df_num.drop("SalePrice", axis=1)
    y_train = df_num["SalePrice"]

    lgb_train = lgb.Dataset(X_train, y_train)
    gbm = lgb.train(params=best_params, train_set=lgb_train)

    # lower = lgb.LGBMRegressor(objective='quantile', alpha=1 - 0.95)
    # lgb_train = lgb.Dataset(X_train, y_train)
    # lower = lower.train(params=best_params, train_set=lgb_train)
    #
    # upper = lgb.LGBMRegressor(objective='quantile', alpha=0.95)
    # lgb_train = lgb.Dataset(X_train, y_train)
    # upper = upper.train(params=best_params, train_set=lgb_train)

    return gbm;


def ShapValue():
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

    df = pd.DataFrame(mydict)
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(df)

    return shap_values


gbm = train()

st.header('House price prediction')
st.write('https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview')


def test(gbm):
    df = pd.DataFrame(mydict)
    # gbm = gbm.set_attr('lower_bound')
    # st.write("lower : ", gbm.lower_bound())
    # st.write("Upper : ", gbm.upper_bound())
    #     desired_representationBellow = "{:0,.4f}".format((gbm.predict(df, num_iteration=gbm.best_iteration)[0])-best_value)
    #     desired_representationAbove = "{:0,.4f}".format((gbm.predict(df, num_iteration=gbm.best_iteration)[0])+best_value)

    desired_representationLow = "{:0,.4f}".format(gbm.predict(df, num_iteration=gbm.best_iteration)[0]);
    st.write("The predicted house value is: ", desired_representationLow)
    # st.write(gbm.lower_bound())

    # desired_representationa = "{:0,.4f}".format(best_value)
    # st.write(desired_representationa)


buttomPress = st.button("Calculate the house price")
if (buttomPress):
    test(gbm)
    shap_values = ShapValue()
    st_shap(shap.plots.bar(shap_values,max_display=12))
    st_shap(shap.plots.waterfall(shap_values[0], max_display=12))
    #st_shap(shap.plots.dependence_plot(shap_values))

    st_shap(shap.plots.scatter(shap_values[:,:]))

    df = pd.DataFrame(mydict)

    testData = pd.read_csv("test.csv")
    testData = testData.drop('Id', axis=1)
    X_test = testData.select_dtypes(include=['float64', 'int64'])
    X_test = X_test[np.isfinite(X_test).all(1)]
    shap_interaction_values = shap.TreeExplainer(gbm).shap_interaction_values(X_test.iloc[:2000, :])
    st_shap(shap.summary_plot(shap_interaction_values, X_test.iloc[:2000, :]))




    #st_shap(shap.summary_plot(ShapValue(), plot_type='violin'))
    buttomPress = False
