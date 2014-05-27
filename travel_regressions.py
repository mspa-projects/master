'''
This module fits, tests and evaluates for the McLean County many Regression Models
against a dataframe containing the Cluster of the McLean County dataset.

Inputs:
    A dataframe containing the cluster with McLane County.

Output:
    standart output print of many regression model's:
    * Mean Squared Error
    * R-Squared Score
    * Cross-Validation Average Score
    * The McLean County funding Prediction

'''

import numpy as np  # efficient processing of numerical arrays
import pandas as pd  # pandas for data frame operations
from sklearn import linear_model, svm, neighbors, gaussian_process, tree
from sklearn.svm import NuSVR
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------
# Here is the evaluations common to all regressions
# see: http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
# Create a boiler-plate evaluation, fit, and prediction for McLean county that is
# seperate from the model being passed
# -------------------------------------------------

def regression_evaluation (model, x, y, x_train, x_test, y_train, y_test, x_McLean) :
    # first fit with training for later compare against testing
    model_fit = model.fit(x_train, y_train)
    
    y_test_pred = model_fit.predict(x_test)
    MSE = mean_squared_error(y_test, y_test_pred)
    print("Mean Squared Error: %0.3f" % round(MSE,3))
    #print("Residual sum of squares: %.3f" % np.mean((regr.predict(x_test) - y_test) ** 2))  # a.k.a. Mean Squared Error
    print("R-Squared Score: %0.3f" % round(r2_score(y_test, y_test_pred),3))
    #print('Variance score: %.3f' % regr.score(x_test, y_test))  # a.k.a. R-Squared Score

    # how about multi-fold cross-validation with 5 folds
    cv_results = cross_val_score(model, x, y, cv=5)
    print("Cross-Validation Average Score: %0.3f" % round(cv_results.mean(),3))
    
    y_pred = model_fit.predict(x_McLean)
    sigma = np.sqrt(MSE)
    print('McLean Prediction: %0.0f' % round(y_pred,0))  # for fitting use 
    print('95%% confidence interval: %i < McLean < %i' % (round(y_pred - 1.9600 * sigma, 0), round(y_pred + 1.9600 * sigma, 0)))


def travel_regressions (df_McLeanCluster) :
    # -------------------------------------------------
    # munge the dataframe targeting McLean County
    # into components for processing
    # Note: the x_McLean array and x (all) array are required to be the same dimension.
    # -------------------------------------------------
    
    #df_McLeanCluster = pd.read_csv('df_countyCluster.csv', ',', dtype = {'countyid':str});
    #list(df_McLeanCluster.columns.values)  # sanity check
    
    # Lets extract the McLean County tuple and target it for regressions
    df_McLean = df_McLeanCluster[df_McLeanCluster.countyid == '17113']
    x_McLean = np.array([
        #np.array(df_McLean['usps']),
        #np.array(df_McLean['name']),
        #np.array(df_McLean['countyid']),
        np.array(df_McLean['pop10']),
        np.array(df_McLean['hu10']),
        #np.array(df_McLean['aland']),
        #np.array(df_McLean['awater']),
        np.array(df_McLean['aland_sqmi'])
        #np.array(df_McLean['awater_sqmi']),
        #np.array(df_McLean['statepop']),
        #np.array(df_McLean['statearea']),
        #np.array(df_McLean['perc_pop']),
        #np.array(df_McLean['perc_area']),
        #np.array(df_McLean['est_pmiles2007_11']),
        #np.array(df_McLean['est_ptrp2007_11']),
        #np.array(df_McLean['est_vmiles2007_11']),
        #np.array(df_McLean['est_vtrp2007_11']),
        #np.array(df_McLean['median_hh_inc2007_11']),
        #np.array(df_McLean['mean_hh_veh2007_11']),
        #np.array(df_McLean['mean_hh_mem2007_11']),
        #np.array(df_McLean['pct_owner2007_11']),
        #np.array(df_McLean['mean_hh_worker2007_11']),
        #np.array(df_McLean['pct_lchd2007_11']),
        #np.array(df_McLean['pct_lhd12007_11']),
        #np.array(df_McLean['pct_lhd22007_11']),
        #np.array(df_McLean['pct_lhd42007_11']),
        #np.array(df_McLean['corrected_funding']),
        #np.array(df_McLeanCluster['label'])
        ]).T
    
    # Remove the McLean County tuple from the McLeanCluster dataframe
    df_McLeanCluster = df_McLeanCluster[df_McLeanCluster.countyid != '17113']
    
    # Remove all other clusters not associated to the McLean County Cluster
    #print ('All counties except McLean: %i' % len(df_McLeanCluster))  # sanity check
    #print (list(df_McLeanCluster.columns.values))  # sanity check
    df_McLeanCluster = df_McLeanCluster[df_McLeanCluster.label == int(df_McLean['label'])]
    #print ('Only other counties in McLean cluster: %i' % len(df_McLeanCluster))  # sanity check
    
    
    df_McLeanCluster = df_McLeanCluster.reset_index(drop=True)  # keep index tidy
    
    y = df_McLeanCluster['corrected_funding']
    x = np.array([
        #np.array(df_McLeanCluster['usps']),
        #np.array(df_McLeanCluster['name']),
        #np.array(df_McLeanCluster['countyid']),
        np.array(df_McLeanCluster['pop10']),
        np.array(df_McLeanCluster['hu10']),
        #np.array(df_McLeanCluster['aland']),
        #np.array(df_McLeanCluster['awater']),
        np.array(df_McLeanCluster['aland_sqmi'])
        #np.array(df_McLeanCluster['awater_sqmi']),
        #np.array(df_McLeanCluster['statepop']),
        #np.array(df_McLeanCluster['statearea']),
        #np.array(df_McLeanCluster['perc_pop']),
        #np.array(df_McLeanCluster['perc_area']),
        #np.array(df_McLeanCluster['est_pmiles2007_11']),
        #np.array(df_McLeanCluster['est_ptrp2007_11']),
        #np.array(df_McLeanCluster['est_vmiles2007_11']),
        #np.array(df_McLeanCluster['est_vtrp2007_11']),
        #np.array(df_McLeanCluster['median_hh_inc2007_11']),
        #np.array(df_McLeanCluster['mean_hh_veh2007_11']),
        #np.array(df_McLeanCluster['mean_hh_mem2007_11']),
        #np.array(df_McLeanCluster['pct_owner2007_11']),
        #np.array(df_McLeanCluster['mean_hh_worker2007_11']),
        #np.array(df_McLeanCluster['pct_lchd2007_11']),
        #np.array(df_McLeanCluster['pct_lhd12007_11']),
        #np.array(df_McLeanCluster['pct_lhd22007_11']),
        #np.array(df_McLeanCluster['pct_lhd42007_11']),
        #np.array(df_McLeanCluster['corrected_funding']),
        #np.array(df_McLeanCluster['label'])
        ]).T
    
    
    
    # set up data for training-test regimen to be common for all regression models
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 9999)
    
    
    # -------------------------------------------------
    # Fit Ordinary Least Squares model
    # -------------------------------------------------
    print('\n-----Ordinary Least Squares Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.LinearRegression(),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Ridge Regression model
    #
    # Ridge regression addresses some of the problems of Ordinary Least Squares by 
    # imposing a penalty on the size of coefficients. The ridge coefficients minimize
    # a penalized residual sum of squares.
    # Here, alpha >= 0 is a complexity parameter that controls the amount of 
    # shrinkage: the larger the value of \alpha, the greater the amount of shrinkage
    # and thus the coefficients become more robust to collinearity.
    # -------------------------------------------------
    print('\n-----Ridge Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.Ridge (alpha = .9),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Lasso Regression model
    #
    # From scikit-learn:
    # The Lasso is a linear model that estimates sparse coefficients. It is useful 
    # in some contexts due to its tendency to prefer solutions with fewer parameter
    # values, effectively reducing the number of variables upon which the given 
    # solution is dependent. For this reason, the Lasso and its variants are 
    # fundamental to the field of compressed sensing.
    # -------------------------------------------------
    print('\n-----Lasso Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.Lasso (alpha = 0.3),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Least Angle Regression model
    #
    # From scikit-learn:
    # Least-angle regression (LARS) is a regression algorithm for high-dimensional 
    # data, developed by Bradley Efron, Trevor Hastie, Iain Johnstone and Robert 
    # Tibshirani.
    # -------------------------------------------------
    print('\n-----Least Angle Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.Lars(n_nonzero_coefs=10),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Bayesian Ridge Regression model
    #
    # From scikit-learn:
    # Bayesian regression techniques can be used to include regularization parameters
    # in the estimation procedure: the regularization parameter is not set in a hard
    # sense but tuned to the data at hand.
    # -------------------------------------------------
    print('\n-----Bayesian Ridge Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.BayesianRidge(),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Support Vector Regression (SVR) using linear and non-linear kernels model
    #
    # From scikit-learn:
    # The method of Support Vector Classification can be extended to solve regression
    # problems. This method is called Support Vector Regression.
    # There are two flavors of Support Vector Regression: SVR and NuSVR.
    # -------------------------------------------------
    print('\n-----Support Vector Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        svm.SVR(),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    print('\n-----Nu Support Vector Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        NuSVR(C=1.0, nu=0.1),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Stochastic Gradient Descent Regression model
    #
    # From scikit-learn:
    # The class SGDRegressor implements a plain stochastic gradient descent learning 
    # routine which supports different loss functions and penalties to fit linear 
    # regression models. SGDRegressor is well suited for regression problems with a 
    # large number of training samples (> 10.000), for other problems we recommend 
    # Ridge, Lasso, or ElasticNet.
    # -------------------------------------------------
    print('\n-----Stochastic Gradient Descent loss="huber" Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.SGDRegressor(loss="huber"),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    print('\n-----Stochastic Gradient Descent loss="epsilon_insensitive" Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        linear_model.SGDRegressor(loss="epsilon_insensitive"),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Nearest Neighbors Regression model
    #
    # From scikit-learn:
    # Neighbors-based regression can be used in cases where the data labels are 
    # continuous rather than discrete variables. The label assigned to a query point
    # is computed based the mean of the labels of its nearest neighbors.
    # -------------------------------------------------
    print('\n-----Nearest Neighbors Regression weights="uniform" Model Accuracy in Test Set-----')
    regression_evaluation (
        neighbors.KNeighborsRegressor(n_neighbors=5, weights="uniform"),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    print('\n-----Nearest Neighbors Regression weights="distance" Model Accuracy in Test Set-----')
    regression_evaluation (
        neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance"),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Gaussian Processes Regression model
    #
    # From scikit-learn:
    # A generic supervised learning method primarily designed to solve regression problems.
    # Note: Takes very long to run (over night might suffice).
    # -------------------------------------------------
    #print('\n-----Gaussian Processes Regression Model Accuracy in Test Set-----')
    #regression_evaluation (
    #    gaussian_process.GaussianProcess(corr='squared_exponential', theta0=1e-1,
    #        thetaL=1e-3, thetaU=1, nugget=1, random_start=100),
    #    x, y, x_train, x_test, y_train, y_test, x_McLean)
    
    
    # -------------------------------------------------
    # Fit Decision Tree Regression model
    #
    # From scikit-learn:
    # Decision trees can also be applied to regression problems, using the DecisionTreeRegressor class.
    # -------------------------------------------------
    print('\n-----Decision Tree Regression Model Accuracy in Test Set-----')
    regression_evaluation (
        tree.DecisionTreeRegressor(),
        x, y, x_train, x_test, y_train, y_test, x_McLean)
    
