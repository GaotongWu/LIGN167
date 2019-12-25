# My name: Gaotong Wu
# My contributions: All of the problems

import numpy as np
import matplotlib.pyplot as plt

#Problem 1
def compute_slope_estimator(x,y):
    sum_xy=0
    sum_x2=0
    n=len(x)
    x_bar=np.sum(x)/n
    y_bar=np.sum(y)/n
    for i in range(n):
        sum_xy=sum_xy+x[i]*y[i]
    sum_xy+n*x_bar*y_bar
    for i in range(n):
        sum_x2=sum_x2+x[i]*x[i]
    slope=(sum_xy-n*x_bar*y_bar)/(sum_x2-n*x_bar*x_bar)
    return slope


#Problem 2
def compute_intercept_estimator(x,y):
    n=len(x)
    x_bar=np.sum(x)/n
    y_bar=np.sum(y)/n
    slope=compute_slope_estimator(x,y)
    intercept=y_bar-slope*x_bar
    return intercept


#Problem 3
def train_model(x,training_set):
    slope=compute_slope_estimator(x,training_set)
    intercept=compute_intercept_estimator(x,training_set)
    return slope,intercept


#Problem 4
def sample_linear_model(x,slope,intercept,sd):
    n=len(x)
    y=np.zeros(shape=(n,))
    for i in range(n):
        epsilon=np.random.normal(0,sd)
        y[i]=slope*x[i]+intercept+epsilon
    return y


#Problem 5
def sample_datasets(x_vals,a,b,sd,n):
    y=np.zeros(shape=(n,len(x_vals)))
    for i in range(n):
        y[i,:]=sample_linear_model(x_vals,a,b,sd)
    return y


#Problem 6
def compute_average_estimated_slope(x_vals):
    y=sample_datasets(x_vals,2,0.5,1.5,1000)
    slope=0
    for i in range(1000):
        slope=slope+compute_slope_estimator(x_vals,y[i,:])
    avg_slope=slope/1000
    return avg_slope


#Problem 7

# Free Response: From the results, we can observe that the  average estimated slope does not significantly change and it is around 2


#Problem 8

#Free Response: From the results, we can observe that the  average squared error decreases as n increases

def compute_estimated_slope_error(x_vals):
    y=sample_datasets(x_vals,2,0.5,1.5,1000)
    error=0
    for i in range(1000):
        error=error+(2-compute_slope_estimator(x_vals,y[i,:]))**2
    avg_error=error/1000
    return avg_error


# Problem 9 

#Free Response: From the histograms, we can observe that as n increases, the mean does not significantly change and it's around 2 while the standard deviation decreases


# Problem 10
def calculate_prediction_error(y,y_hat):
    n=len(y)
    Sum=0
    for i in range(n):
        Sum=Sum+(y[i]-y_hat[i])**2
    return Sum/n


# Problem 11

#Free Response: As n increases, the average training set error increases

def average_training_set_error(x_vals):
    y=sample_datasets(x_vals,2,0.5,1.5,500)
    slope=np.zeros(shape=(500,1))
    intercept=np.zeros(shape=(500,1))
    for i in range(500):
        slope[i]=compute_slope_estimator(x_vals,y[i,:])
        intercept[i]=compute_intercept_estimator(x_vals,y[i,:])
    y_hat=slope*x_vals+intercept
    error=calculate_prediction_error(y,y_hat)
    return sum(error)/500


# Problem 12

#Free Response: As n increases, the average training set error decreases

def average_test_set_error(x_vals):
    y_train=sample_datasets(x_vals,2,0.5,1.5,500)
    slope=np.zeros(shape=(500,1))
    intercept=np.zeros(shape=(500,1))
    for i in range(500):
        slope[i]=compute_slope_estimator(x_vals,y_train[i,:])
        intercept[i]=compute_intercept_estimator(x_vals,y_train[i,:])
    y_hat=slope*x_vals+intercept
    y_test=np.zeros(shape=(500,len(x_vals)))
    for i in range(500):
        y_test[i]=sample_linear_model(x_vals,2,0.5,1.5)
    error=np.zeros(shape=(500,1))
    for i in range(500):
        error[i]=calculate_prediction_error(y_test[i],y_hat[i,:])
    error=int(sum(error))
    return error/500

