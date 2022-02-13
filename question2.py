#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from tsmoothie.smoother import ConvolutionSmoother
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

#read data and format into a dataframe
x = pd.read_csv('data/xvalsSine.csv')
sine = pd.read_csv('data/cleanSine.csv')

data = pd.DataFrame([x.iloc[:,0],sine.iloc[:,0]],index=['x','sin(x)']).T

#split data into test and train sets randomly with 70/30 train/test split
X_train, X_test, y_train, y_test = train_test_split(data['x'], data['sin(x)'],
                                                    shuffle = True, 
                                                    test_size=0.3, 
                                                    random_state=1)

#use a polynomial order three to approximate well sine curve
polynomial_features= PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train.values.reshape(-1,1))
X_test_poly = polynomial_features.fit_transform(X_test.values.reshape(-1,1))

#calculate OLS model 
model = sm.OLS(y_train.values, X_train_poly).fit()
model.summary()

print('''The OLS R-squared is equal to '''+str(model.rsquared))
print('''The OLS MSE is '''+str(model.mse_model))
y_predicted = model.predict(X_test_poly) 

#bonus use a stochastic gradient descent 
n_iter=100
sgd = SGDRegressor(max_iter=n_iter)
sgd.fit(X_train.values.reshape(-1,1), y_train)
y_predicted_sgd=sgd.predict(X_test.values.reshape(-1,1))
sgd_mse = mean_squared_error(y_test,y_predicted_sgd)
print('''The SGD MSE is '''+str(sgd_mse))

#start part ii
noisy_sine = pd.read_csv('data/noisySine.csv')
noisy_sine.columns = ['noisy sin(x)']

#filter noisy data 
smoother = ConvolutionSmoother(window_len=50, window_type='ones') #can alter window length for further improvement
smoother.smooth(noisy_sine)

filtered_noisy_sine = pd.Series(smoother.smooth_data[0],name='Filtered_sin(x)')

data = data.join(filtered_noisy_sine)

#calculate sum of squared errors
errors = data['sin(x)']-data['Filtered_sin(x)']
sq_errors = errors**2
sse = sum(sq_errors)

print('''The sum of squared errors between the sin(x) dataset and smoothed "noisy" sin(x) dataset is '''+str(round(sse,3)))   

#bonus predict 10 samples from a Kalman filter

# initial parameters
x_sig = x.var().values[0]
sin_sig = noisy_sine.var().values[0]
mu = noisy_sine.mean().values[0]
sig = sin_sig**2


#initial parameters

x_k = np.asarray([x.iloc[0,0],noisy_sine.iloc[0,0]]) #first estimate
Q = np.asarray([[0.001,0],[0,0.001]]) #Estimate error covariance (took a guess here)
A = np.asarray([[1,0],[0,1]]) #Transition matrix
R = np.asarray([[0,0],[0,np.mean(noisy_sine.values-sine.values)]]) #Measurement error
H = np.asarray([[1,0],[0,1]]) #Observation matrix
P = np.asarray([[0,0],[0,0]]) #Error matrix

estimation = []

for k in range(10):
    
    z_k = np.asarray([x.iloc[k,0], noisy_sine.iloc[k,0]])
    
    x_k = A.dot(x_k) #predict estimate
    P = (A.dot(P)).dot(A.T) + Q #predict error covariance
    
    K = (P.dot(H.T)).dot(np.linalg.inv((H.dot(P).dot(H.T)) + R)) #update Kalman Gain
    x_k = x_k + K.dot((z_k - H.dot(x_k))) #update estimate
    
    P = (np.identity(2) - K.dot(H)).dot(P) #update error covariance
    
    estimation.append((x_k[0], x_k[1])) #append the estimations
# %%
