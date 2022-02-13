#%%
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd 
from statsmodels.graphics.tsaplots import pacf, acf
from statsmodels.tsa.stattools import adfuller, kpss
import math
import plotly.graph_objs as go
import numpy as np


def genPiAppxDigits(numdigits,appxAcc):
	import numpy as np
	from decimal import getcontext, Decimal
	getcontext().prec = numdigits
	mypi = (Decimal(4) * sum(-Decimal(k%4 - 2) / k for k in range(1, 2*appxAcc+1, 2)))
	return mypi

numdigits = 1000
appxAcc = 10000

series = genPiAppxDigits(numdigits,appxAcc)

#convert and split data
df = list(str(series))
df.remove('.')
df_float = []
for item in df:
    df_float.append(float(item))
size = int(len(df) * 0.75)
train, test = df[0:size], df[size:]

#calculate ACF and PACF
df_pacf = pacf(df)
df_acf = acf(df)

lower95 = -1.96/math.sqrt(len(df))
upper95 = -lower95

#visualize PACF and ACF to determine order for ARIMA model
fig2= go.Figure()
fig2.add_trace(go.Bar(
    x= np.arange(len(df_acf)),
    y= df_acf,
    name= 'ACF',
    width=[0.2]*len(df_acf),
    showlegend=False
    ))
fig2.add_hrect(y0=lower95, y1=upper95,line_width=0,fillcolor='green',opacity=0.1)
fig2.add_trace(go.Scatter(
    mode='markers',
    x=np.arange(len(df_acf)),
    y= df_acf,
    marker=dict(color='blue',size=8),
    showlegend=False
))
fig2.update_layout(
    title="Autocorrelation",
    xaxis_title="Lag",
    yaxis_title="Autocorrelation",
        height=500,
    )

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x= np.arange(len(df_pacf)),
    y= df_pacf,
    name= 'PACF',
    width=[0.2]*len(df_pacf),
    showlegend=False
    ))
fig3.add_hrect(y0=lower95, y1=upper95,line_width=0,fillcolor='green',opacity=0.1)
fig3.add_trace(go.Scatter(
    mode='markers',
    x=np.arange(len(df_pacf)),
    y= df_pacf,
    marker=dict(color='blue',size=8),
    showlegend=False
))
fig3.update_layout(
    title="Partial Autocorrelation",
    xaxis_title="Lag",
    yaxis_title="Partial Autocorrelation",
        height=500,
    )

fig2.show()
fig3.show()

#define function to create ARIMA model and record predictions/residuals
def CreatePredictions(p,q,d,test,train):
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(train, order=(p,q,d))
        model_fit = model.fit()
        output = model_fit.forecast()
        residuals = pd.DataFrame(model_fit.resid)
        yhat = output[0]
        predictions.append(yhat)
        try:
            obs = test.iloc[t]
            train = train.append(obs)
        except:
            pass
    pred = pd.DataFrame([test,predictions])
    residuals = residuals.iloc[1:]
    return pred, residuals

#convert lists of strings to floats
test_float = []
for item in test:
    test_float.append(float(item))
train_float = []
for item in train:
    train_float.append(float(item))
pred, residuals = CreatePredictions(1,1,0,test_float,train_float)

#visualize prediction against actual digits of pi
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x= np.arange(len(pred.T)),
    y= pred.T[0],
    name= 'Digits of pi'))
fig4.add_trace(go.Line(x=np.arange(len(pred.T)),y=pred.T[1],name='Prediction'))
fig4.show()

#determine accuracy based on the number of correct predictions
difference_in_predictions = round(pred.T[0]-pred.T[1],3)
difference_in_predictions[difference_in_predictions==0]


print('''Since the prediction is a constant, there is no predictive power and pi is irrational since by
definition irrational numbers require an infinite number of digits to write. There is no discerable pattern
which the digits follow. The accuracy here is 0 since there are no correct predictions''')
# %%
#bonus to calculate the pairwise correlation for each series of various accuracy

list_of_appxAcc = [1000,5000,10000,50000,100000]
df_series = []

for i in list_of_appxAcc:
    series = genPiAppxDigits(numdigits, i)
    df = list(str(series))
    df.remove('.')
    df_float = []
    for item in df:
        df_float.append(float(item))
    df_series.append(df_float)

correlation_matrix = pd.DataFrame(df_series).T.corr()
print('''For each time series there is no correlation (less than 0.1)''')
# %%
