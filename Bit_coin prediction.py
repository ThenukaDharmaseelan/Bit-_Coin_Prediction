#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[2]:


data = pd.read_csv('/Users/thenuka/Downloads/Bitcoin-price-Prediction-using-LSTM-master/bitcoin_ticker.csv')


# In[3]:


data.head()


# In[4]:


data['rpt_key'].value_counts()


# In[5]:


df = data.loc[(data['rpt_key']== 'btc_usd')]


# In[6]:


df.head()


# In[7]:


#Convert datetime_id to data type and filter dates greater than 2017-06-28 00:00:00
df = df.reset_index(drop = True)
df['datetime'] = pd.to_datetime(df['datetime_id'])
df = df.loc[df['datetime'] > pd.to_datetime('2017-06-28 00:00:00')]


# In[8]:


df = df[['datetime', 'last', 'diff_24h', 'diff_per_24h', 'bid', 'ask', 'low', 'high', 'volume']]


# In[9]:


df.head()


# In[10]:


#we require only the last value, so we subset that and convert it to numpy array
df = df[['last']]


# In[11]:


dataset = df.values
dataset = dataset.astype('float32')


# In[12]:


dataset


# In[13]:


#Neural networks are sensitive to input data, especiallly when we are using activation functions like sigmoid or tanh activation functions are used. ISo we rescale our data to the range of 0-to-1, using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)


# In[14]:


dataset


# In[15]:


train_size = int(len(dataset)*0.67)
test_size = len(dataset)-train_size
train, test = dataset[0:train_size, :],dataset[train_size:len(dataset), :]
print(len(train),len(test))


# In[16]:


#Now let us define the function called create_dataset, which take two inputs,
#Dataset - numpy array that we want to convert into a dataset
#look_back - number of previous time steps to use as input variables to predict the next time period
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[17]:


look_back =10
trainX, trainY = create_dataset(train, look_back=look_back)
testX,testY = create_dataset(test,look_back=look_back)


# In[18]:


trainX


# In[19]:


trainY


# In[20]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))


# In[21]:


#Build Model


# In[22]:


model = Sequential()
model.add(LSTM(4, input_shape=(1,look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error',optimizer='adam')
model.fit(trainX, trainY,epochs=100,batch_size=256,verbose=2)


# In[23]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[24]:


#We have to invert the predictions before calculating error to so that reports will be in same units as our original data
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[25]:


trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[26]:


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict


# In[27]:


# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[look_back:len(testPredict) + look_back, :] = testPredict


# In[28]:


plt.plot(df['last'], label='Actual')
plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=df.index).close, label='Training')
plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=df.index).close, label='Testing')
plt.legend(loc='best')
plt.show()


# In[29]:


print(trainPredict)


# In[30]:


print(testPredict)


# In[ ]:




