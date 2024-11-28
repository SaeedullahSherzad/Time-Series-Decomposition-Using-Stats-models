import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose



file_path = 'AirPassengers.csv'  
data = pd.read_csv(file_path)


data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)


decomposition = seasonal_decompose(data['#Passengers'], model='additive', period=12)


trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


plt.figure(figsize=(12, 10))


plt.subplot(4, 1, 1)
plt.plot(data['#Passengers'], label='Observed', color='blue')
plt.title('Observed Time Series')
plt.legend(loc='best')


plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend', color='orange')
plt.title('Trend Component')
plt.legend(loc='best')


plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal', color='green')
plt.title('Seasonal Component')
plt.legend(loc='best')