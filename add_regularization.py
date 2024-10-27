# %%

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, Attention, LSTM, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# %%
def s(x,A,b1,phi1,c, b2,phi2):
    y = A*np.sin(b1*x + phi1)*np.sin(b2*x + phi2) +c
    return y

# Parameters
A = 1
b1 = 2 * np.pi / 5  # Frequency
phi1 = 0
b2 = 4
phi2 = 0
c = 0

# %%
# Generate x values
x = np.linspace(0, 20, 1000)

# Calculate y values
y = np.sin(2*x)

# %%

import yfinance as yf

import pandas as pd



import numpy as np

# %%
prices = []
up_down = []
num_hours = 200
def get_data(ticker:str):
    m = yf.Ticker(ticker)
    msft = m.history(period='2y', interval = '1h')

    msft = msft['Open']
    smoothing_factor = 7
    for i in range(msft.shape[0] - smoothing_factor + 1):
        a= sum(msft[i:i + smoothing_factor].to_list())/smoothing_factor
        #print(msft[i])
        msft[i] = a

    global prices
    global up_down
    for i in range(msft.shape[0] - num_hours - 1):
        prices.append(msft.iloc[i:i+num_hours].to_list())
        #print(msft.iloc[i:i+num_hours].to_list())
        #print(prices)
        up_down.append(msft.iloc[i+num_hours])#> msft.iloc[i+num_hours - 1])
    #up_down = [1  if i == np.True_ else 0 for i in up_down]

companies = ['msft', 'aapl', 'intc', 'nvda', 'amd', 'GOOG', "META"]
for c in companies:
    get_data(c)
num_ones = 0

print(prices[523])
for i in up_down:
    if i:
        num_ones += 1
print(num_ones/len(up_down))
def fft_mapping(iprices: list) -> list:
    f = np.fft.fft(np.array(iprices))
    a = np.real(f)
    b = np.imag(f)
    return np.array([a,b])
prices = list(prices)

stock_scaler = StandardScaler()

prices_scaled = stock_scaler.fit(prices)

#prices = prices_scaled.transform(prices)

plt.figure(figsize=(9, 9))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.subplot(311)
plt.title("Data")
plt.ylabel("Amplitude")
plt.plot(prices[0])
plt.subplot(312)
plt.title("Frequency and Amplitude")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")

fftprices = np.array(list(map(fft_mapping, prices)))
a = fftprices[0][0]
b = fftprices[0][1]
plt.plot(np.sqrt(a**2+b**2))
plt.subplot(313)
plt.title("Phase")
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.plot(np.arctan2(b,a))
plt.show()

input1 = Input(shape=(2,num_hours), name="Input")
dense1 = Dense(64, activation="relu")(input1)
#dense1 = Dropout(0.3)(dense1)
dense1 = Dense(64, activation="relu")(dense1)
dense1 = Dropout(0.3)(dense1)
dense1 = Dense(64, activation="relu")(dense1)
dense1 = Dense(64, activation="relu")(dense1)
dense1 = Dropout(0.4)(dense1)
dense1 = Dense(64, activation="relu", kernel_regularizer=L2(0.5))(dense1)

#flatten = Flatten()(dense1)
attention_output = Attention()([dense1,dense1])
flatten = Flatten()(attention_output)

#input2 = Input(shape=(num_hours, 1))
#l1 = tf.keras.layers.LSTM(128, return_sequences = True, activation = 'relu')(input2)
#a# = Attention()([l1,l1])
#flatten2 = Flatten()(a)
#flatten3 = tf.keras.layers.Concatenate()([flatten, flatten2])
output = Dense(1, activation="relu")(flatten)
model = Model(inputs=input1,outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss='mse', metrics=['accuracy'])

print(model.summary())

fftprices_reshaped = np.array(fftprices).reshape(-1, 2, num_hours)  # Adjust shape for input1
prices_reshaped = np.array(prices).reshape(-1, num_hours, 1)  # Adjust shape for input2
up_down_array = np.array(up_down).astype(np.float32)  # Convert to numpy array

history = model.fit(
    fftprices_reshaped,
    up_down_array, 
    epochs=100, 
    batch_size = 64,
    shuffle = True,
    validation_split=0.1,  # Split some data for validation
    verbose=1  # Verbose output for training
)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

val_accuracy = history.history['val_accuracy']
epochs = list(range(1, len(val_accuracy) + 1))
# Fit a line to (epochs, val_accuracy) and get the slope
slope, _ = np.polyfit(epochs, val_accuracy, 1)
print("Slope of the line of best fit for val_accuracy:", slope)
