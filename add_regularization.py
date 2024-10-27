# %%

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, Attention
from tensorflow.keras.models import Model
import yfinance as yf

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

m = yf.Ticker("MSFT")

# %%
msft = m.history(period='2y', interval = '1h')

# %%
import pandas as pd

# %%
msft = msft['Open']

# %%
import numpy as np

# %%
prices = []
up_down = []

num_hours = 200
#msft.shape[0] - num_hours - 1
for i in range(msft.shape[0] - num_hours - 1):
    prices.append(msft.iloc[i:i+num_hours].to_list())
    up_down.append(msft.iloc[i+num_hours] > msft.iloc[i+num_hours - 1])
up_down = [1  if i == np.True_ else 0 for i in up_down]



# %%

# %%
def fft_mapping(iprices: list) -> list:
    f = np.fft.fft(np.array(iprices))
    a = np.real(f)
    b = np.imag(f)
    return np.array([a,b])
prices = list(prices)

# %%
prices

# %%
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
#a, b = fft_mapping(prices[0])
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

print(len(a))
print(len(b))

# %%
input1 = Input(shape=(2,num_hours), name="Input")
dense1 = Dense(1024, activation="relu")(input1)
dense1 = Dense(1024, activation="relu")(dense1)
dense1 = Dense(1024, activation="relu")(dense1)
#flatten = Flatten()(dense1)
attention_output = Attention()([dense1,dense1])
flatten = Flatten()(attention_output)

#input2 = Input(shape=(num_hours, 1))
#l1 = tf.keras.layers.LSTM(128, return_sequences = True, activation = 'relu')(input2)
#l1 = Flatten()(l1)
#a = Attention()([l1,l1])
#flatten2 = Flatten()(a)
#flatten3 = tf.keras.layers.Concatenate()([flatten, flatten2])
output = Dense(1, activation="sigmoid")(flatten)
model = Model(inputs=input1,outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
#ValueError: Can't convert non-rectangular Python sequence to Tensor
#print(np.array(prices).shape)
#print(np.array(prices).reshape(-1,num_hours,1).shape)
# %%
#print(fftprices[0:2])
#print(up_down[0:2])
#model.fit(tf.convert_to_tensor([tf.convert_to_tensor(fftprices),tf.convert_to_tensor(np.array(prices).reshape(-1,num_hours,1))]), tf.convert_to_tensor(up_down), epochs = 100)

# %%
# Prepare data for model fitting
fftprices_reshaped = np.array(fftprices).reshape(-1, 2, num_hours)  # Adjust shape for input1
prices_reshaped = np.array(prices).reshape(-1, num_hours, 1)  # Adjust shape for input2
up_down_array = np.array(up_down).astype(np.float32)  # Convert to numpy array
print('up down')
print(up_down_array)
print("isnan")
print(np.isnan(prices_reshaped).any())
print("fftprices_reshaped shape:", fftprices_reshaped.shape)
print("prices_reshaped shape:", prices_reshaped.shape)
print("up_down_array shape:", up_down_array.shape)
print("NaN in fftprices_reshaped:", np.isnan(fftprices_reshaped).any())
print("NaN in prices_reshaped:", np.isnan(prices_reshaped).any())
print("NaN in up_down_array:", np.isnan(up_down_array).any())

# Fit the model
# %%
#model.fit([fftprices_reshaped, prices_reshaped], up_down_array, epochs=1, batch_size = 32)
#model.predict([fftprices_reshaped[0], prices_reshaped[0]])
history = model.fit(
    fftprices_reshaped, 
    up_down_array, 
    epochs=100, 
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

