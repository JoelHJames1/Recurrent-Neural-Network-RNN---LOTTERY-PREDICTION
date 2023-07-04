import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load the data
df = pd.read_excel('LeidsaQuinielaPale.xlsx', usecols=[0], header=None)
df[0] = df[0].apply(lambda x: [int(i) for i in str(x).split(',') if i.strip()])
df = df[df[0].apply(len) == 3]

# Convert list of lists into DataFrame
df = pd.DataFrame(df[0].to_list(), columns=['num1', 'num2', 'num3'])

# Reverse the order of the data
df = df.iloc[::-1].reset_index(drop=True)

# Normalize the data
scaler = MinMaxScaler().fit(df.values)
transformed_df = pd.DataFrame(scaler.transform(df.values), index=df.index)

# Prepare data for LSTM
window_length = 7  # length of your sequence
number_of_features = df.values.shape[1]  # number of features

X = np.empty([len(df) - window_length, window_length, number_of_features])
y = np.empty([len(df) - window_length, number_of_features])

for i in range(len(df) - 1, window_length - 1, -1):
    X[len(df) - 1 - i] = transformed_df.iloc[i - window_length + 1: i + 1]
    y[len(df) - 1 - i] = transformed_df.iloc[i - window_length]

# Define LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(240, return_sequences=True), 
                        input_shape=(window_length, number_of_features)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(480, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(240, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(number_of_features))

# Compile and train model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])

# Define the ModelCheckpoint and EarlyStopping callbacks
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(x=X, y=y, batch_size=100, epochs=300, verbose=2, 
          validation_split=0.2, callbacks=[checkpoint, early_stopping])

# Load the best model
model.load_weights('best_model.hdf5')

# Predict the next numbers
scaled_to_predict = transformed_df.head(window_length).values.reshape(1, window_length, number_of_features)
y_pred = model.predict(scaled_to_predict)

# Rescale the predicted values
predicted_numbers = scaler.inverse_transform(y_pred).astype(int)[0]
print("Predicted numbers: ", predicted_numbers)

# Get the actual numbers in the last lottery game
actual = df.head(1)
actual = np.array(actual)
print("The actual numbers in the last lottery game were:", actual[0])