import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
df = pd.read_excel('LeidsaQuinielaPale.xlsx', usecols=[0], header=None)

# Process the data
df[0] = df[0].apply(lambda x: [int(i) for i in str(x).split(',') if i.strip()])

# Filter rows that do not have exactly 3 numbers
df = df[df[0].apply(len) == 3]

# We will predict the next number for each of the three positions separately
first_numbers = [x[0] for x in df[0]]
second_numbers = [x[1] for x in df[0]]
third_numbers = [x[2] for x in df[0]]

# Prepare data for linear regression
X = np.array(range(len(first_numbers))).reshape(-1, 1)
y1 = np.array(first_numbers).reshape(-1, 1)
y2 = np.array(second_numbers).reshape(-1, 1)
y3 = np.array(third_numbers).reshape(-1, 1)

# Train the models
model1 = LinearRegression().fit(X, y1)
model2 = LinearRegression().fit(X, y2)
model3 = LinearRegression().fit(X, y3)

# Predict the next numbers
next_index = np.array([len(first_numbers)]).reshape(-1, 1)
predicted_numbers = [model1.predict(next_index), model2.predict(next_index), model3.predict(next_index)]
predicted_numbers = [int(round(num[0][0])) for num in predicted_numbers]

print("Predicted numbers: ", predicted_numbers)
