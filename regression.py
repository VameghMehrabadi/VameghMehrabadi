import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# READ OUR DATD TABLE
df = pd.read_csv('co2.csv')

#ONLY GET ONE FEATURE AS AN INPUT FOR REGRESSION MODEL
x = df.drop("out1",axis=1)
x = x.drop("fuelcomb",axis=1)
x = x.drop("cylandr",axis=1)
y = df.out1

#TRAIN AND TEXT OUR INPUT AND OUTPUT
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#CALLING LINEAR_REGRESSION MODEL
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
output_bot = model.predict(X_test)

#TEST OUR LINEAR REGRESSION WITHE New_arr
New_arr = np.array([1.36])
prediction = model.predict(New_arr)
print(prediction)
