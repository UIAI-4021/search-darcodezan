# import libraries
import time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# regression class
class Regression:
    
    def __init__(self, number_of_iterations, learning_rate):
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        
    def h(self, x):
        return self.theta.dot(x)

    def J(self):
        res = 0
        for i in range(self.n):
            res += (self.h(self.X[i])-self.y[i])**2
        res /= 2*self.n
        return res

    def gradiant(self):
        nabla = np.array(self.n)
        Y = np.zeros(self.m)
        for i in range(self.m):
            Y[i] = (self.h(self.X[i])-self.y[i])/self.m
        nabla = Y.dot(self.X)
        if (np.linalg.norm(nabla) != 0):
            nabla /= np.linalg.norm(nabla)
        return nabla
        
    def gradiant_decent(self):
        for k in range(self.number_of_iterations):
            nabla = self.gradiant()
            for j in range(self.n):
                self.theta[j] -= self.learning_rate*nabla[j]
            
    def run(self, input_dataset, output_dataset):
        self.X = np.array(input_dataset)
        self.y = np.array(output_dataset)
        
        # m(number of data), n(number of features or dimentions)
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.theta = np.zeros(self.n)
        self.gradiant_decent()

    def test(self, X_test, Y_test):
        X_test = np.array(X_test)
        y_hat = np.zeros(len(X_test))
        for i in range(len(X_test)):
            y_hat[i] = self.h(X_test[i])
        
        with open("1-UIAI4021-PR1-Q2.txt", "a") as f:
            f.write("Logs:\n")
            f.write(f"MSE: {(mean_squared_error(y_hat, Y_test))}\n")
            f.write(f"RMSE: {(mean_squared_error(y_hat, Y_test))**0.5}\n")
            f.write(f"MAE: {mean_absolute_error(y_hat, Y_test)}\n")
            f.write(f"R2: {r2_score(y_hat, Y_test)}\n")

# read from csv file
df = pd.read_csv("~/downloads/Flight_Price_Dataset_Q2.csv")

# some mappings for non-numeric data
departure_time_mapping = {
    "Early_Morning": 1,
    "Morning": 3,
    "Afternoon": 4,
    "Night": 2, 
    "Late_Night": 0
}
stops_mapping = {
    "zero": 2,
    "one": 1,
    "two_or_more": 0
}
class_mapping = {
    "Economy": 0,
    "Business": 1
}
df["departure_time"] = df["departure_time"].map(departure_time_mapping)
df["stops"] = df["stops"].map(stops_mapping)
df["arrival_time"] = df["arrival_time"].map(departure_time_mapping)
df["class"] = df["class"].map(class_mapping)

# remove nan data and normalize it
df = df.dropna()
df = df.reset_index(drop=True)
sclr = MinMaxScaler()
df_norm = pd.DataFrame(sclr.fit_transform(df), columns=df.columns)

# split data to test and train
Y = df_norm["price"]
X = df_norm.drop("price", axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# add fake feature for regression
X_test["fake_feature"] = 1
X_train["fake_feature"] = 1

learning_rate = 0.01
number_of_iterations = 500

start_time = time.time()
reg = Regression(number_of_iterations, learning_rate)
reg.run(X_train, Y_train)

# write to file
with open("1-UIAI4021-PR1-Q2.txt", "w") as f:
    f.write("PRICE = ")
    splt = " + "
    for i in range(reg.n):
        col = df.columns[i]
        if i == (reg.n)-1:
            splt = ""
            col = "bias"
        f.write(f"{reg.theta[i]} * {col}")
        f.write(splt)
    f.write("\n")
    f.write(f"Training Time: {round(time.time() - start_time, 6)}s\n\n")

# results of our regression model
reg.test(X_test, Y_test)
