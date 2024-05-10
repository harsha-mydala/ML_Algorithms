import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from Dataset import LinearRegressionDataset

dataset = LinearRegressionDataset(n_samples=100,n_features=1)
X,y =  dataset.x,dataset.y

X_test,X_train,y_test,y_train = train_test_split(X, y, test_size=0, random_state=42)


fig = plt.figure(figsize=(8,6))
plt.plot(X,y)
plt.show()


