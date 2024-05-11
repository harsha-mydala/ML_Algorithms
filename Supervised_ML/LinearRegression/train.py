import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from Dataset import LinearRegressionDataset
torch.manual_seed(420)

dataset = LinearRegressionDataset(n_samples=100,n_features=1)
X,y =  dataset.x,dataset.y

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=42)




model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Scatter plot of training data and line plot of predictions on the first subplot
axes[0].scatter(X, y, color='blue', label='Training Data')
axes[0].plot(X_test, y_pred, color='red', label='Predictions')
axes[0].set_title('Scatter Plot of Training Data with Predictions')
axes[0].set_xlabel('X values')
axes[0].set_ylabel('Y values')
axes[0].legend()
axes[0].grid(True)

# Line plot for model loss on the second subplot
axes[1].plot(model.loss, color='green', label='Loss Over Time')
axes[1].set_title('Model Loss During Training')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

# Show the plot
plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area.
plt.show()



# fig = plt.figure(figsize=(8,6))
# plt.scatter(X_train,y_train)
# plt.plot(X_test,y_pred)
# plt.plot
# plt.show()


# fig = plt.figure(figsize=(8,6))
# plt.plot(model.loss)
# plt.show()






