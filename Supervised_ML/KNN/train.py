import sklearn
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from KNN import KNN


iris = datasets.load_iris()

X = torch.tensor(iris['data'])
y = torch.tensor(iris['target'])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle= True,random_state=42)



model = KNN()

model.fit(X_train,y_train)
y_pred = torch.tensor([model.predict(X_test[row,:]) for row in range(X_test.shape[0])])

acc = (y_pred == y_test).float().mean()

print(f'Accuracy:',acc*100)


fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)


scatter_actual = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='viridis', marker='o', edgecolors='k')
axes[0].set_title('Actual Classes')
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')
fig.colorbar(scatter_actual, ax=axes[0], label='Classes')


scatter_pred = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=50, cmap='cool', marker='^', edgecolors='k')
axes[1].set_title('Predicted Classes')
axes[1].set_xlabel('Sepal Length')
fig.colorbar(scatter_pred, ax=axes[1], label='Classes')

plt.tight_layout()
plt.show()




