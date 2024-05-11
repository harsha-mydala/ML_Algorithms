import torch 

class LinearRegression():

    def __init__(self,lr,n_iter):

        self.lr = lr 
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):

        n_samples,n_features = X.shape
        self.weights = torch.randn(1+n_features)
        X = torch.cat((X,self.ones(n_samples,1)),axis = 1)

        for _ in range(self.n_iter):
            y_pred = torch.matmul(X,self.weights) #changed from dot to matmul as they are 2D tensors.

            dw = 1/n_samples*(torch.dot(X.T,(y_pred-y)))
            self.weights = self.weights-self.lr*dw

    def predict(self,X):

        X = torch.cat((X,torch.ones(X.shape[0],1)),axis = 1)
        y_pred =   torch.dot(X,self.weights)

        return y_pred

