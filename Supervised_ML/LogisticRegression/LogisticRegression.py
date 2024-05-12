import torch


class LogisticRegression():

    def __init__(self, lr=0.001, n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters

    def sigmoid(self,x):

        return 1/(1+torch.exp(-x))

    def predictjithin(self, X, y):

        n_samples,n_features = X.shape[0], X.shape[1]

        self.weights = torch.randn(n_features+1)
        X = torch.cat(X,torch.ones(n_samples,1),axis = 1)
