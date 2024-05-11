import torch 
torch.manual_seed(42)

class LinearRegression():

    def __init__(self,lr=0.01,n_iter=150):

        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.loss = torch.tensor(())

    def fit(self,X,y):

        n_samples,n_features = X.shape
        self.weights = torch.randn(1+n_features,1)
        X = torch.cat((X,torch.ones(n_samples,1)),axis = 1)

        for _ in range(self.n_iter):
            y_pred = torch.matmul(X,self.weights)
            self.update_loss(y_pred,y)
            dw = 1/n_samples*((torch.matmul(X.T,(y_pred-y))))
            self.weights = self.weights-self.lr*dw

    def predict(self,X):

        X = torch.cat((X,torch.ones(X.shape[0],1)),axis = 1)
        y_pred =   torch.matmul(X,self.weights)

        return y_pred

    def update_loss(self,y_pred,y):

        cur_loss = torch.mean(torch.pow((y-y_pred),2))
        self.loss = torch.cat((self.loss,(cur_loss.unsqueeze(0))))


