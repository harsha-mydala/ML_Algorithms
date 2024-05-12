import torch

class KNN():

    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None
    
    def fit(self,X,y):

        self.X = X
        self.y = y

    def predict(self,x):

        return torch.argmax(torch.bincount(self.k_neignous(x)))
    
    def k_neignous(self,x):

        distances = self.distance(x)
        nearest_k = torch.argsort(distances)[:self.k]

        return self.y[nearest_k]

    def distance(self,x):

        return torch.sqrt(torch.sum(torch.square(self.X-x),axis = 1))






