import torch
# torch.manual_seed(410)


class LinearRegressionDataset():
    def __init__(self, n_samples=100, n_features=1, noise=1.0):
        super().__init__()  
        self.x = torch.randn(n_samples, n_features) * 10 
        w = torch.randn(n_features, 1) 
        self.y = torch.matmul(self.x, w) + 2 + torch.randn(n_samples, 1) * noise
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __get_all_data__(self):
        return self.x,self.y


