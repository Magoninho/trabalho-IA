import numpy as np


class GaussianClassifier:
    def __init__(self,X_train, y_train, lmb):
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p,self.N = X_train.shape
        self.X =  [X_train[:,y_train[0,:]==i] for i in self.classes]
        self.n =  [Xi.shape[1] for Xi in self.X]
        self.Sigma = [None]*self.C
        self.mu = [None]*self.C
        self.Sigma_inv = [None]*self.C
        self.Sigma_det = [None]*self.C
        self.Py = [None]*self.C #probabilidade a priori
        self.predicao = [None]*self.C
        self.x_train_full = X_train
        self.lmb = lmb
    
    def fit(self):
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i],axis=1).reshape(self.p,1)
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])
            self.Py[i] = self.n[i]/self.N
            if self.lmb == 0:
                self.Sigma[i] = np.cov(self.X[i]) # matriz de covariancia
            elif self.lmb == 1:
                self.Sigma_agregada = np.cov(self.x_train_full) # matriz de covariancia agregada
            else:
                self.Sigma[i] = (1 - self.lmb * np.cov(self.X[i]) + self.lmb * np.cov(self.x_train_full))
            
    def predict(self,X_teste):
        epsilon = 1e-10  # small constant to prevent log(0)
        
        for i in range(self.C):
            mahalanobis_q = (X_teste-self.mu[i]).T@self.Sigma_inv[i]@(X_teste-self.mu[i])
            self.predicao[i] = np.log(self.Py[i] + epsilon) - 1/2*np.log(self.Sigma_det[i] + epsilon) - 1/2*mahalanobis_q[0,0]
        bp=1