import numpy as np


class GaussianClassifier:
    def __init__(self,X_train, y_train, lmb):
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        y_train_flat = y_train.flatten() # alteração chat
        self.X =  [X_train[:,y_train_flat==i] for i in self.classes]
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

        Sigma_agregada = np.cov(self.x_train_full)
        for i in range(self.C):

            self.mu[i] = np.mean(self.X[i],axis=1).reshape(self.p,1)
            # self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            # self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])
            self.Py[i] = self.n[i]/self.N
            if self.lmb == 0:
                self.Sigma[i] = np.cov(self.X[i]) # matriz de covariancia
            elif self.lmb == 1:
                self.Sigma[i] = Sigma_agregada  #matriz de covariancia agregada
            
            else:
                self.Sigma[i] = (1 - self.lmb) * np.cov(self.X[i]) + self.lmb * Sigma_agregada            
    
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])

    def predict(self, X_teste):
        scores_ponto_atual = []
        N_teste = X_teste.shape[1]
        predicoes_finais = []
        epsilon = 1e-10  # small constant to prevent log(0)
        for j in range(N_teste):
            # Extrai a amostra atual como um vetor coluna (shape p, 1)
            ponto_atual = X_teste[:, [j]]
             
            # Laço interno: calcula o score de cada classe para a amostra atual
            for i in range(self.C):

                # Sua lógica original, que funciona perfeitamente para um ponto
                mahalanobis_q = (ponto_atual - self.mu[i]).T @ self.Sigma_inv[i] @ (ponto_atual - self.mu[i])
                
                # O resultado é uma matriz (1,1), então [0,0] extrai o valor
                score = np.log(self.Py[i] + epsilon) - (1/2)*np.log(self.Sigma_det[i] + epsilon) - (1/2)*mahalanobis_q[0,0]
                scores_ponto_atual.append(score)
            
            # Encontra a classe com o maior score para o ponto atual
            classe_predita = np.argmax(scores_ponto_atual)
            ()
            predicoes_finais.append(classe_predita)
        
        return np.array(predicoes_finais)
        
        
        
        bp=1