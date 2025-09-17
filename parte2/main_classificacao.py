import numpy as np
import matplotlib.pyplot as plt
from GaussianClassifiers import GaussianClassifier
data = np.loadtxt("EMGsDataset.csv",delimiter=',')

# data = data.T
X,y = data[:-1,:], data[-1:,:]

X_treino = X[:,:int(.8*X.shape[1])]
y_treino = y[:,:int(.8*X.shape[1])]

x_teste = X[:,int(.8*X.shape[1]):].reshape(2,10000)
y_teste = y[:,int(.8*X.shape[1]):]

gc = GaussianClassifier(X_treino,y_treino)
gc.fit()
gc.predict(x_teste)

plt.figure(figsize=(10, 6))
plt.scatter(x_teste[0], x_teste[1], c=y_teste.flatten(), cmap='viridis')
plt.colorbar(label='Class')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Classifier Results')
plt.show()

bp = 1