import numpy as np
import matplotlib.pyplot as plt
from GaussianClassifiers import GaussianClassifier
data = np.loadtxt("EMGsDataset.csv",delimiter=',')

# data = data.T
X,y = data[:-1,:], data[-1:,:]

X_treino = X[:,:int(.8*X.shape[1])]
y_treino = y[:,:int(.8*X.shape[1])]

x_teste = X[:,int(.8*X.shape[1]):]
y_teste = y[:,int(.8*X.shape[1]):]

LAMBDAS = [0, 0.25, 0.5, 0.75, 1]

COLORS = ['red', 'cyan', 'blue', 'purple', 'pink']
predictions = []

for lmb in LAMBDAS:
    # print(f"  - Treinando com λ = {lmb}")
    gc = GaussianClassifier(X_treino, y_treino, lmb)
    gc.fit()
    y_pred = gc.predict(x_teste)
    predictions.append(y_pred)

#print(predictions[0])

for i, lmb in enumerate(LAMBDAS):
    ax = axes[i]
    y_p = predictions[i]
    point_colors = [COLORS[int(label) % len(COLORS)] for label in y_p]

    ax.scatter(x_teste[0, :], x_teste[1, :], c=point_colors, cmap='jet', marker='.')
    ax.set_title(f'Predição com λ = {lmb}')
     #ax.set_xlabel('Feature 1')
     # Coloca o rótulo Y apenas no primeiro gráfico para não poluir
    if i == 0:
      ax.set_ylabel('Feature 2')

 #plt.figure(figsize=(12, 7))
    
#     # Plot dos dados de teste reais
#     # plt.subplot(1, 2, 1)
#     # plt.scatter(x_teste[0, :], x_teste[1, :], c=y_teste.flatten(), cmap='jet', marker='.')
#     # plt.title('Dados de Teste Reais')
#     # plt.xlabel('Feature 1')
#     # plt.ylabel('Feature 2')
    
#     # # Plot dos dados de teste com a predição do classificador
    
#     # plt.subplot(1, 2, 2)
#     # plt.scatter(x_teste[0, :], x_teste[1, :], c=y_pred, cmap='jet', marker='.')
#     # plt.title('Predições do Classificador')
#     # plt.xlabel('Feature 1')
    
plt.tight_layout()
plt.show()

bp = 1