import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# Função de predição (não precisa de alteração)
def predicao(X, beta):
    return X @ beta

# --- TAREFA 1: Carregar e visualizar os dados ---

data = np.loadtxt("aerogerador.dat", delimiter='\t')
#controle_figura = True    

fig = plt.figure(1)
ax = fig.add_subplot()
ax.scatter(data[:, 0], data[:, 1], edgecolor='k') #, alpha=0.6
ax.set_title("Todo o conjunto de dados.")
ax.set_xlabel("Velocidade do vento")
ax.set_ylabel("Potência gerada pelo aerogerador")
plt.show()

# --- TAREFA 2: Organização dos dados ---
#Dados sem a coluna de 1s (X_)
X_ = data[:, :-1]
y = data[:, -1:]
N, p = X_.shape

# Adiciona a coluna de 1s para o intercepto
X = np.hstack((
    np.ones((N, 1)), X_
))

# --- TAREFA 4 e 5: Preparação para a Simulação de Monte Carlo ---
rodadas = 500
lambdas = [0.25, 0.5, 0.75, 1.0]

# Listas para armazenar o RSS (SSE) de cada modelo em cada rodada
rss_media_lista = []
rss_mqo_lista = []
# Usamos um dicionário para os modelos regularizados, a chave é o lambda
rss_Tikhonov_map = {l: [] for l in lambdas} # Faria diferença 4 vetores ?

# --- Início do loop da Simulação de Monte Carlo (AMOSTRAGEM ALEATORIA) ---
for r in range(rodadas):
    # Embaralhar o conjunto de dados
    idx = np.random.permutation(N)
    Xr_ = X_[idx,:] #conferir
    Xr = X[idx, :]
    yr = y[idx, :]

    # Particionamento do conjunto de dados (80% treino, 20% teste)
    X_treino_ = Xr_[:int(N*.8),:]
    X_treino = Xr[:int(N*.8), :]
    y_treino = yr[:int(N*.8), :]
    
    X_teste_ = Xr_[int(N*.8):,:]
    X_teste = Xr[int(N*.8):, :] 
    y_teste = yr[int(N*.8):, :]
    # --- TAREFA 3: Treinamento e Avaliação dos Modelos ---

    # 1. Modelo baseado na Média
    beta_hat_media = np.array([
        [np.mean(y_treino)],
        [0]])
    
    #Predição
    y_pred_media = predicao(X_teste, beta_hat_media)
    #Medida de desempenho
    rss_media = np.sum((y_teste - y_pred_media)**2)
    rss_media_lista.append(rss_media)

    # 2. Modelo MQO Tradicional
    beta_hat_mqo = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino
    #Predição
    y_pred_mqo = predicao(X_teste, beta_hat_mqo)
    #Medida de desempenho
    rss_mqo = np.sum((y_teste - y_pred_mqo)**2)
    rss_mqo_lista.append(rss_mqo)
    
    # 3. Modelo MQO Regularizado (Tikhonov)
    # Matriz de identidade (p+1 x p+1) para a regularização
    I = np.identity(p + 1)
    I[0, 0] = 0

    for l in lambdas:
        # Fórmula do MQO Regularizado
        beta_hat_Tikhonov = np.linalg.pinv(X_treino.T @ X_treino + l * I) @ X_treino.T @ y_treino
        y_pred_Tikhonov = predicao(X_teste, beta_hat_Tikhonov)
        rss_Tikhonov = np.sum((y_teste - y_pred_Tikhonov)**2)
        rss_Tikhonov_map[l].append(rss_Tikhonov)

# --- Apresentação dos resultados ---
print("\n--- Resultados Finais da Simulação de Monte Carlo (500 rodadas) ---")
print("Métrica de Desempenho: RSS (Soma dos Desvios Quadráticos)\n")

# Dicionário para armazenar os resultados finais
resultados = {
    "Média da variável dependente": rss_media_lista,
    "MQO tradicional": rss_mqo_lista,
    "MQO regularizado (λ=0.25)": rss_Tikhonov_map[0.25],
    "MQO regularizado (λ=0.50)": rss_Tikhonov_map[0.50],
    "MQO regularizado (λ=0.75)": rss_Tikhonov_map[0.75],
    "MQO regularizado (λ=1.00)": rss_Tikhonov_map[1.00],
}

# Imprime o cabeçalho da tabela
print(f"{'Modelo':<30} | {'Média':>15} | {'Desvio-Padrão':>15} | {'Maior Valor':>15} | {'Menor Valor':>15}")
print("-" * 98)

# Calcula e imprime as estatísticas para cada modelo
for nome, lista_rss in resultados.items():
    media = np.mean(lista_rss)
    desvio_padrao = np.std(lista_rss)
    maior_valor = np.max(lista_rss)
    menor_valor = np.min(lista_rss)
    print(f"{nome:<30} | {media:15.4f} | {desvio_padrao:15.4f} | {maior_valor:15.4f} | {menor_valor:15.4f}")