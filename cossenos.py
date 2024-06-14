import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

# Suprimindo os warnings
import warnings
warnings.filterwarnings("ignore")

def calcular_similaridade(X_test):
    # Inicializar a matriz de similaridade
    n = len(X_test)
    M_test = np.zeros((n, n))

    # Calcular similaridade do cosseno entre todos os pares de pontos
    for i in range(n):
        for j in range(i, n):
            sim = 1 - cosine(X_test.iloc[i], X_test.iloc[j])
            M_test[i][j] = sim
            M_test[j][i] = sim  # Como a matriz é simétrica, preenchemos ambos os lados

    return M_test

def calcular_resultados(X_test, y_test, valor_especifico):
    M_test = calcular_similaridade(X_test)

    # Normalizar os valores reais
    y_test_norm = (y_test - y_test.min()) / (y_test.max() - y_test.min())

    # Calculando a similaridade média ponderada pelos valores reais normalizados
    sim_ponderada = np.mean(np.multiply(M_test, np.expand_dims(y_test_norm.values, axis=1)), axis=1)

    # Calculando o valor estimado como a média ponderada dos valores reais
    resultado_estimado = np.mean(sim_ponderada)

    # Calculando o RMSE
    rmse = np.sqrt(mean_squared_error(y_test, sim_ponderada * (y_test.max() - y_test.min()) + y_test.min()))

    # Dividindo os valores preditos e reais com o valor específico
    resultado_estimado /= valor_especifico
    resultado_real = y_test.mean() / valor_especifico

    # Criando um DataFrame com os valores preditos e reais
    resultados = pd.DataFrame({'Valor Estimado': sim_ponderada / valor_especifico, 'Valor Real': y_test / valor_especifico})

    return resultados, resultado_estimado, resultado_real, rmse

# Valores específicos para cada base de dados
valores_especificos = [5, 7, 5, 5, 9, 6, 5, 5, 7, 8, 30, 6, 13, 5]

caminhos_arquivos = [
    "/content/database_dir_adm.csv",
    "/content/database_dir_cons.csv",
    "/content/database_dir_hum.csv",
    "/content/database_dir_penal.csv",
    "/content/database_dir_proc.csv",
    "/content/database_etica.csv",
    "/content/database_fisica.csv",
    "/content/database_geopolitica.csv",
    "/content/database_informatica.csv",
    "/content/database_ingles.csv",
    "/content/database_legislacao.csv",
    "/content/database_matematica.csv",
    "/content/database_portugues.csv",
    "/content/database_redacao.csv"
]

for i, caminho_arquivo in enumerate(caminhos_arquivos):
    # Carregando a base de dados
    data = pd.read_csv(caminho_arquivo)

    # Preparando os dados
    df_dados = pd.DataFrame(data)
    X = df_dados.iloc[:, :-1]
    y = df_dados.iloc[:, -1]

    # Divisão dos dados em conjuntos de treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculando os resultados
    resultados, resultado_estimado, resultado_real, rmse = calcular_resultados(X_test, y_test, valores_especificos[i])

    # Salvando os resultados em um arquivo XLSX separado para cada base de dados
    nome_arquivo_resultados = f"/content/resultados_{caminho_arquivo.split('/')[-1].replace('.csv', '.xlsx')}"
    resultados.to_excel(nome_arquivo_resultados, index=False)

    print(f"Resultados salvos em {nome_arquivo_resultados}:")
    print(resultados.head(21))

    print("Resultado Estimado:", resultado_estimado)
    print("Resultado Real:", resultado_real)
    print("RMSE:", rmse)
    print("\n\n")