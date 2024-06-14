import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Suprimindo os warnings
import warnings
warnings.filterwarnings("ignore")

# Lista de arquivos de base de dados
datasets = [
    "database_dir_adm.csv",
    "database_dir_cons.csv",
    "database_dir_hum.csv",
    "database_dir_penal.csv",
    "database_dir_proc.csv",
    "database_etica.csv",
    "database_fisica.csv",
    "database_geopolitica.csv",
    "database_informatica.csv",
    "database_ingles.csv",
    "database_legislacao.csv",
    "database_matematica.csv",
    "database_portugues.csv",
    "database_redacao.csv"
]

# Valores específicos para dividir o predito e o real para cada base de dados
valores_divisao = [5, 7, 5, 5, 9, 6, 5, 5, 7, 8, 30, 6, 13, 5]

for i, dataset in enumerate(datasets):
    # Carregar os dados
    df = pd.read_csv(f"/content/{dataset}")

    # Separar dados de entrada e saída
    X = df.iloc[:, :-1]  # Todas as colunas exceto a última
    y = df.iloc[:, -1]   # Última coluna

    # Dividir os dados em 80% treinamento e 20% teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    # Dividir o valor específico para essa base de dados
    valor_divisao = valores_divisao[i]

    # Inicialização do K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Inicializa um array vazio onde será gravado o erro quadrático médio
    RMSE = []

    # Melhor RMSE
    best_RMSE = float('inf')
    best_k = None
    best_weight = None

    # Loop sobre valores de k e tipos de pesos
    for k in range(1, 21):
        for weight in ["distance", "uniform"]:
            RMSE_k_weight = []  # Armazena o RMSE para cada fold
            for train_index, test_index in kf.split(X_train_val):
                X_train, X_val = X_train_val[train_index], X_train_val[test_index]
                y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[test_index]

                knr = KNeighborsRegressor(n_neighbors=k, weights=weight)
                knr.fit(X_train, y_train)
                y_pred = knr.predict(X_val)

                # Dividir os valores reais e preditos de acordo com o valor específico
                y_val_dividido = y_val / valor_divisao
                y_pred_dividido = y_pred / valor_divisao

                RMSE_k_weight.append(mean_squared_error(y_val_dividido, y_pred_dividido, squared=False))

            RMSE_mean = np.mean(RMSE_k_weight)  # Calcula a média do RMSE para todos os folds
            if RMSE_mean < best_RMSE:
                best_RMSE = RMSE_mean
                best_k = k
                best_weight = weight

    # Treinar o modelo com os melhores parâmetros encontrados
    knr = KNeighborsRegressor(n_neighbors=best_k, weights=best_weight)
    knr.fit(X_train_val, y_train_val)

    # Obter as previsões para X_test
    y_pred = knr.predict(X_test)

    # Dividir os valores reais e preditos de acordo com o valor específico
    y_test_dividido = y_test / valor_divisao
    y_pred_dividido = y_pred / valor_divisao

    # Calcular o RMSE para o conjunto de teste
    test_RMSE = mean_squared_error(y_test_dividido, y_pred_dividido, squared=False)

    # Imprimir os melhores parâmetros e o RMSE para cada disciplina
    print(f"\nResultados para {dataset}:")
    print(f"Melhores parâmetros: k = {best_k}, weights = {best_weight}")
    print(f"RMSE: {test_RMSE}")

    # Criar um DataFrame para armazenar os resultados
    results = pd.DataFrame({
        'Real': y_test_dividido,
        'Predito': y_pred_dividido
    })

    # Salvar os resultados em uma planilha Excel
    results.to_excel(f"/content/resultados_{dataset.split('.')[0]}.xlsx", index=False)
    print(f"Resultados salvos em: resultados_{dataset.split('.')[0]}.xlsx")

print("Processamento concluído.")