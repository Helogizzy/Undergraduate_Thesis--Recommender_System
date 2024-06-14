import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler

# Suprimindo os warnings
import warnings
warnings.filterwarnings("ignore")

# Lista de arquivos de base de dados
datasets = [
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

# Valores específicos para dividir o predito e o real para cada base de dados
valores_divisao = [5, 7, 5, 5, 9, 6, 5, 5, 7, 8, 30, 6, 13, 5]

for i, dataset in enumerate(datasets):
    # Importando a base de dados
    data = pd.read_csv(dataset)

    # Preparando os dados
    df_dados = pd.DataFrame(data)
    X = df_dados.iloc[:, :-1]
    Y = df_dados.iloc[:, -1]

    # Normalização dos dados
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Divisão dos dados em conjuntos de treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=42)

    # Definindo os parâmetros para busca do GridSearchCV
    parametros = {'n_clusters':[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'max_iter':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

    # Inicialização do KMeans
    kmeans = KMeans()

    # Executando a busca de GridSearchCV
    Agrupador = GridSearchCV(estimator=kmeans,param_grid=parametros,scoring='neg_mean_squared_error',cv=5)
    Agrupador.fit(X_train, y_train)

    # Obtendo os melhores parâmetros
    best_params = Agrupador.best_params_
    n_clusters = best_params['n_clusters']
    max_iter = best_params['max_iter']

    print("\nMelhores parâmetros encontrados para", dataset, ":", best_params)

    # Ajuste do modelo KMeans aos dados de treinamento
    kmeans_model = KMeans(n_clusters=n_clusters, max_iter=max_iter, init='random')
    kmeans_model.fit(X_train)

    # Executa o modelo KMeans nos dados de teste
    test_clusters = kmeans_model.predict(X_test)

    # Avalie o modelo nos dados de teste
    test_RMSE = sqrt(mean_squared_error(y_test, test_clusters / valores_divisao[i]))
    print("RMSE para", dataset, ":", test_RMSE)

    # Crie um DataFrame para armazenar os valores reais e previstos
    df_resultados = pd.DataFrame({'Valor Real': y_test / valores_divisao[i], 'Valor Estimado': test_clusters / valores_divisao[i]})

    # Salvar os resultados em uma planilha Excel
    nome_arquivo = f"resultados_{dataset.split('/')[-1].split('.')[0]}.xlsx"
    df_resultados.to_excel(nome_arquivo, index=False)
    print(f"Resultados salvos em: {nome_arquivo}")