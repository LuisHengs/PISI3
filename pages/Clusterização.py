import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import euclidean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


def carregar_dados():
    df = pd.read_parquet("data/dados_netflix|amazon_5.parquet")
    return df

def pre_processamento(dados, colunas_selecionadas, normalizar):

    if not colunas_selecionadas:
        colunas_selecionadas = dados.columns.tolist()

    colunas_selecionadas = ['Filme/Série', 'ano_lancamento', 'Categoria', 'duração', 'Generos']
    dados_selecionados = dados[colunas_selecionadas].copy()

    dados_numericos = dados_selecionados.select_dtypes(include=['float64', 'int64'])
    dados_categoricos = dados_selecionados.select_dtypes(include=['object'])

    if normalizar:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    dados_numericos_processados = scaler.fit_transform(dados_numericos)
    dados_numericos_processados = pd.DataFrame(dados_numericos_processados, columns=dados_numericos.columns)
    dados_processados = pd.concat([dados_numericos_processados, dados_categoricos.reset_index(drop=True)], axis=1)

    return dados_processados

def calcular_distancia_entre_clusters(centros_clusters):
    num_clusters = len(centros_clusters)
    distancias = []

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distancia = euclidean(centros_clusters[i], centros_clusters[j])
            distancias.append(f'Distância entre Cluster {i} e Cluster {j}: {distancia:.4f}')

    return distancias

def plotar_grafico_dispersao(dados, coluna_x, coluna_y, coluna_hue):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dados, x=coluna_x, y=coluna_y, hue=coluna_hue, palette='tab20', s=50, alpha=1.0, edgecolor='w', linewidth=0.5)
    plt.title(f'{coluna_x} vs {coluna_y}', fontsize=8)
    plt.xlabel(coluna_x, fontsize=6)
    plt.ylabel(coluna_y, fontsize=6)
    plt.legend(title=coluna_hue, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.grid(True, linestyle='--', alpha=1.0)
    st.pyplot(plt.gcf())


def main():
    st.title("Clusterização")

    # Carregar dados
    dados = carregar_dados()

    colunas_selecionadas = ['Filme/Série', 'ano_lancamento', 'Categoria', 'duração', 'Generos']

    # Escolha entre normalização e padronização
    normalizar_dados = st.checkbox("Normalizar dados")

    dados_processados = pre_processamento(dados, colunas_selecionadas, normalizar_dados)

    # Escolha do número de clusters (K)
    numero_clusters = st.slider("Escolha o número de clusters (K)", 5, 45, 5)

    # Aplicação do K-means
    kmeans = KMeans(n_clusters=numero_clusters, random_state=42)
    kmeans.fit(dados_processados)
    rotulos_clusters = kmeans.labels_
    centros_clusters = kmeans.cluster_centers_

    dados_resultado = dados[['titulo', 'categoria', 'classificacao', 'duracao']].copy()
    dados_resultado['Clusters'] = rotulos_clusters

    colunas_centros = dados_processados.columns.tolist()
    dados_centros = pd.DataFrame(centros_clusters, columns=colunas_centros)

    # Calcular e exibir a distância entre os clusters
    distancias_entre_clusters = calcular_distancia_entre_clusters(centros_clusters)
    
    # Exibir resultados
    st.subheader("Resultados da Clusterização:")
    st.write("Resultados com Títulos e Rótulos dos Clusters:", dados_resultado)
    st.write("Centros dos Clusters:", dados_centros)
    st.subheader("Distância entre Clusters:")
    st.write(distancias_entre_clusters)

    # Plotar gráfico de dispersão para cada combinação de colunas
    st.subheader("Gráficos de Dispersão dos Clusters:")
    for col1, col2 in combinations(dados_processados.columns, 2):
        plotar_grafico_dispersao(pd.concat([dados_resultado, dados_processados], axis=1), col1, col2, 'Clusters')
    
    # Plotar gráfico de dispersão para a coluna 'Clusters'
    st.subheader("Gráfico de Dispersão para 'Clusters':")
    plotar_grafico_dispersao(dados_resultado, 'Clusters', 'Clusters', 'Clusters')

if __name__ == "__main__":
    main()