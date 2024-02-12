import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_parquet('data/dados_netflix|amazon_5.parquet')
columns_to_normalize = ['ano_lancamento', 'duração', 'Filme/Série', 'Categoria', 'Generos']


X = df[columns_to_normalize]
X_scaled = StandardScaler().fit_transform(X)

st.title('Análise do Cotovelo e Silhueta com Streamlit')

# Range de possíveis números de clusters
range_n_clusters = range(2, 11)

# Armazenar valores de inércia e silhueta
inertia_values = []
silhouette_avg_values = []

# Iterar sobre o número de clusters
for n_clusters in range_n_clusters:
    # Inicializar o clusterer e prever os rótulos
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calcular a inércia e armazenar
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
    st.write(f'Número de clusters = {n_clusters}, Inércia = {inertia:.2f}')

    # Calcular a pontuação média de silhueta
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_avg_values.append(silhouette_avg)
    st.write(f'Número de clusters = {n_clusters}, Pontuação de Silhueta Média = {silhouette_avg:.2f}')

# Plotar o método do cotovelo
fig, ax1 = plt.subplots()
fig.set_size_inches(10, 6)

# Gráfico da inércia
ax1.plot(range_n_clusters, inertia_values, marker='o', label='Inércia')
ax1.set_xlabel('Número de Clusters')
ax1.set_ylabel('Inércia', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Adicionar um segundo eixo y para a pontuação de silhueta
ax2 = ax1.twinx()
ax2.plot(range_n_clusters, silhouette_avg_values, marker='s', color='tab:red', label='Silhueta')
ax2.set_ylabel('Pontuação de Silhueta Média', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

st.pyplot(fig)


