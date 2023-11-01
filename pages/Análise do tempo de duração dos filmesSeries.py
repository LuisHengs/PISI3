import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Carregue os conjuntos de dados Parquet
data_netflix = pd.read_parquet('data/netflix_titles.parquet')
data_amazon_prime = pd.read_parquet('data/amazon_prime_titles.parquet')

# Função para criar e mostrar o gráfico de duração dos filmes
def plot_duration(dataset, title):
    if 'duration' in dataset.columns:
        duration_data = dataset['duration'].str.extract('(\d+)').astype(float).dropna()
        if not duration_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(duration_data, bins=40, alpha=0.5)
            ax.set_xlabel('Duração dos Filmes (em minutos)')
            ax.set_ylabel('Número de Filmes/Séries')
            ax.set_title(title)
            st.pyplot(fig)
        else:
            st.write("A coluna 'duration' não contém valores válidos no conjunto de dados.")
    else:
        st.write("A coluna 'duration' não está presente no conjunto de dados.")

# Configuração da página Streamlit
st.title("Análise da Duração dos Filmes")
selected_dataset = st.selectbox('Escolha o conjunto de dados:', ['Netflix', 'Amazon Prime'])

if selected_dataset == 'Netflix':
    st.header('Duração dos Filmes na Netflix')
    plot_duration(data_netflix, 'Duração dos Filmes na Netflix')

elif selected_dataset == 'Amazon Prime':
    st.header('Duração dos Filmes no Amazon Prime')
    plot_duration(data_amazon_prime, 'Duração dos Filmes no Amazon Prime')
