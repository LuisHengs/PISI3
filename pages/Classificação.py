import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para criar o gráfico com a legenda
def criar_grafico(coordenadas_filmes, cores_legenda, classes_filmes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Atribuir marcadores diferentes para cada classe
    marcadores = ['o', 's', '^', 'D', 'v']
    
    # Iterar sobre as classes e plotar os pontos
    for i, classe in enumerate(np.unique(classes_filmes)):
        ax.scatter(
            coordenadas_filmes[classes_filmes == classe, 0],
            coordenadas_filmes[classes_filmes == classe, 1],
            c=cores_legenda[classes_filmes == classe],
            marker=marcadores[i],
            label=f'Classe {classe}'
        )

    # Adicionar legenda
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Adicionar rótulos aos eixos
    ax.set_xlabel('Coordenada 1')
    ax.set_ylabel('Coordenada 2')

    # Adicionar título ao gráfico
    ax.set_title('Classificação de Filmes usando k-NN')

    # Adicionar barra de cores para as legendas
    criar_barra_cores(cores_legenda)
    
    return fig

# Função para criar a barra de cores da legenda
def criar_barra_cores(cores_legenda):
    fig, ax = plt.subplots(figsize=(8, 1))
    
    # Converter a matriz numpy para uma série do pandas e, em seguida, chamar unique()
    ax.imshow([pd.Series(cores_legenda).unique()], aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

# Criar dados de exemplo
np.random.seed(42)
coordenadas_filmes = np.random.rand(100, 2)
cores_legenda = np.random.rand(100)
classes_filmes = np.random.choice([1, 2, 3, 4, 5], size=100)

# Criar e exibir o gráfico
st.pyplot(criar_grafico(coordenadas_filmes, cores_legenda, classes_filmes))
