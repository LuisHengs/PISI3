import streamlit as st
import pandas as pd
import plotly.express as px

# Título e subtítulo
st.title('Análise quantitativa de filmes e séries.')
st.subheader('Selecione o dataset que deseja ser analisado.')

# Caixa de seleção para escolher o arquivo Parquet
selected_dataset = st.selectbox('Escolha o dataset:', ['Netflix', 'Amazon Prime'])

# Função para carregar os dados
def load_data(dataset):
    if dataset == 'Netflix':
        return pd.read_parquet('netflix_titles.parquet')
    elif dataset == 'Amazon Prime':
        return pd.read_parquet('amazon_prime_titles.parquet')

data = load_data(selected_dataset)

# Análise de Filmes e Séries
st.subheader(f'Análise dos Dados da {selected_dataset}')
num_filmes = len(data[data['type'] == 'Movie'])
num_series = len(data[data['type'] == 'TV Show'])

# Gráfico interativo com Plotly (Análise de Filmes e Séries)
fig = px.bar(x=['Filmes', 'Séries'], y=[num_filmes, num_series], text=[num_filmes, num_series], color=['Filmes', 'Séries'])
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(
    title=f'Análise dos Dados da {selected_dataset}',
    xaxis_title='Tipo de Título',
    yaxis_title='Número de Títulos'
)
st.plotly_chart(fig)

# Comparação entre os catálogos da Netflix e Amazon Prime
st.subheader('Comparação entre os catálogos da Netflix e da Amazon Prime.')

netflix_data = load_data('Netflix')
amazon_data = load_data('Amazon Prime')

# Função para contar filmes e séries
def count_movies_and_series(data):
    num_filmes = len(data[data['type'] == 'Movie'])
    num_series = len(data[data['type'] == 'TV Show'])
    return num_filmes, num_series

num_filmes_netflix, num_series_netflix = count_movies_and_series(netflix_data)
num_filmes_amazon, num_series_amazon = count_movies_and_series(amazon_data)

# Caixa de seleção para escolher entre Filmes e Séries na comparação
selected_comparison = st.selectbox('Escolha a comparação:', ['Filmes', 'Séries'])

# Gráfico interativo com Plotly (Comparação entre Filmes e Séries)
if selected_comparison == 'Filmes':
    data = pd.DataFrame({
        'Plataforma': ['Netflix', 'Amazon Prime'],
        'Filmes': [num_filmes_netflix, num_filmes_amazon]
    })
else:
    data = pd.DataFrame({
        'Plataforma': ['Netflix', 'Amazon Prime'],
        'Séries': [num_series_netflix, num_series_amazon]
    })

fig_comparison = px.bar(data, x='Plataforma', y=data.columns[1], text=data.columns[1], color='Plataforma')
fig_comparison.update_traces(texttemplate='%{text}', textposition='outside')
fig_comparison.update_layout(
    title=f'Comparação de {selected_comparison} entre Netflix e Amazon Prime',
    xaxis_title='Plataforma',
    yaxis_title=f'Número de {selected_comparison}'
)
st.plotly_chart(fig_comparison)
