import streamlit as st
import pandas as pd
import plotly.express as px

# Título e subtítulo
st.title('Análise quantitativa de filmes e séries.')
st.subheader('Selecione o dataset que deseja ser analisado.')

# Caixa de seleção para escolher o arquivo Parquet
selected_dataset = st.selectbox('Escolha o dataset:', ['Netflix', 'Amazon Prime'])

# Carregar os datasets da Netflix e Amazon Prime
if selected_dataset == 'Netflix':
    data = pd.read_parquet('netflix_titles.parquet')
    series_color = 'Séries'
elif selected_dataset == 'Amazon Prime':
    data = pd.read_parquet('amazon_prime_titles.parquet')
    series_color = 'Séries'

# Análise de Filmes e Séries
st.subheader(f'Análise dos Dados da {selected_dataset}')
num_filmes = len(data[data['type'] == 'Movie'])
num_series = len(data[data['type'] == 'TV Show'])

# Definir cores das barras
cores = ['Filmes', series_color]

# Gráfico interativo com Plotly (Análise de Filmes e Séries)
fig = px.bar(x=['Filmes', 'Séries'], y=[num_filmes, num_series], text=[num_filmes, num_series], color=cores)
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(
    title=f'Análise dos Dados da {selected_dataset}',
    xaxis_title='Tipo de Título',
    yaxis_title='Número de Títulos'
)
st.plotly_chart(fig)

# Comparação entre os catálogos da Netflix e Amazon Prime
st.subheader('Comparação entre os catálogos da Netflix e da Amazon Prime.')

# Carregar os datasets da Netflix e Amazon Prime em formato Parquet
netflix_data = pd.read_parquet('netflix_titles.parquet')
amazon_data = pd.read_parquet('amazon_prime_titles.parquet')

# Filmes e Séries na Netflix
num_filmes_netflix = len(netflix_data[netflix_data['type'] == 'Movie'])
num_series_netflix = len(netflix_data[netflix_data['type'] == 'TV Show'])

# Filmes e Séries na Amazon Prime
num_filmes_amazon = len(amazon_data[amazon_data['type'] == 'Movie'])
num_series_amazon = len(amazon_data[amazon_data['type'] == 'TV Show'])

# Caixa de seleção para escolher entre Filmes e Séries na comparação
selected_comparison = st.selectbox('Escolha a comparação:', ['Filmes', 'Séries'])

# Gráfico interativo com Plotly (Comparação entre Filmes e Séries)
if selected_comparison == 'Filmes':
    data = pd.DataFrame({
        'Plataforma': ['Netflix', 'Amazon Prime'],
        'Filmes': [num_filmes_netflix, num_filmes_amazon]
    })

    fig_comparison = px.bar(data, x='Plataforma', y='Filmes', text='Filmes', color='Plataforma')
    fig_comparison.update_traces(texttemplate='%{text}', textposition='outside')
    fig_comparison.update_layout(
        title='Comparação de Filmes entre Netflix e Amazon Prime',
        xaxis_title='Plataforma',
        yaxis_title='Número de Filmes'
    )
    st.plotly_chart(fig_comparison)
else:
    data = pd.DataFrame({
        'Plataforma': ['Netflix', 'Amazon Prime'],
        'Séries': [num_series_netflix, num_series_amazon]
    })

    fig_comparison = px.bar(data, x='Plataforma', y='Séries', text='Séries', color='Plataforma')
    fig_comparison.update_traces(texttemplate='%{text}', textposition='outside')
    fig_comparison.update_layout(
        title='Comparação de Séries entre Netflix e Amazon Prime',
        xaxis_title='Plataforma',
        yaxis_title='Número de Séries'
    )
    st.plotly_chart(fig_comparison)
