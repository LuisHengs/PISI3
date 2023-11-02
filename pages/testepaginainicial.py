import streamlit as st
import pandas as pd
import plotly.express as px

def build_header():
    st.write(f'''<h1 style='text-align: center'>
             Produção de um modelo alternativo para recomendação de filmes e séries.<br></h1>
             ''', unsafe_allow_html=True)
    
    st.write(f'''<p style='text-align: center'>
            <br>A produção de um modelo alternativo para recomendação de filmes e séries é uma abordagem que busca auxiliar a forma como conteúdos audiovisuais são sugeridos aos espectadores. Utilizando algoritmos de aprendizado de máquina e análise de dados, esse modelo considera as informações obtidas através do histórico de visualizações, para oferecer recomendações personalizadas.
            <br>
            <br>Explore uma visão abrangente dos conjuntos de dados, incluindo estatísticas gerais, com insights provenientes dos dois conjuntos de dados: Netflix e Amazon Prime.<br></p>
            ''', unsafe_allow_html=True)
    st.markdown("---")

def build_dataframes():
    st.write(f'''<h2 style='text-align: center; font-size: 36px'>
            Visão geral dos conjuntos de dados<br><br></h2>
             ''', unsafe_allow_html=True)

    container = st.container()

    tabela_dataframe_combined_data(container)

    st.write(f'''<p style='text-align: center'>
                 Os números refletem a magnitude dessa biblioteca virtual: um total de 18.475 filmes e séries, envolvendo um impressionante conjunto de 9.677 diretores e mais de 60.841 atores. Essas estatísticas proporcionam uma visão abrangente da riqueza e diversidade do catálogo disponível, destacando a grande quantidade de talentos que contribuíram para a produção desse vasto universo de entretenimento.<br><br></p>
                 ''', unsafe_allow_html=True)

def tabela_dataframe_combined_data(container):
    st.write(f'''<p style='text-align: center'>
                 <br>A tabela abaixo apresenta palavras-chave extraídas dos corpos das avaliações dos filmes e séries da Netflix e da Amazon Prime.<br><br></p>
                 ''', unsafe_allow_html=True)

    # Extrair o conjunto de dados da Netflix a partir do arquivo 'data/netflix_titles.parquet'
    netflix_data = pd.read_parquet('data/netflix_titles.parquet')

    # Extrair o conjunto de dados da Amazon Prime a partir do arquivo 'data/amazon_prime_titles.parquet'
    amazon_prime_data = pd.read_parquet('data/amazon_prime_titles.parquet')

    # Combina os dados dos dois conjuntos
    combined_data = pd.concat([netflix_data, amazon_prime_data], ignore_index=True)

    # Realizar a análise com base nos dados combinados
    total_movies = len(combined_data)  # Total de filmes e séries
    total_directors = len(combined_data['director'].unique())  # Total de diretores únicos
    total_actors = len(combined_data['cast'].str.split(', ', expand=True).stack().unique())  # Total de atores únicos

    st.write("Análise Geral dos Dados Combinados:")
    st.write(f"Quantidade total de filmes e séries: {total_movies}")
    st.write(f"Quantidade total de diretores: {total_directors}")
    st.write(f"Quantidade total de atores: {total_actors}")

    labels = ["Quantidade total de filmes e séries", "Quantidade total de diretores", "Quantidade total de atores"]
    values = [total_movies, total_directors, total_actors]

    fig = px.pie(names=labels, values=values, title="Análise Geral dos Dados Combinados")
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig)

build_header()
build_dataframes()
