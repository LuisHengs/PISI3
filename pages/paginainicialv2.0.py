import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .header {
        background-color: #0078D4;
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 24px;
    }
    .stats-container {
        background-color: #F4F4F4;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .stats {
        font-size: 20px;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def build_header():
    st.markdown("<div class='header'>Produção de um modelo alternativo para recomendação de filmes e séries.</div>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>Explore uma visão abrangente dos conjuntos de dados, incluindo estatísticas gerais, com insights provenientes dos conjuntos de dados da Netflix e Amazon Prime.</p>", unsafe_allow_html=True)
    st.markdown("---")

def build_dataframes():
    st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center;'>Visão geral dos conjuntos de dados</h2>", unsafe_allow_html=True)

    netflix_data = pd.read_parquet('data/netflix_titles.parquet')
    amazon_prime_data = pd.read_parquet('data/amazon_prime_titles.parquet')

    total_movies = len(netflix_data) + len(amazon_prime_data)

    total_directors = len(set(netflix_data['director'].dropna()) | set(amazon_prime_data['director'].dropna()))

    total_actors = len(set(','.join(netflix_data['cast'].dropna()).split(', ')) | set(','.join(amazon_prime_data['cast'].dropna()).split(', ')))

    st.write("<h3>Análise Geral dos Dados Combinados:</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='stats'>Quantidade total de filmes e séries:<br><b>{total_movies}</b></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stats'>Quantidade total de diretores:<br><b>{total_directors}</b></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stats'>Quantidade total de atores:<br><b>{total_actors}</b></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

build_header()
build_dataframes()
