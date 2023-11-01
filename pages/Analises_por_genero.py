import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def carregar_dataset_netflix():
    data = pd.read_parquet("netflix_titles.parquet")
    return data

@st.cache_data
def carregar_dataset_amazon():
    data = pd.read_parquet("amazon_prime_titles.parquet")
    return data

st.title("Distribuição de Filmes e Séries por Gênero")


selected_service = st.radio("Escolha o serviço de streaming", ["Netflix", "Amazon Prime"])


if selected_service == "Netflix":
    df = carregar_dataset_netflix()
else:
    df = carregar_dataset_amazon()


selected_chart = st.radio("Escolha o tipo de gráfico", ["Filmes", "Séries"])


if selected_chart == "Filmes":
    chart_data = df[df["type"] == "Movie"]
else:
    chart_data = df[df["type"] == "TV Show"]


genre_counts = chart_data["listed_in"].value_counts().reset_index()
genre_counts.columns = ["Gênero", f"Número de {selected_chart}"]
genre_counts = genre_counts.sort_values(by=f"Número de {selected_chart}", ascending=False)


outros_threshold = 10
top_genres = genre_counts.head(outros_threshold)
outros_genres = genre_counts.tail(len(genre_counts) - outros_threshold)
outros_row = pd.DataFrame(data={
    "Gênero": ["Outros"],
    f"Número de {selected_chart}": [outros_genres[f"Número de {selected_chart}"].sum()]
})
genre_counts = pd.concat([top_genres, outros_row])

fig = px.bar(genre_counts, x="Gênero", y=f"Número de {selected_chart}",
             title=f"Distribuição de {selected_chart} por Gênero no {selected_service}",
             color="Gênero", color_discrete_sequence=px.colors.qualitative.Set3)


fig.update_xaxes(tickangle=45, categoryorder="total ascending")


fig.update_xaxes(title_text="Gênero")
fig.update_yaxes(title_text=f"Número de {selected_chart}")



st.plotly_chart(fig)
