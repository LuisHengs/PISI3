import streamlit as st
import pandas as pd
import codecs
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import streamlit.components.v1 as components
import sweetviz as sv
import plotly.express as px
import matplotlib.pyplot as plt


st.title('Análise exploratória dos Datasets')
st.subheader('Selecione a análise que você quer ver.')

def pagina1():
    st.title("Análise geral dos datasets")

def pagina2():
    st.title("Análise quantitativa de filmes e séries.")

def pagina3():
    st.title("Análise do tempo de duração de filmes e séries.")
    
def pagina4():
    st.title("Análise com base nos gêneros")

pagina_selecionada = st.selectbox("Selecione uma análise", ("Análise geral", "Análise quantitativa de filmes e séries.", "Análise tempo de duração.", "Análise com base nos gêneros"))

if pagina_selecionada == "Análise geral":
    
    def st_display_sweetviz(report_html, width=1000, height=500):
      report_file = codecs.open(report_html, 'r')
      page = report_file.read()
      components.html(page, width=width, height=height, scrolling=True)

    def main():

        selected_dataset = st.selectbox('Escolha o dataset:', ['Netflix', 'Amazon Prime'])

        if selected_dataset == 'Netflix':
            data = pd.read_parquet('data/netflix_titles.parquet')
        elif selected_dataset == 'Amazon Prime':
            data = pd.read_parquet('data/amazon_prime_titles.parquet')

        menu = ["Pandas Profile", "Sweetviz"]
        choice = st.selectbox("Escolha sua ferramenta de análise exploratória", menu)

        if choice == "Pandas Profile":
            st.subheader("Pandas Profile")
            data_file = selected_dataset
            if data_file is not None:
                st.dataframe(data.head())
                profile = ProfileReport(data)
                st_profile_report(profile)

        elif choice == "Sweetviz":
            st.subheader("Sweetviz")
            data_file = selected_dataset
            if data_file is not None:
                st.dataframe(data.head())
                if st.button("Gerar relatorio Sweetviz"):
                    # Normal Workflow
                    report = sv.analyze(data)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html")

    if __name__ == '__main__':
        main()
elif pagina_selecionada == "Análise quantitativa de filmes e séries.":
    # Título e subtítulo
    st.title('Análise quantitativa de filmes e séries.')
    st.subheader('Selecione o dataset que deseja ser analisado.')

    # Caixa de seleção para escolher o arquivo Parquet
    selected_dataset = st.selectbox('Escolha o dataset:', ['Netflix', 'Amazon Prime'])

    # Carregar os datasets da Netflix e Amazon Prime
    if selected_dataset == 'Netflix':
        data = pd.read_parquet('data/netflix_titles.parquet')
        series_color = 'Séries'
    elif selected_dataset == 'Amazon Prime':
        data = pd.read_parquet('data/amazon_prime_titles.parquet')
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
    netflix_data = pd.read_parquet('data/netflix_titles.parquet')
    amazon_data = pd.read_parquet('data/amazon_prime_titles.parquet')

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


elif pagina_selecionada == "Análise tempo de duração.":
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


elif pagina_selecionada == "Análise com base nos gêneros":
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
