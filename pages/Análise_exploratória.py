import streamlit as st
import pandas as pd
import codecs
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import streamlit.components.v1 as components
import sweetviz as sv
import plotly.express as px



st.title('Análise exploratória dos Datasets')
st.subheader('Selecione a análise que você quer ver.')

def pagina1():
    st.title("Análise geral dos datasets")

def pagina2():
    st.title("Análise quantitativa de filmes e séries.")

def pagina3():
    st.title("Página 3 - Tópico C")
    st.write("Conteúdo relacionado ao tópico C")

def pagina4():
    st.title("Página 4 - Tópico D")
    st.write("Conteúdo relacionado ao tópico D")

pagina_selecionada = st.selectbox("Selecione uma análise", ("Análise geral", "Análise quantitativa de filmes e séries.", "Página 3", "Página 4"))

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


elif pagina_selecionada == "Página 3":
    pagina3()
    #não esquecer do tab
elif pagina_selecionada == "Página 3":
    pagina4()
    #não esquecer do tab