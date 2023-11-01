import streamlit as st
import pandas as pd
import codecs
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import streamlit.components.v1 as components
import sweetviz as sv



st.title('Análise exploratória dos Datasets')
st.subheader('Selecione a análise que você quer ver.')

def pagina1():
    st.title("Análise geral dos datasets")

def pagina2():
    st.title("Página 2 - Tópico B")
    st.write("Conteúdo relacionado ao tópico B")

def pagina3():
    st.title("Página 3 - Tópico C")
    st.write("Conteúdo relacionado ao tópico C")

def pagina4():
    st.title("Página 4 - Tópico D")
    st.write("Conteúdo relacionado ao tópico D")

pagina_selecionada = st.selectbox("Selecione uma análise", ("Análise geral", "Página 2", "Página 3", "Página 4"))

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
elif pagina_selecionada == "Página 2":
    pagina2()
    #não esquecer do tab
elif pagina_selecionada == "Página 3":
    pagina3()
    #não esquecer do tab
elif pagina_selecionada == "Página 3":
    pagina4()
    #não esquecer do tab