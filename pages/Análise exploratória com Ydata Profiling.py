import streamlit as st
import pandas as pd
import codecs
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import streamlit.components.v1 as components
import sweetviz as sv

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