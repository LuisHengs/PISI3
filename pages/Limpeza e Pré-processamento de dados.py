import streamlit as st
import pandas as pd

# Função para carregar os datasets
def load_datasets():
    netflix_data = pd.read_parquet('data/netflix_titles.parquet')
    amazon_data = pd.read_parquet('data/amazon_prime_titles.parquet')
    df1 = pd.read_parquet('data/dados_netflix_tratados.parquet')
    df2 = pd.read_parquet('data/dados_amazon_prime_titles_tratados.parquet')
    df3 = pd.read_parquet('data/dados_netflix_com_5.parquet')
    df4 = pd.read_parquet('data/dados_amazonprime_com_5.parquet')
    df5 = pd.read_parquet('data/dados_netflix|amazon_5.parquet')
    return netflix_data, amazon_data, df1, df2, df3, df4, df5

# Função para mostrar o DataFrame no Streamlit com opção de remover coluna
def show_dataframe(dataframe, title, remove_columns=None):
    if remove_columns:
        # Verifica se as colunas existem antes de tentar removê-las
        existing_columns = set(dataframe.columns)
        columns_to_remove = [col for col in remove_columns if col in existing_columns]
        
        if columns_to_remove:
            dataframe = dataframe.drop(columns=columns_to_remove, axis=1)
        else:
            st.warning(f"Nenhuma das colunas fornecidas existe no DataFrame.")

    st.subheader(title)
    st.dataframe(dataframe)

# Configuração do Streamlit
st.title('Visualização de Dados - Netflix e Amazon Prime')

# Introdução
st.write("Este aplicativo Streamlit visa demonstrar o processo de limpeza de dados e pré-processamento em conjuntos de dados da Netflix e Amazon Prime.")

# Carregando Datasets
netflix_data, amazon_data, df1, df2, df3, df4, df5 = load_datasets()

# Mostrando Datasets Originais
show_dataframe(netflix_data, 'Netflix - Antes da Limpeza')
st.write(
    "Antes da limpeza, os conjuntos de dados da Netflix e Amazon Prime podem conter valores ausentes e formatos de dados inconsistentes."
    " A análise inicial nos permite identificar esses problemas e abordá-los durante o processo de limpeza."
)

show_dataframe(amazon_data, 'Amazon Prime - Antes da Limpeza')

# Mostrando Datasets Após Limpeza e Pré-processamento
show_dataframe(df1, 'Netflix - Após Limpeza',)
st.write(
    "Após a limpeza de dados, tratamos valores ausentes, padronizamos formatos de datas e categorias de classificação."
    "No processo inicial de limpeza de dados para o conjunto da Netflix, identificamos e preenchemos valores ausentes em informações-chave, como diretor, elenco e país. Essa abordagem visa garantir a consistência dos dados e evitar lacunas que poderiam prejudicar as análises subsequentes. "
    "Além disso, realizamos imputação na coluna 'date_added', convertendo-a para o formato de data adequado e mantendo apenas as informações relevantes. Essa transformação é essencial para uma análise temporal mais precisa. "
    "Para melhorar a legibilidade e compreensão, optamos por renomear algumas colunas do DataFrame, adotando nomes mais descritivos e intuitivos."
)

show_dataframe(df2, 'Amazon Prime - Após Limpeza',)
show_dataframe(df3, 'Netflix - Pré-processamento', 
                remove_columns=['duracao', 'id', 'diretor', 'elenco', 'pais', 'data_adicao', 'classificacao', 'descricao', 'categoria'])

st.write(
    "Iniciamos o processo de pré-processamento. Mapeamos as classificações indicativas para categorias específicas, simplificando a interpretação dos dados. "
    "Em seguida, utilizamos técnicas de codificação binária para representar informações categóricas, como gêneros. "
    "Além disso, realizamos a codificação de variáveis categóricas e agrupamento hierárquico de gêneros. "
    "Essas etapas são essenciais para tornar os dados mais acessíveis e prontos para análises mais avançadas."
)

show_dataframe(df4, 'Amazon Prime - Pré-processamento', 
                remove_columns=['duracao', 'id', 'diretor', 'elenco', 'pais', 'data_adicao', 'classificacao', 'descricao', 'categoria'])

# Mostrando Dataset Combinado
show_dataframe(df5, 'Netflix e Amazon Prime - Dados Combinados',
                remove_columns=['duracao', 'id', 'diretor', 'elenco', 'pais', 'data_adicao', 'classificacao', 'descricao', 'categoria'])

st.write(
    "Os conjuntos de dados da Netflix e Amazon Prime foram combinados em um único conjunto para análises posteriores."
    " Essa integração pode ser útil em estudos comparativos ou análises conjuntas."
)
