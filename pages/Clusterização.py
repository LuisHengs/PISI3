import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Função para pré-processamento
def preprocess_data(df):
    # Tratamento de dados ausentes
    df.fillna({'director': df['director'].mode()[0]}, inplace=True)

    # Transformação de dados categóricos (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['type', 'country', 'rating'], prefix=['type', 'country', 'rating'])

    # Pré-processamento de texto (remoção de stopwords)
    stop_words = set(stopwords.words('english'))
    df_encoded['description'] = df_encoded['description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

    # Feature Engineering (extração do ano da coluna 'date_added')
    df_encoded['year_added'] = pd.to_datetime(df_encoded['date_added'], errors='coerce').dt.year

    # Tratamento da coluna 'duration' (extrair valores numéricos)
    df_encoded['duration'] = pd.to_numeric(df_encoded['duration'].str.extract('(\d+)', expand=False), errors='coerce')

    # Normalização/Padronização
    scaler = StandardScaler()
    df_encoded[['release_year_scaled', 'duration_scaled']] = scaler.fit_transform(df_encoded[['release_year', 'duration']])
    return df_encoded

# Função para aplicar clustering
def apply_clustering(df_encoded, num_clusters):
    # Selecionar features para clustering
    features_for_clustering = ['release_year_scaled', 'duration_scaled']

    # Selecionar apenas as features relevantes
    df_cluster = df_encoded[features_for_clustering]

    # Aplicar padronização
    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster)

    # Imputar valores ausentes
    imputer = SimpleImputer(strategy='mean')
    df_cluster_scaled_imputed = imputer.fit_transform(df_cluster_scaled)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_encoded['cluster'] = kmeans.fit_predict(df_cluster_scaled_imputed)

    return df_encoded

# Função fictícia para criar um gráfico de dispersão com base nos resultados do clustering
def create_scatter_plot(df, num_clusters):
    fig = px.scatter(df, x='release_year_scaled', y='duration_scaled', color='cluster',
                     title=f'Scatter Plot (K-Means, {num_clusters} clusters)',
                     labels={'release_year_scaled': 'Ano de Lançamento Padronizado',
                             'duration_scaled': 'Duração Padronizada'})
    return fig

# Função fictícia para criar um box plot com base nos resultados do clustering
def create_box_plot(df, num_clusters):
    fig = go.Figure()
    for cluster in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster]
        fig.add_trace(go.Box(y=cluster_data['duration_scaled'], name=f'Cluster {cluster}'))

    fig.update_layout(title=f'Box Plot (K-Means, {num_clusters} clusters)',
                      xaxis_title='Cluster',
                      yaxis_title='Duração Padronizada')

    return fig

# Título do aplicativo
st.title("Carregamento, Pré-Processamento e Clustering de Conjunto de Dados com Streamlit")

# Seção para upload do arquivo
st.sidebar.header("Upload do Conjunto de Dados")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo Parquet", type=["parquet"])

# Seção para seleção do número de clusters e tipo de gráfico
num_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=3)
chart_type = st.sidebar.selectbox("Tipo de Gráfico", ['scatter', 'box'])  # Adicione outros tipos de gráfico conforme necessário

# Verificar se um arquivo foi carregado
if uploaded_file is not None:
    # Carregar o conjunto de dados em um DataFrame
    df = pd.read_parquet(uploaded_file)

    # Exibir as primeiras linhas do DataFrame original
    st.write("*Conjunto de Dados Original:*")
    st.write(df.head())

    # Pré-processar o conjunto de dados
    df_encoded = preprocess_data(df)

    # Exibir as primeiras linhas do DataFrame pré-processado
    st.write("*Conjunto de Dados Pré-Processado:*")
    st.write(df_encoded.head())

    # Aplicar clustering
    df_encoded = apply_clustering(df_encoded, num_clusters)

    # Exibir as primeiras linhas do DataFrame com as informações de clustering
    st.write("*Conjunto de Dados com Informações de Clustering:*")
    st.write(df_encoded.head())

    # Escolher e exibir o tipo de gráfico
    if chart_type == 'scatter':
        st.plotly_chart(create_scatter_plot(df_encoded, num_clusters))
    elif chart_type == 'box':
        st.plotly_chart(create_box_plot(df_encoded, num_clusters))

    # Continuar com análises adicionais conforme necessário
else:
    st.info("Por favor, faça o upload de um arquivo Parquet.")
