import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Função para carregar os dados
def load_data():
    # Carregue seus dados aqui, por exemplo, a partir de um arquivo CSV
    data = pd.read_parquet('dados_netflix_amazon_5.parquet')
    return data
    
# Função principal para a aplicação Streamlit
def main():
    st.title('Regressão Logística com Streamlit')

    # Carregar os dados
    data = load_data()

    # Fazer uma cópia dos dados
    data_copy = data.copy()
    
    # Pré-processamento dos dados
    X = data_copy[['id', 'titulo', 'diretor', 'elenco', 'pais', 'data_adicao', 'ano_lancamento', 'duracao', 'descricao']]
    y = data_copy['classificacao']  # Substitua 'alvo' pelo nome da sua coluna alvo

    # Converter colunas categóricas em numéricas
    label_encoder = LabelEncoder()
    X['titulo'] = label_encoder.fit_transform(X['titulo'])
    X['diretor'] = label_encoder.fit_transform(X['diretor'])
    X['elenco'] = label_encoder.fit_transform(X['elenco'])
    X['pais'] = label_encoder.fit_transform(X['pais'])

    # Remover observações com valores não numéricos na coluna 'duracao'
    X['duracao'] = pd.to_numeric(X['duracao'].str.replace(' min', ''), errors='coerce')
    X.dropna(subset=['duracao'], inplace=True)

    # Redefinir y após a remoção das amostras inválidas
    y = data_copy.loc[X.index, 'classificacao']

    # Remover classes com poucas amostras
    class_counts = y.value_counts()
    min_samples = 10
    classes_to_remove = class_counts[class_counts < min_samples].index
    X = X[~y.isin(classes_to_remove)]
    y = y[~y.isin(classes_to_remove)]

    # Dividir os dados em conjuntos de treinamento e teste com amostragem estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Construir e treinar o modelo
    model = Pipeline([
        ('scaler', ColumnTransformer([('scaler', StandardScaler(), [6, 7])])),
        ('logistic_regression', LogisticRegression())
    ])

    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Acurácia:", accuracy)

    # Exibir a matriz de confusão
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    st.write("Matriz de Confusão:")
    st.write(confusion_matrix_result)

    # Plotar gráfico de dispersão com classificação
    np.random.seed(0)
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    colors = np.random.rand(100)
    area = (30 * np.random.rand(100))**2  # Tamanhos dos pontos
    labels = model.predict(X)[:100]  # Prever classes para os primeiros 100 pontos

    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=colors, s=area, cmap='viridis', alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend)
    st.pyplot(fig)
