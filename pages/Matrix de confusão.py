import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_parquet('data/dados_netflix|amazon_5.parquet')


colunas_predefinidas = ['Filme/Série', 'Categoria', 'Generos', 'duração', 'ano_lancamento']

colunas_selecionadas = st.multiselect('Selecione até 5 colunas:', colunas_predefinidas, default=colunas_predefinidas, key="colunas")
if len(colunas_selecionadas) > 5:
    st.warning("Você selecionou mais de 5 colunas. Apenas as primeiras 5 serão analisadas.")

st.write("### Matriz de Confusão")

X = df[colunas_selecionadas].dropna()
y = df['Categoria'].dropna()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Treinar o modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Obter as previsões do modelo
y_pred = modelo.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão usando seaborn
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=modelo.classes_, yticklabels=modelo.classes_, ax=ax)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
st.pyplot(fig)

