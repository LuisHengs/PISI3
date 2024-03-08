import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Função para criar e treinar modelos
def treinar_modelo(X_train, y_train, hiperparametros):
    modelo = GaussianNB(**hiperparametros)
    modelo.fit(X_train, y_train)
    return modelo

# Função para avaliar modelos
def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return cm, acuracia, precisao, recall, f1

# Carregar os dados
df = pd.read_parquet("data/dados_netflix|amazon_5.parquet")

# Selecionar colunas
colunas_predefinidas = ['Filme/Série', 'Categoria', 'Generos', 'duração', 'ano_lancamento']
colunas_selecionadas = st.multiselect('Selecione até 5 colunas:', colunas_predefinidas, default=colunas_predefinidas, key="colunas")

if len(colunas_selecionadas) > 5:
    st.warning("Você selecionou mais de 5 colunas. Apenas as primeiras 5 serão analisadas.")

# Preencher valores ausentes
X = df[colunas_selecionadas].fillna(df[colunas_selecionadas].mean())  

# Engenharia de Recursos: Criar nova característica
X['ano_decada'] = (X['ano_lancamento'] // 10) * 10  # Cria uma nova característica representando a década do ano de lançamento
X = X.drop(columns=['ano_lancamento'])  # Remove a característica original de ano de lançamento

# Transformação de Características: Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
y = df['Categoria'].fillna(df['Categoria'].mode()[0])  # Preencher valores ausentes com a moda
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Otimização de hiperparâmetros
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
melhores_hiperparametros = grid_search.best_params_
st.write("Melhores Hiperparâmetros:", melhores_hiperparametros)

# Treinar o modelo com os melhores hiperparâmetros
modelo = treinar_modelo(X_train, y_train, melhores_hiperparametros)

# Avaliar o modelo
cm, acuracia, precisao, recall, f1 = avaliar_modelo(modelo, X_test, y_test)

# Exibir a matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=modelo.classes_, yticklabels=modelo.classes_, ax=ax)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
st.pyplot(fig)

# Exibir métricas de avaliação
st.write("### Métricas de Avaliação")
st.write(f"Acurácia (A): {acuracia:.4f}")
st.write(f"Precisão (P): {precisao:.4f}")
st.write(f"Recall (R): {recall:.4f}")
st.write(f"F1-Score (F1): {f1:.4f}")

# Calcular a importância das características (Feature Importance)
variancia_total = np.var(X_train)
variancias_por_classe = []
for classe in modelo.classes_:
    indices_classe = (y_train == classe)
    variancia_por_classe = np.var(X_train[indices_classe], axis=0)
    variancias_por_classe.append(variancia_por_classe)

importancia_caracteristicas = np.array(variancias_por_classe) / variancia_total

# Normalizar a importância para que somem 100%
importancia_normalizada = (importancia_caracteristicas / importancia_caracteristicas.sum(axis=0)) * 100

# Exibir a importância das características
st.write("### Importância das Características")
for coluna, importancia in zip(colunas_selecionadas + ['ano_decada'], importancia_normalizada.T):
    st.write(f"{coluna}: {importancia[0]:.2f}%")
