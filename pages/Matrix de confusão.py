import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import joblib

def treinar_avaliar_modelo(X_train, y_train, X_test, y_test, otimizar_hiperparametros=False):
    if otimizar_hiperparametros:
        # Definir a grade de hiperparâmetros reduzida
        param_grid = {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        modelo = DecisionTreeClassifier()

        # Criar o objeto GridSearchCV
        grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='accuracy')

        # Ajustar o modelo aos dados
        grid_search.fit(X_train, y_train)

        melhores_hiperparametros = grid_search.best_params_
        st.write("Melhores Hiperparâmetros:", melhores_hiperparametros)

        modelo_otimizado = DecisionTreeClassifier(**melhores_hiperparametros)

        modelo_otimizado.fit(X_train, y_train)

        y_pred_otimizado = modelo_otimizado.predict(X_test)

        cm_otimizado, acuracia_otimizado, precisao_otimizado, recall_otimizado, f1_otimizado = avaliar_modelo(y_test, y_pred_otimizado)

        # Salvar o modelo otimizado
        joblib.dump(modelo_otimizado, 'modelo_otimizado.pkl')

        # Salvar resultados do modelo otimizado
        resultados_otimizado = {
            'cm': cm_otimizado,
            'acuracia': acuracia_otimizado,
            'precisao': precisao_otimizado,
            'recall': recall_otimizado,
            'f1': f1_otimizado
        }
        joblib.dump(resultados_otimizado, 'resultados_otimizado.pkl')

        fig_otimizado, ax_otimizado = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_otimizado, annot=True, fmt="d", cmap="Blues", xticklabels=modelo_otimizado.classes_, yticklabels=modelo_otimizado.classes_, ax=ax_otimizado)
        plt.title("Matriz de Confusão (Modelo Otimizado)")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        st.pyplot(fig_otimizado)

        st.write("### Métricas de Avaliação (Modelo Otimizado)")
        st.write(f"Acurácia (A): {acuracia_otimizado:.4f}")
        st.write(f"Precisão (P): {precisao_otimizado:.4f}")
        st.write(f"Recall (R): {recall_otimizado:.4f}")
        st.write(f"F1-Score (F1): {f1_otimizado:.4f}")

    modelo_nao_otimizado = DecisionTreeClassifier()

    modelo_nao_otimizado.fit(X_train, y_train)

    y_pred_nao_otimizado = modelo_nao_otimizado.predict(X_test)

    cm_nao_otimizado, acuracia_nao_otimizado, precisao_nao_otimizado, recall_nao_otimizado, f1_nao_otimizado = avaliar_modelo(y_test, y_pred_nao_otimizado)

    fig_nao_otimizado, ax_nao_otimizado = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_nao_otimizado, annot=True, fmt="d", cmap="Blues", xticklabels=modelo_nao_otimizado.classes_, yticklabels=modelo_nao_otimizado.classes_, ax=ax_nao_otimizado)
    plt.title("Matriz de Confusão (Modelo Não Otimizado)")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    st.pyplot(fig_nao_otimizado)

    st.write("### Métricas de Avaliação (Modelo Não Otimizado)")
    st.write(f"Acurácia (A): {acuracia_nao_otimizado:.4f}")
    st.write(f"Precisão (P): {precisao_nao_otimizado:.4f}")
    st.write(f"Recall (R): {recall_nao_otimizado:.4f}")
    st.write(f"F1-Score (F1): {f1_nao_otimizado:.4f}")

def avaliar_modelo(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # Calcular métricas de avaliação
    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return cm, acuracia, precisao, recall, f1

df = pd.read_parquet('data/dados_netflix|amazon_5.parquet')

colunas_predefinidas = ['Filme/Série', 'Generos', 'duração', 'ano_lancamento']

colunas_selecionadas = st.multiselect('Selecione até 4 colunas:', colunas_predefinidas, default=colunas_predefinidas, key="colunas")

if len(colunas_selecionadas) > 5:
    st.warning("Você selecionou mais de 4 colunas. Apenas as primeiras 4 serão analisadas.")

st.write("### Avaliação do Modelo")

X = df[colunas_selecionadas].dropna()
y = df['Categoria'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

otimizar_hiperparametros = st.checkbox('Otimizar Hiperparâmetros')

treinar_avaliar_modelo(X_train, y_train, X_test, y_test, otimizar_hiperparametros)
