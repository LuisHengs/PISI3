import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance  # Adicione esta importação
import joblib
from sklearn.model_selection import GridSearchCV

st.set_option('deprecation.showPyplotGlobalUse', False)

# Função para treinar e avaliar o modelo
def treinar_avaliar_modelo(X_train, y_train, X_test, y_test, otimizar_hiperparametros=False):
    if otimizar_hiperparametros:

        st.text("Otimizando hiperparâmetros... Isso pode levar algum tempo.")

        # Reduzir a grade de hiperparâmetros
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5]
        }

        # Criar o modelo XGBoost
        modelo = XGBClassifier()

        # Criar o objeto GridSearchCV
        grid_search = GridSearchCV(modelo, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

        # Ajustar o modelo aos dados
        grid_search.fit(X_train, y_train)

        # Obter os melhores hiperparâmetros
        melhores_hiperparametros = grid_search.best_params_
        st.write("Melhores Hiperparâmetros:", melhores_hiperparametros)

        # Criar um novo modelo otimizado XGBoost
        modelo_otimizado = XGBClassifier(**melhores_hiperparametros)

        # Treinar o modelo otimizado
        modelo_otimizado.fit(X_train, y_train)

        # Obter as previsões do modelo otimizado
        y_pred_otimizado = modelo_otimizado.predict(X_test)

        # Avaliar o modelo otimizado
        cm_otimizado, acuracia_otimizado, precisao_otimizado, recall_otimizado, f1_otimizado = avaliar_modelo(y_test, y_pred_otimizado)

        # Salvar o modelo otimizado
        joblib.dump(modelo_otimizado, 'modelo_otimizado_xgboost.pkl')
        
        # Salvar resultados do modelo otimizado
        resultados_otimizado = {
            'cm': cm_otimizado,
            'acuracia': acuracia_otimizado,
            'precisao': precisao_otimizado,
            'recall': recall_otimizado,
            'f1': f1_otimizado
        }
        joblib.dump(resultados_otimizado, 'resultados_otimizado_xgboost.pkl')

      # Exibir a matriz de confusão usando seaborn para o modelo otimizado
        fig_otimizado, ax_otimizado = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_otimizado, annot=True, fmt="d", cmap="Blues", xticklabels=modelo_otimizado.classes_, yticklabels=modelo_otimizado.classes_, ax=ax_otimizado)
        plt.title("Matriz de Confusão (Modelo Otimizado XGBoost)")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        st.pyplot(fig_otimizado)


        # Exibir métricas de avaliação para o modelo otimizado
        st.write("### Métricas de Avaliação (Modelo Otimizado XGBoost)")
        st.write(f"Acurácia (A): {acuracia_otimizado:.4f}")
        st.write(f"Precisão (P): {precisao_otimizado:.4f}")
        st.write(f"Recall (R): {recall_otimizado:.4f}")
        st.write(f"F1-Score (F1): {f1_otimizado:.4f}")

       # Exibir importância das características para o modelo otimizado
        st.write("### Importância das Características (Modelo Otimizado XGBoost)")

        # Utilizando a função plot_importance do XGBoost para visualizar a importância das características
        fig_otimizado, ax_otimizado = plt.subplots(figsize=(8, 6))
        plot_importance(modelo_otimizado, importance_type='weight', xlabel='Contagem', ax=ax_otimizado)
        st.pyplot(fig_otimizado)

        # Adicionar texto com as porcentagens abaixo do último gráfico
        st.write("### Feature Importance (Não Otimizado):")
        st.write("Filmes/Séries: 0,83%")
        st.write("Genero: 35,22%")
        st.write("Ano: 30%")
        st.write("Duração: 33,94%")


        
      




    else:
        # Criar o modelo não otimizado
        modelo_nao_otimizado = XGBClassifier()  

        # Treinar o modelo não otimizado
        modelo_nao_otimizado.fit(X_train, y_train)

        # Obter as previsões do modelo não otimizado
        y_pred_nao_otimizado = modelo_nao_otimizado.predict(X_test)

        # Avaliar o modelo não otimizado
        cm_nao_otimizado, acuracia_nao_otimizado, precisao_nao_otimizado, recall_nao_otimizado, f1_nao_otimizado = avaliar_modelo(y_test, y_pred_nao_otimizado)

        # Salvar o modelo não otimizado
        joblib.dump(modelo_nao_otimizado, 'modelo_nao_otimizado_xgboost.pkl')

        # Salvar resultados do modelo não otimizado
        resultados_nao_otimizado = {
            'cm': cm_nao_otimizado,
            'acuracia': acuracia_nao_otimizado,
            'precisao': precisao_nao_otimizado,
            'recall': recall_nao_otimizado,
            'f1': f1_nao_otimizado
        }
        joblib.dump(resultados_nao_otimizado, 'resultados_nao_otimizado_xgboost.pkl')

        # Exibir a matriz de confusão usando seaborn para o modelo não otimizado
        fig_nao_otimizado, ax_nao_otimizado = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_nao_otimizado, annot=True, fmt="d", cmap="Blues", xticklabels=modelo_nao_otimizado.classes_, yticklabels=modelo_nao_otimizado.classes_, ax=ax_nao_otimizado)
        plt.title("Matriz de Confusão (Modelo Não Otimizado XGBoost)")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        st.pyplot(fig_nao_otimizado)


        # Exibir métricas de avaliação para o modelo não otimizado
        st.write("### Métricas de Avaliação (Modelo Não Otimizado XGBoost)")
        st.write(f"Acurácia (A): {acuracia_nao_otimizado:.4f}")
        st.write(f"Precisão (P): {precisao_nao_otimizado:.4f}")
        st.write(f"Recall (R): {recall_nao_otimizado:.4f}")
        st.write(f"F1-Score (F1): {f1_nao_otimizado:.4f}")

        st.write("### Importância das Características (Modelo Não Otimizado XGBoost)")

        # Utilizando a função plot_importance do XGBoost para visualizar a importância das características
        plot_importance(modelo_nao_otimizado, importance_type='weight', xlabel='Contagem')
        st.pyplot()

        

# Função para avaliar modelos
def avaliar_modelo(y_true, y_pred):
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Calcular métricas de avaliação
    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(y_true, y_pred, average='weighted')  # Usando 'weighted' para lidar com classes desbalanceadas
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return cm, acuracia, precisao, recall, f1

# Carregar dados
df = pd.read_parquet('data/dados_netflix|amazon_5.parquet')

# Lista de colunas predefinidas
colunas_predefinidas = ['Filme/Série','Generos', 'duração', 'ano_lancamento']

# Permitir que o usuário selecione até 5 colunas predefinidas
colunas_selecionadas = st.multiselect('Selecione até 5 colunas:', colunas_predefinidas, default=colunas_predefinidas, key="colunas")

# Verificar se o usuário selecionou até 5 colunas
if len(colunas_selecionadas) > 5:
    st.warning("Você selecionou mais de 5 colunas. Apenas as primeiras 5 serão analisadas.")

# Remover a coluna 'classificacao' dos dados
df = df.drop(columns=['classificacao'])

# Exibir a matriz de confusão e métricas de avaliação de modelos
st.write("### Avaliação do Modelo")

# Exemplo de classificação com XGBoost
X = df[colunas_selecionadas].dropna()
y = df['Categoria'].dropna()

# Convertendo categorias para valores numéricos
y = pd.Categorical(y).codes

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Opção para otimizar hiperparâmetros
otimizar_hiperparametros = st.checkbox('Otimizar Hiperparâmetros')

# Avaliar e exibir os modelos
treinar_avaliar_modelo(X_train, y_train, X_test, y_test, otimizar_hiperparametros)


