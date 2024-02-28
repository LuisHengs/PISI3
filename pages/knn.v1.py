import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset em formato parquet
merged_data = pd.read_parquet('dados_netflix_amazon_5.parquet')

# Adicionar título
st.title('Classificação de Filmes e Séries baseada no Algoritmo k-NN')

# Visualizar uma tabela expandida das primeiras linhas do DataFrame
st.write("Tabela Expandida das Primeiras Linhas:")
st.write(merged_data.head(20))  # Ajuste o número para exibir mais ou menos linhas

# Pré-processamento dos dados
le = LabelEncoder()
merged_data['encoded_genre'] = le.fit_transform(merged_data['Generos'])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    merged_data[['Generos', 'ano_lancamento']],
    merged_data['encoded_genre'],
    test_size=0.2,
    random_state=42
)

# Treinar o modelo k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Avaliar a precisão do modelo
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy:.2%}')

# Criar gráfico de classificação por gêneros
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Generos', y='ano_lancamento', hue='encoded_genre', data=merged_data, palette='viridis', s=100)
plt.title('Classificação por Gêneros usando k-NN')
plt.xlabel('Generos')
plt.ylabel('Ano de Lançamento')
plt.legend(title='Encoded Genre', loc='upper right', bbox_to_anchor=(1.25, 1))
fig = plt.gcf()  # Obter a referência para a figura atual
st.pyplot(fig)
