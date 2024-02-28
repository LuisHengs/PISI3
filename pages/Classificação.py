import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset em formato parquet
# Substitua 'merged_dataset.parquet' pelo nome real do seu arquivo parquet
merged_data = pd.read_parquet('dados_amazonprime_com_5.parquet')

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

# Substituir os rótulos da legenda diretamente
plt.legend(
    title='Genre Name',
    labels=['TV Shows', "Kid's", 'Drama', 'Comedy', 'Documentaries', 'Children & Family Movies', 'Romantic Movies'],
    loc='upper right',
    bbox_to_anchor=(1.25, 1)
)

fig = plt.gcf()  # Obter a referência para a figura atual
st.pyplot(fig)


# Adicionar opções interativas para sugestões de filmes
service_streaming = st.selectbox('Selecione o Serviço de Streaming:', ['Netflix', 'Amazon Prime Video'])
production_type = st.selectbox('Selecione o Tipo de Produção:', ['Filme', 'Série'])

# Sugestões de autocompletar para o título
titles_suggestions = merged_data['titulo'].unique().tolist()
selected_title = st.selectbox("Títulos Disponíveis:", titles_suggestions)

# Filtrar o DataFrame com base no título selecionado
filtered_data_title = merged_data[merged_data['titulo'] == selected_title]

# Exibir sugestões de filmes/séries com o mesmo código de gênero e ano de lançamento
if not filtered_data_title.empty:
    genre_code = filtered_data_title.iloc[0]['encoded_genre']
    release_year = filtered_data_title.iloc[0]['ano_lancamento']
    
    same_genre_and_year_movies = merged_data[
        (merged_data['encoded_genre'] == genre_code) &
        (merged_data['ano_lancamento'] == release_year)
    ].sample(n=5, replace=False, random_state=42)

    st.write("Filmes/Séries do Mesmo Gênero e Mesmo Ano de Lançamento:")
    st.write(same_genre_and_year_movies[['titulo', 'Generos', 'ano_lancamento']])
else:
    st.warning("Nenhuma sugestão disponível para o título selecionado. Tente outras opções.")

# Exibir sugestões de filmes/séries de gêneros iguais no dataset inteiro
st.write("Filmes/Séries de Gêneros Iguais no Dataset:")
if not filtered_data_title.empty:
    same_genre_movies = merged_data[merged_data['encoded_genre'] == genre_code].sample(n=5, replace=False, random_state=42)
    st.write(same_genre_movies[['titulo', 'Generos']])
else:
    st.warning("Nenhuma sugestão disponível para o título selecionado. Tente outras opções.")
