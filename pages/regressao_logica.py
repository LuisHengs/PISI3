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
