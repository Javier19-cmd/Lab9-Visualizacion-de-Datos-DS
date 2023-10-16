# Importaciones necesarias
import pandas as pd
import re
from textblob import TextBlob
import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carga del conjunto de datos
train = pd.read_csv("train.csv")

# Título y subheader
st.title("Laboratorio 9: Visualización Interactiva")
st.sidebar.header("Parámetros de Configuración")

# Mostrar datos originales
st.subheader('Datos Originales')
st.write(train.head())

# Convertir todo a minúsculas
train = train.apply(lambda x: x.astype(str).str.lower()
                    if x.dtype == "object" else x)

# Limpiar caracteres especiales


def clean_text(text):
    if isinstance(text, str):
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return cleaned_text
    else:
        return text


train = train.applymap(clean_text)

# Mostrar datos limpios
st.subheader('Datos Limpíos')
st.write(train.head())

# Calcular longitud del tweet y polaridad
train['tweet_length'] = train['text'].apply(len)
train['polarity'] = train['text'].apply(
    lambda x: TextBlob(x).sentiment.polarity)

# Entrenamiento de Modelos
# Definición de características y etiquetas
features = ['tweet_length', 'polarity']
X = train[features]
y = train['target'].astype(int)

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Modelo de regresión logística basado en la longitud del tweet
logreg1 = LogisticRegression().fit(X_train[['tweet_length']], y_train)
y1_pred = logreg1.predict(X_test[['tweet_length']])
accuracy1 = accuracy_score(y_test, y1_pred)

# Modelo de regresión logística basado en longitud y polaridad del tweet
logreg2 = LogisticRegression().fit(X_train, y_train)
y2_pred = logreg2.predict(X_test)
accuracy2 = accuracy_score(y_test, y2_pred)

# Modelo de árbol de decisión
tree_depth = st.sidebar.slider(
    'Profundidad máxima del árbol de decisión', 1, 5, 3)
tree_model = DecisionTreeClassifier(max_depth=tree_depth).fit(X_train, y_train)
y_tree_pred = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_tree_pred)

# Mostrar resultados de modelos
st.subheader('Resultados de Modelos de Predicción')
st.write(
    f'Modelo basado solo en longitud del tweet: Precisión {accuracy1:.2f}')
st.write(f'Modelo basado en longitud y polaridad: Precisión {accuracy2:.2f}')
st.write(f'Modelo de Árbol de Decisión: Precisión {accuracy_tree:.2f}')

# Definición de la paleta de colores
color_palette = {
    "Rojo oscuro": "#8B0000",
    "Negro": "#000000",
    "Gris oscuro": "#A9A9A9",
    "Azul nocturno": "#191970",
    "Plateado": "#C0C0C0",
    "Naranja oscuro": "#FF8C00"
}

# Histograma de longitudes de textos con color "Azul nocturno"
fig_text_length = plt.figure(figsize=(10, 6))
sns.histplot(train['tweet_length'].dropna(), bins=30,
             kde=True, color=color_palette["Azul nocturno"])
plt.title('Distribución de Longitudes de Textos')
plt.xlabel('Longitud de Textos')
plt.ylabel('Frecuencia')
st.pyplot(fig_text_length)

# Gráfico de palabras clave más frecuentes con color "Rojo oscuro"
keyword_freq = train['keyword'].value_counts()
top_keywords = keyword_freq.head(30)
fig_top_keywords = plt.figure(figsize=(10, 6))
sns.barplot(x=top_keywords.values, y=top_keywords.index,
            color=color_palette["Rojo oscuro"])
plt.title('Palabras Clave Más Frecuentes')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra Clave')
st.pyplot(fig_top_keywords)

# Sliders y gráficos enlazados
tweet_length_range = st.sidebar.slider('Selecciona un rango de longitud de tweet', int(
    train['tweet_length'].min()), int(train['tweet_length'].max()), (25, 125))
polarity_range = st.sidebar.slider('Selecciona un rango de polaridad', float(
    train['polarity'].min()), float(train['polarity'].max()), (-1.0, 1.0), 0.1)
filtered_data = train[(train['tweet_length'] >= tweet_length_range[0]) &
                      (train['tweet_length'] <= tweet_length_range[1]) &
                      (train['polarity'] >= polarity_range[0]) &
                      (train['polarity'] <= polarity_range[1])]

# Histograma de polaridad filtrado con color "Azul nocturno"
fig_polarity = plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['polarity'], bins=30,
             kde=True, color=color_palette["Azul nocturno"])
plt.title('Distribución de Polaridad Filtrada')
plt.xlabel('Polaridad')
plt.ylabel('Frecuencia')
st.pyplot(fig_polarity)
