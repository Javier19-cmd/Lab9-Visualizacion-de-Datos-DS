import pandas as pd
import re
import nltk
from nltk import word_tokenize, bigrams
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ngrams
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
import streamlit as st
import numpy as np
import time

# Cargando el dataset desde un archivo CSV

train = pd.read_csv("train.csv")

# Creando un grid para enseñar las transformaciones
st.title("Transformaciones del Dataset")
# Mostrar el DataFrame original en la primera columna
st.header("DataFrame Original")
st.write(train.head())

# Cargando las primeras 5 filas del dataset.

print("train")
print(train.head())

# Convertiendo todas las columnas de texto a minúsculas del dataset train.
train = train.apply(lambda x: x.astype(str).str.lower() if x.dtype == "object" else x)
st.write(train.head())


# Función para limpiar los caracteres especiales
def clean_text(text):
    if isinstance(text, str):
        # Eliminar caracteres no alfanuméricos excepto espacios
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return cleaned_text
    else:
        return text

# Aplicando la función a todas las columnas de texto del df train.
train = train.applymap(clean_text)

# Imprimir df resultante
print(train)

st.header("DataFrame con Texto Limpio")
st.write(train.head())

# Obteneniendo estadísticas sobre las longitudes de los textos en la columna "keyword"
train['tweet_length'] = train['text'].apply(len)  # Agregando una columna con las longitudes de los textos
text_stats = train['tweet_length'].describe()

# Obteneniedo la cantidad de categorías únicas en la columna "location"
location_unique_count = train['location'].nunique()

# Imprimiendo las estadísticas de longitud de textos y la cantidad de categorías únicas en la ubicación
print("Estadísticas de longitud de textos:\n", text_stats)
print("\nCantidad de categorías únicas en base al texto:", location_unique_count)

# Coloca aquí el código de procesamiento de datos que deseas combinar
train['tweet_length'] = train['keyword'].apply(len)
text_stats = train['tweet_length'].describe()
location_unique_count = train['location'].nunique()

# Gráfico de histograma de longitudes de textos excluyendo NaN
fig_text_length = plt.figure(figsize=(10, 6))
sns.histplot(train['tweet_length'].dropna(), bins=30, kde=True, color="steelblue")
plt.title('Distribución de Longitudes de Textos')
plt.xlabel('Longitud de Textos')
plt.ylabel('Frecuencia')

# Calcular la frecuencia de las palabras clave
keyword_freq = train['keyword'].value_counts()

# Seleccionar las palabras clave más frecuentes.
top_keywords = keyword_freq.head(30)

# Crear un gráfico de barras de las palabras clave más frecuentes
fig_top_keywords = plt.figure(figsize=(10, 6))
sns.barplot(x=top_keywords.values, y=top_keywords.index, color="darkred")
plt.title('Palabras Clave Más Frecuentes')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra Clave')

# Sección de la aplicación Streamlit
st.title("Laboratorio 9")

# Barra de progreso y gráfico
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    #chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")

# Visualizar las gráficas generadas en la sección de procesamiento de datos
st.pyplot(fig_text_length)
st.pyplot(fig_top_keywords)

# Comando para correr streamlit run dashboard.py en la terminal y en el browser.