import streamlit as st
# Primero hay que importar las librerías necesarias para leer el data set y hacer el modelo predictivo
# Librerías generales
import pandas as pd
import numpy as np
from copy import deepcopy

#Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

# Librerías de Procesamiento de datos
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# Librerías para calcular el score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

st.title('EDA COUSINES RATING')
st.markdown("<hr style='margin-top: 0'>", unsafe_allow_html=True) # Un separador

st.subheader("*Dataset para realizar el EDA*")
st.markdown("[Enlace al dataset](https://www.kaggle.com/datasets/surajjha101/cuisine-rating)")

# Leamos el dataset que vamos a utilizar para el EDA
df=pd.read_csv('Cuisine_rating.csv')
st.dataframe(df)
st.markdown("<hr style='margin-top: 0'>", unsafe_allow_html=True) # Un separador

# EDA VARIABLES CUANTITATIVAS
st.title("EDA VARIABLES CUANTITATIVAS")
st.subheader("*Calificación Promedio por Tipo de Cocina*")
# Calificación promedio por tipo de cocina
promedio_por_cocina = df.groupby('Cuisines')['Food Rating'].mean().reset_index()

# Gráfico de barras con Seaborn
fig, ax = plt.subplots()
colores = sns.color_palette("Blues", n_colors=len(promedio_por_cocina))
sns.barplot(x='Cuisines', y='Food Rating', data=promedio_por_cocina,palette=colores,ax=ax)
plt.xticks(rotation=90)
plt.xlabel('Tipos de Cocina')
plt.ylabel('Calificación Promedio')
st.pyplot(fig)

st.subheader("*Calificación Promedio del Servicio por Ubicación*")
# Calificación promedio por ubicación
promedio_por_ubicacion = df.groupby('Location')['Service Rating'].mean().reset_index()

# Colores para las barras (tonalidades de verde)
colores = sns.color_palette("Greens", n_colors=len(promedio_por_ubicacion))

# Gráfico de barras con Seaborn
fig, ax = plt.subplots()
sns.barplot(x='Location', y='Service Rating', data=promedio_por_ubicacion, palette=colores, ax=ax)
plt.xticks(rotation=90)
plt.xlabel('Ubicaciones de los restaurantes')
plt.ylabel('Calificación Promedio')
plt.title('Calificación Promedio del Servicio')

# Mostrar gráfico en Streamlit
st.pyplot(fig)

st.subheader("*Calificación Promedio por Grupo:*")
# Columnas que se pueden seleccionar
columnas_seleccionables_promedio = ['Gender', 'Marital Status', 'Smoker', 'Alcohol', 'Activity', 'Often A S']

# Widget selectbox para elegir la columna
columna_seleccionada = st.selectbox("Agrupar por:", columnas_seleccionables_promedio)

# Crear la gráfica basada en la columna seleccionada
fig, ax = plt.subplots(figsize=(12, 6))

# Obtener el promedio de las calificaciones para cada valor de la columna seleccionada
promedio_rating = df.groupby(columna_seleccionada)['Overall Rating'].mean()

# Definir la paleta de colores morados
paleta_morada = sns.color_palette("Purples", len(promedio_rating))

# Graficar asignando un color de la paleta a cada barra
promedio_rating.plot(kind='bar', ax=ax, color=paleta_morada)

ax.set_xlabel(columna_seleccionada, fontsize=16)
ax.set_ylabel('Promedio Calificaciones', fontsize=16)

# Ajustar el rango del eje y
ax.set_ylim(0, 5)

# Ajustes al formato de las etiquetas del eje x (letras más grandes)
plt.xticks(rotation=45, fontsize=16) 

# Mostrar gráfico en Streamlit
st.pyplot(fig)
st.markdown("<hr style='margin-top: 0'>", unsafe_allow_html=True) # Un separador

# EDA VARIABLES CUALITATIVAS
st.title("EDA VARIABLES CUALITATIVAS")

# Columnas que se pueden seleccionar
columnas_seleccionables_frecuencia = ['Gender', 'Marital Status', 'Smoker', 'Alcohol', 'Activity', 'Often A S']

# Widget selectbox para elegir la columna
columna_seleccionada = st.selectbox("Selecciona una columna:", columnas_seleccionables_frecuencia)

# Calcular la frecuencia de valores en la columna seleccionada
frecuencia_valores = df[columna_seleccionada].value_counts()

# Mostrar la frecuencia de valores en una tabla
st.write(f"### Frecuencia de valores en la columna '{columna_seleccionada}':")
st.table(frecuencia_valores)

st.write(f"### Cocina preferida según el grupo: '{columna_seleccionada}':")

# Crear un subplot
fig, ax = plt.subplots(figsize=(12, 6))

# Obtener la frecuencia de cada cocina para cada valor de la columna categórica
cocinas_frecuencia = df.groupby(['Cuisines', columna_seleccionada])['Food Rating'].mean().unstack()
paleta_BG = sns.color_palette("twilight", len(cocinas_frecuencia))

# Graficar
cocinas_frecuencia.plot(kind='bar', stacked=False, ax=ax, color=paleta_BG)

ax.set_xlabel('Cocina')
ax.set_ylabel('Food Rating')
ax.legend(title=columna_seleccionada)

# Ajustes de diseño
plt.xticks(rotation=45, ha='right')
plt.yticks(range(int(cocinas_frecuencia.max().max()) + 1))

# Mostrar la gráfica en Streamlit
st.pyplot(fig)
st.markdown("<hr style='margin-top: 0'>", unsafe_allow_html=True) # Un separador

# MODELO KNEIGHBORS
st.title("MODELO KNEIGHBORS :)")
st.subheader("*Pre-procesamiento*")
st.write("Así queda nuestro DataFrame original después del Pre-Procesamiento:")

# Redondear los valores de 'Overall Rating'
df['Overall Rating']=df['Overall Rating'].round().astype('int64')
# Crear una columna de edad para sustituir la columna de YOB que es el año de nacimiento
df['Age'] = 2024 - df['YOB']
# Aplicamos un OrdinalEncoder a las columnas donde hay un tipo de orden/jerarquía que indice que algo tiene más relevancia que otro
# La única columna que presenta un tipo de orden es Maritual Status
ordinal_encoder = OrdinalEncoder()
df['Marital Status Encoded'] = ordinal_encoder.fit_transform(df[['Marital Status']])
# Nos deshacemos de las columnas que no son relevantes para el df por las siguientes razones:
    # 1- User Id es un contador de filas, no indica nada 
    # 2- Nos deshacemos de Area Code ya que nos quedamos con Location y brindan la misma información
    # 3- Yob la codifcicamos para que nos diera la edad del cliente, es inecesaria aplica lo mismo para Marital Status
    # 4- Food y Service rating son columnas que su promedio genera la target, por lo tanto se descartan
df=df.drop(['User ID','Area code','YOB','Food Rating','Service Rating', 'Marital Status'],axis=1)
# El modelo KNN solo usa variables numéricas asi que aplicamos codificación One Hot a las variables categóricas 
df_onehot=pd.get_dummies(df).astype('int64')
# Establecemos los parámetros del modelado
x = df_onehot.drop(columns=['Overall Rating']) # Independientes
y = df_onehot['Overall Rating'] # Target
# Establecemos los parámetros del modelado
# Los datos deben de tener la misma distancia para que la precisión del modelo sea la óptima, hagamos un MinMax ya que la mayoría...
    # De nuestras columnas son One-Hot encoded, asi que sus valores son 0 o 1.
scaler = MinMaxScaler()
X = scaler.fit_transform(x)
# Crear DataFrame con X e y
df_visualizacion = pd.DataFrame(data=X, columns=x.columns)
df_visualizacion['Overall Rating'] = y

# Visualizar el DataFrame en Streamlit
st.write(df_visualizacion)

st.subheader("*OverSampling  :,(*")
st.write("Como se puede ver, hay un claro OverSampling en nuestros datos, esto afecta la precisión del modelo.")
# Obtener la frecuencia de cada clase en y
frecuencia_y = y.value_counts()

# Crear la paleta de colores "inferno"
paleta_inferno = sns.color_palette("viridis", len(frecuencia_y))

# Crear la gráfica de barras
fig, ax = plt.subplots()
sns.barplot(x=frecuencia_y.index, y=frecuencia_y.values, ax=ax, palette=paleta_inferno)
ax.set_xlabel('Clase de Overall Rating')
ax.set_ylabel('Frecuencia')
ax.set_title('Frecuencia de Clases en Overall Rating')

# Mostrar gráfico en Streamlit
st.pyplot(fig)

st.subheader("*¿Y ahora que hacemos?*")
st.write("Usando la técnica de SMOTE (Syntetic Minority Over-sampling Technique) el df queda algo así:")
# Smote funciona como un kneighbors, lo que hace es crear datos que tengan las caracteristicas de la clase menor balanceada
smote = SMOTE(random_state=707)
X_smote, y_smote = smote.fit_resample(X, y)
st.write("Valores de y_smote:", y_smote)
# Como no tenemos un df de testeo, separemos el que tenemos en 4
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=707, stratify=y_smote)
# Usando la librería de GridSearch, buscamos los parámetros de nuestro modelo de clasificación que generen el mejor score, así ...
    # Evitamos hacer la búsqueda manualmente

parametros = {
    'n_neighbors': list(range(3,11)),  # Diferentes valores para el número de vecinos
    'weights': ['uniform', 'distance'],  # Diferentes métodos de ponderación
    'metric': ['euclidean', 'manhattan', 'chebyshev'] 
}

knn_parametros = KNeighborsClassifier()

# Crear un objeto GridSearchCV
grid_search = GridSearchCV(knn_parametros, parametros, cv=5)

# Entrenar el modelo con los datos después de aplicar SMOTE
grid_search.fit(X_train, y_train)
mejores_hiperparametros = grid_search.best_params_

st.subheader("*KNeighborsClassifier Visualización*")
# Crear y entrenar el modelo KNN con los mejores hiperparámetros
knn_classifier = KNeighborsClassifier(**mejores_hiperparametros)
knn_classifier.fit(X_train, y_train)
# Realizar predicciones en el conjunto de prueba
y_pred = knn_classifier.predict(X_test)
# Calcular el score de predicción en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
# Mostrar la precisión en Streamlit
st.write(f'La precisión del modelo es: {accuracy:.2f}')
x_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=10, random_state=707).fit_transform(X_test)

# Paleta de colores personalizada
colors = ['#DCDA36', '#48D5E0', '#2323E0', '#AA23E0', '#5DE818']

# Graficar los puntos con etiquetas y tamaño de los marcadores más grandes
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))

# Crear una figura 3D con dos subplots
fig = plt.figure(figsize=(12, 6))

# Primer subplot para y_test
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(
    xs=x_embedded[:, 0], 
    ys=x_embedded[:, 1],
    zs=x_embedded[:, 2],
    c=y_test,
    cmap=cmap_custom,
    s=50
)
ax1.set_title('TSNE de KNeighbors [Y_Test]')
legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
ax1.add_artist(legend1)

# Segundo subplot para y_pred
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(
    xs=x_embedded[:, 0], 
    ys=x_embedded[:, 1],
    zs=x_embedded[:, 2],
    c=y_pred,
    cmap=cmap_custom,
    s=50
)
ax2.set_title('TSNE de KNeighbors [Y_Pred]')
legend2 = ax2.legend(*scatter2.legend_elements(), title="Clusters")
ax2.add_artist(legend2)

# Ajustar el diseño de la figura
plt.tight_layout()

# Mostrar la figura en Streamlit
st.pyplot(fig)

st.write("Es algo díficil ver que valores son los que el modelo classificó erróneamente verdad?, Usemos otra herramienta para ver el rendimiento:")
st.subheader("*Matriz de Confusión*")
conf_matrix = confusion_matrix(y_test, y_pred)

# Graficar la matriz de confusión
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)

# Ajustar las etiquetas del eje x y y
ax.set_xticks([i + 0.5 for i in range(len(conf_matrix))], minor=False)
ax.set_yticks([i + 0.5 for i in range(len(conf_matrix))], minor=False)
ax.set_xticklabels([i + 1 for i in range(len(conf_matrix))], minor=False)
ax.set_yticklabels([i + 1 for i in range(len(conf_matrix))], minor=False)

ax.set_xlabel('Clases Kneigbors')
ax.set_ylabel('Clases Reales')
ax.set_title('Matriz de Confusión')
plt.tight_layout()

# Mostrar la matriz de confusión en Streamlit
st.pyplot(fig)
st.markdown("<hr style='margin-top: 0'>", unsafe_allow_html=True) # Un separador

st.title('Fuentes consultadas para la realización del Proyecto:')
import streamlit as st

st.markdown("""
- [SMOTE — version 0.12.2.](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [K Nearest Neighbors (KNN) hyperparameter tuning with gridsearch (no talking)](https://www.youtube.com/watch?v=h3ARWw3uSoE)
- [SMOTE - Handle imbalanced dataset | Synthetic Minority Oversampling Technique | Machine Learning](https://www.youtube.com/watch?v=adHqzek--d0)
- [Imbalanced data classification: Oversampling and Undersampling](https://medium.com/@debspeaks/imbalanced-data-classification-oversampling-and-undersampling-297ba21fbd7c)
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
""")




