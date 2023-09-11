# Importar librerias 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# Importar el dataset
dataset = pd.read_csv('Framework_LinearRegression/placement.csv')

# Visualizar el dataset
dataset.head()

# Visualizar la informacion del dataset
dataset.info()

# Visualizar la descripcion del dataset
dataset.describe()

# Visualizar la correlacion entre las variables
sns.heatmap(dataset.corr(), annot=True)

plt.figure(figsize = (16,5))
plt.subplot(1,2,1)
sns.distplot(dataset['cgpa'])

plt.subplot(1,2,2)
sns.distplot(dataset['placement_exam_marks'])

plt.show()

print('Valor promedio del cgpa',dataset['cgpa'].mean())
print('Desviación estandar de cgpa',dataset['cgpa'].std())
print('Valor mínimo del cgpa',dataset['cgpa'].min())
print('Valor máximo del cgpa',dataset['cgpa'].max())

# Eliminar los valores que estan en los extremos
# En este caso vamos a permitir los valores que esten dentro de 3 desviaciones estandar
high_limit = dataset['cgpa'].mean() + 3*dataset['cgpa'].std()
low_limit = dataset['cgpa'].mean() - 3*dataset['cgpa'].std()
print("Limite mayor: ", high_limit)
print("Limite menor: ", low_limit)

print('Cantidad de valores antes de eliminar los valores extremos',dataset.shape)
dataset = dataset[(dataset['cgpa'] < high_limit) & (dataset['cgpa'] > low_limit)]
print('Cantidad de valores despues de eliminar los valores extremos',dataset.shape)



