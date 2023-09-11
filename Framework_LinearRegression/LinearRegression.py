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

#asignación de variables
X = dataset.iloc[:,0:1]
y = dataset.iloc[:,1]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 10)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

#Entrenamiento del modelo
lin_reg.fit(x_train, y_train)

#Predicción de los valores de test
y_pred = lin_reg.predict(x_test)

#Visualización de los resultados de entrenamiento
plt.figure(figsize=(9,8))
plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train, lin_reg.predict(x_train), color = 'red')
plt.title('CGPA vs Placement Exam Marks (Training set)')
plt.xlabel('CGPA')
plt.ylabel('Placement Exam Marks')
plt.show()

#Evaluacion del desempeno del modelo en el conjunto de entrenamiento
print('R2 score: ', r2_score(y_train, lin_reg.predict(x_train)))
print('RMSE score: ', np.sqrt(mean_squared_error(y_train, lin_reg.predict(x_train))))

#Visualización de los resultados de test
plt.figure(figsize=(9,8))
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, lin_reg.predict(x_test), color = 'red')
plt.title('CGPA vs Placement Exam Marks (Test set)')
plt.xlabel('CGPA')
plt.ylabel('Placement Exam Marks')
plt.show()

#Evaluacion del desempeno del modelo en el conjunto de test
print('R2 score: ', r2_score(y_test, lin_reg.predict(x_test)))
print('RMSE score: ', np.sqrt(mean_squared_error(y_test, lin_reg.predict(x_test))))






