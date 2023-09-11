# Importar librerias 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Importar el dataset
dataset = pd.read_csv('Framework_LinearRegression/insurance.csv')

# Visualizar el dataset
print("Visualizar el dataset")
print(dataset.head())

# Visualizar la informacion del dataset
print("Visualizar la informacion del dataset")
print(dataset.info())

# Visualizar la descripcion del dataset
print("Visualizar la descripcion del dataset")
print(dataset.describe())

#Contar el numero de valores nulos
print("Contar el numero de valores nulos")
print(dataset.isnull().sum())

sns.set(style='whitegrid')
f, ax = plt.subplots(1,1, figsize=(12, 8))
ax = sns.histplot(dataset['charges'], kde = True, color = 'c')
plt.title('Distribution of Charges')
plt.show()

# Visualizar el precio del seguro medico por regiones
charges = dataset['charges'].groupby(dataset.region).sum().sort_values(ascending = True)
f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.barplot(x=charges.head(), y=charges.head().index, palette='Blues')
plt.show()


#Visualizar el precio del seguro medico, pero tomando tambien en cuenta el sexo 
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='sex', data=dataset, palette='cool')
plt.show()
#Visualizar el precio del seguro medico, pero tomando tambien en cuenta si fuma o no
f, ax = plt.subplots(1,1, figsize=(12,8))
ax = sns.barplot(x = 'region', y = 'charges',
                 hue='smoker', data=dataset, palette='Reds_r')
plt.show()


#Visualizr el precio del seguro medico, pero tomando tambien en cuenta el numero de hijos
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='children', data=dataset, palette='Set1')
plt.show()


##Convertir las variables categoricas a numericas
dataset[['sex', 'smoker', 'region']] = dataset[['sex', 'smoker', 'region']].astype('category')

# Convertir las variables categoricas a numericas
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(dataset.sex.drop_duplicates())
dataset.sex = label.transform(dataset.sex)
label.fit(dataset.smoker.drop_duplicates())
dataset.smoker = label.transform(dataset.smoker)
label.fit(dataset.region.drop_duplicates())
dataset.region = label.transform(dataset.region)
print("Tipo de datos del dataset despues de la conversion:")
print(dataset.dtypes)

# Ver la correlacion entre las variables
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(dataset.corr(), annot=True, cmap='cool')

from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression

#Asignamos las variables y entrenamos el modelo 
x = dataset.drop(['charges'], axis = 1)
y = dataset['charges']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.intercept_)
print(lr.coef_)
print("Puntaje R2: ", lr.score(x_train, y_train))

#Visualizar la prediccion
y_pred = lr.predict(x_test)
df = pd.DataFrame({'Verdadero': y_test, 'Prediccion': y_pred})
print(df.head(10))


#Medir el MSE del modelo 
from sklearn.metrics import mean_squared_error
y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#Asignamos las variables y entrenamos el modelo 
x = dataset.drop(['charges'], axis = 1)
y = dataset['charges']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.intercept_)
print(lr.coef_)

print("\n")

#Visualizar la prediccion
y_pred = lr.predict(x_test)
df = pd.DataFrame({'Verdadero': y_test, 'Prediccion': y_pred})
print("Predicciones:")
print(df.head(25))

from sklearn.model_selection import learning_curve
# Define una función para trazar la curva de aprendizaje
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, 
                                                            scoring='neg_mean_squared_error')
    train_scores = -train_scores  
    test_scores = -test_scores    

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Validación")
    plt.xlabel("Tamaño del Conjunto de Entrenamiento")
    plt.ylabel("Error Cuadrático Medio")
    plt.title("Curva de Aprendizaje")
    plt.legend(loc="best")
    plt.show()

# Llama a la función para trazar la curva de aprendizaje
plot_learning_curve(lr, x_train, y_train)


print("\n")
print("\tMetricas del desempeno del modelo")
print("Puntaje R2: ", lr.score(x_train, y_train))
#Medir el MSE del modelo 
from sklearn.metrics import mean_squared_error
y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")

#Calular el MAE del modelo
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error:{mae:.3f}")


