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
# Divide el conjunto de datos en entrenamiento (70%) y prueba (30%)
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.3, random_state=0)

# Divide el conjunto de prueba (30%) en prueba (15%) y validación (15%)
x_test, x_val, y_test, y_val = holdout(x_test, y_test, test_size=0.5, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.intercept_)
print(lr.coef_)

print("\n")

#Prediccion con el conjunto de prueba
y_pred = lr.predict(x_test)
df = pd.DataFrame({'Verdadero': y_test, 'Prediccion': y_pred})
print("Predicciones con el conjunto de test o prueba:")
print(df.head(15))

#Predicción con el conjunto de validación
y_pred_val = lr.predict(x_val)
df = pd.DataFrame({'Verdadero': y_val, 'Prediccion': y_pred_val})
print("Predicciones con el conjunto de validación:")
print(df.head(15))

from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define una función para trazar la curva de aprendizaje para train, test y validación
def plot_learning_curve(estimator, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True)
    
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
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Prueba")
    plt.xlabel("Tamaño del Conjunto de Entrenamiento")
    plt.ylabel("Error Cuadrático Medio")
    plt.title("Curva de Aprendizaje")
    plt.legend(loc="best")
    plt.show()

# Llama a la función para trazar la curva de aprendizaje
plot_learning_curve(lr, x_train, y_train)

# Métricas del desempeño del modelo en el conjunto de test
r2_test = lr.score(x_test, y_test)
mse_test = mean_squared_error(y_test, y_pred)
mae_test= mean_absolute_error(y_test, y_pred)

# Métricas del desempeño del modelo en el conjunto de validación
r2_val = lr.score(x_val, y_val)
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)


print("\n")
print("\tMétricas del desempeño del modelo en el conjunto de test o pruebas")
print(f"Puntaje R2 (Validación): {r2_test:.3f}")
print(f"Mean Squared Error (Validación): {mse_test:.3f}")
print(f"Mean Absolute Error (Validación): {mae_test:.3f}")

print("\n")
print("\tMétricas del desempeño del modelo en el conjunto de validación")
print(f"Puntaje R2 (Validación): {r2_val:.3f}")
print(f"Mean Squared Error (Validación): {mse_val:.3f}")
print(f"Mean Absolute Error (Validación): {mae_val:.3f}")
