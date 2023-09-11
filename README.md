# AI_Data_Science

### :arrow_right: Contenido del repositorio 
El repositorio consta de los siguientes archivos y carpetas:
_Carpetas_:
- Framework_LinearRegression: Esta carpeta contiene todos los archivos relevantes hacia la implementación de regresión lineal
- No_Framework_LogisticRegression: Esta carpeta contiene todos los archivos relevantes hacia la implementación de regresión logistica
- _Archivos_: 
- `LogisticRegression.py`: Este archivo contiene la implementación del algoritmo de Regresión Logística desde cero en Python.
- `LinearRegression.py`: Este archivo contiene la implementación del algoritmo de Regresión Lineal con el uso de un framework. Asimismo, dentro del código se encuentran algunas métricas para evaluar el modelo.

## **Implementación de una técnica de aprendizaje máquina sin el uso de un framework.**

### :ledger: Instrucciones de Uso 
- El archivo que se debe correr es: [LogisticRegression.py](LogisticRegression.py)

### Dataset :information_source:
El conjunto de datos utilizado en este proyecto es ***load_breast_cancer de scikit-learn***. Este conjunto de datos contiene características relacionadas con el diagnóstico de cáncer de mama, con el objetivo de predecir si un tumor es maligno o benigno.

### Objetivo :dart:
La Regresión Logística se utiliza en este contexto para **predecir si un tumor es _maligno (cáncer)_ o _benigno (no canceroso)_** en función de ciertas características y características clínicas relacionadas con el tumor, como el tamaño, la textura y otros atributos medidos a partir de imágenes médicas. 

> Este tipo de cáncer es una de las principales causas de muerte entre las mujeres en todo el mundo, y la detección temprana es esencial para un tratamiento efectivo y una mayor tasa de supervivencia.

### Resultados y Evaluación :chart: 
En el archivo `LogisticRegression.py`, se incluyen métricas de evaluación como la precisión, el puntaje F1 y la matriz de confusión para evaluar el rendimiento del modelo de Regresión Logística en el conjunto de datos de cáncer de mama. Estas métricas proporcionan información sobre la capacidad del modelo para hacer predicciones precisas.

### Conclusiones :triangular_flag_on_post:
Este proyecto demuestra la capacidad de implementar manualmente un algoritmo de Regresión Logística sin depender de bibliotecas o frameworks de aprendizaje automático. 

## Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.
### :ledger: Instrucciones de Uso 
- El archivo que se debe correr es: [LinearRegression.py](LinearRegression.py)

# Dataset :information_source:
- **Nombre del Dataset:** insurance.csv
- **Variables:**
  - **Edad:** Edad del beneficiario principal.
  - **Sexo:** Sexo del contratante del seguro (mujer, hombre).
  - **BMI (Índice de Masa Corporal):** 
  - **Niños:** Número de hijos cubiertos por el seguro de enfermedad o número de personas a cargo.
  - **Fumador:** Indica si la persona es fumadora.
  - **Región:** La zona de residencia del beneficiario en los EE.UU. (noreste, sureste, suroeste, noroeste).
  - **Gastos (Objetivo):** Gastos médicos individuales facturados por el seguro de enfermedad.

# Objetivo :dart:
El objetivo de este análisis es comprender cómo las diferentes variables, como la edad, el sexo, el BMI, el número de hijos, el hábito de fumar y la región de residencia, afectan los gastos médicos individuales facturados por el seguro de enfermedad.

# Resultados y Evaluación :chart:
- Se utilizó un modelo de regresión lineal para predecir los gastos médicos en función de las variables independientes mencionadas.
- Se obtuvieron métricas de evaluación, como el coeficiente de determinación (R²), el error cuadrático medio (MSE) y el error absoluto medio (MAE), para evaluar el desempeño del modelo.

# Conclusiones :triangular_flag_on_post:
- El modelo de regresión lineal sugiere que las variables como la edad, el BMI y el hábito de fumar tienen un impacto significativo en los gastos médicos. Por ejemplo, se observa que las personas más jóvenes tienden a tener gastos médicos más bajos en promedio.
- El sexo y la región de residencia también pueden influir en los gastos médicos, aunque su impacto puede ser menor en comparación con otras variables.
- Estos resultados pueden ser útiles para comprender cómo diferentes factores afectan los costos del seguro médico y pueden informar decisiones relacionadas con políticas de precios y estrategias comerciales en la industria de seguros de salud.






