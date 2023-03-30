# Importar las bibliotecas necesarias
import numpy as np
from sklearn.linear_model import LogisticRegression

# Crear un conjunto de datos de entrenamiento
X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
y_train = np.array([0, 0, 1, 1])

# Crear el modelo y entrenarlo
model = LogisticRegression()
model.fit(X_train, y_train)

# Crear un conjunto de datos de prueba
X_test = np.array([[0, 1], [1, 0], [2, 3]])

# Hacer predicciones con el modelo
predictions = model.predict(X_test)

# Imprimir las predicciones
print(predictions)
