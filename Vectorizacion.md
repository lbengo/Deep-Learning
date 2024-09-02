# ¿Qué es la Vectorización?
La vectorización es una técnica que permite realizar operaciones en grandes cantidades de datos de una vez, en lugar de hacerlo elemento por elemento con bucles explícitos (es decir, usando for loops). La idea es aprovechar operaciones eficientes que las bibliotecas de programación, como NumPy en Python, ya tienen implementadas.

¿Por qué es Importante?
Velocidad: La vectorización puede hacer que tu código sea mucho más rápido. Cuando trabajas con grandes conjuntos de datos, como en el aprendizaje profundo, este aumento de velocidad es crucial. Los métodos vectorizados suelen ser cientos de veces más rápidos que los métodos que usan bucles explícitos.

Eficiencia: Los cálculos vectorizados se realizan en bloques de datos, lo que permite a los procesadores (CPU o GPU) trabajar de manera más eficiente. Esto significa que puedes obtener resultados más rápido y con menos tiempo de espera.

Ejemplo de Vectorización
Supongamos que quieres calcular una operación en dos vectores:

Operación: $Z = W^{t} X + b$

Aquí:
- $W$ y $X$ son vectores (por ejemplo, con 1 millón de elementos).
- $W^{t} \cdot X$ es el producto punto entre $W$ y $X$.
- $B$ es un sesgo que se suma al resultado.

### **Sin Vectorización (con bucles explícitos):**
```python
Z = 0
for i in range(len(X)):
    Z += W[i] * X[i]
Z += B
```

Esto significa que estás usando un bucle para multiplicar cada elemento de $W$ con el elemento correspondiente de $X$ y luego sumando los resultados.

### **Con Vectorización:**
```python
import numpy as np
Z = np.dot(W, X) + B
```

Aquí, 'np.dot' realiza el producto punto de $W$ y $X$ en una sola operación, y la suma del sesgo $B$ se hace directamente.


### Demostración de Velocidad
En una demostración, se comparó el tiempo que tarda en realizar el cálculo usando la vectorización frente a un bucle explícito. Aquí está lo que se encontró:

- Versión Vectorizada: ~1.5 milisegundos
- Versión con Bucle Explícito: ~500 milisegundos

La versión vectorizada es aproximadamente 300 veces más rápida que la versión con bucle explícito.

### ¿Por Qué es Tan Rápido?
Las operaciones vectorizadas pueden ser procesadas de manera más eficiente por la CPU o la GPU, ya que estas máquinas pueden manejar múltiples datos en paralelo.

- CPU: Utiliza instrucciones especiales para realizar operaciones en múltiples datos a la vez.
- GPU: Está diseñada específicamente para manejar grandes cantidades de operaciones paralelas, lo que hace que sea aún más rápida en operaciones vectorizadas.

### Regla General
Siempre que sea posible, evita usar bucles explícitos y opta por **operaciones vectorizadas**. Esto hará que tu código sea mucho más rápido y eficiente.

<p>&nbsp;</p><p>&nbsp;</p>

# Vectorización Paso a Paso

## Propagación Hacia Adelante (Forwdard Propagation)
Calcula las predicciones del modelo

### 1. Definición de la Matriz de Entrenamiento $X$
Supongamos que tienes una matriz $X$ que contiene todos los ejemplos de entrenamiento. Cada columna de $X$ representa un ejemplo, y cada fila representa una característica. Así que, si tienes $N$ características y $M$ ejemplos, $X$ sería una matriz de tamaño $N x M$.

### 2. Calcular $Z$ para Todos los Ejemplos
En lugar de calcular $Z$ para cada ejemplo por separado, puedes calcularlo para todos los ejemplos de una vez usando:

$Z = W^{t} X + B$

Aquí:
- $W^{t}$ es el vector de pesos transpuesto (ahora es un vector fila).
- $X$ es la matriz de ejemplos.
- $B$ es el sesgo que se suma a cada cálculo. En Python, el sesgo $B$ se "transmite" automáticamente a lo largo de todas las columnas de la matriz, un proceso conocido como broadcasting.

En Python, esto se vería así:
```python
Z = np.dot(W.T, X) + B
```

Esto calcula todos los $Z$ en un solo paso. $Z$ es una matriz donde cada columna representa el $Z$ para un ejemplo.

### 3. Calcular $\hat P$ para Todos los Ejemplos
Una vez que tienes $Z$, necesitas aplicar la función sigmoide para obtener $\hat P$. Puedes hacer esto para todos los ejemplos de una vez:
```python
A = sigmoid(Z)
```
Aquí, sigmoid es una función que aplica la transformación sigmoide a cada elemento de $Z$.


## Propagación Hacia Atrás (Backpropagation)
Calcula los gradientes del error con respecto a los parámetros del modelo para poder actualizar los parámetros.

### 1. Cálculo de los Gradientes del Error

#### a. Calcular el Gradiente del Error con Respecto a las Predicciones ($d\hatY$)

La derivada de la función de error con respecto a las probabilidades ($\hat Y$) es simplemente la diferencia entre las probabilidades y el prámetro verdadero.

$d\hat Y = \hat Y -Y$

En phyton 
```python
dA = A - Y
```

Aquí, $dA$ es una matriz donde cada columna representa el gradiente del error para un ejemplo.
