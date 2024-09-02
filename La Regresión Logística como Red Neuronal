# La Regresión Logística como Red Neuronal explicado de forma sencilla

## 1. ¿Qué se hace en la Regresión Logística?
En la regresión logística, queremos ajustar unos parámetros para que nuestro modelo haga predicciones lo más precisas posible. Imagina que estás ajustando los botones de un control para que el televisor muestre la mejor imagen. En este caso, los botones son nuestros parámetros (llamados pesos y sesgo), y queremos encontrar el ajuste perfecto.

## 2. ¿Cómo funcionan los Gráficos de Cáluclo?
Piensa en un gráfico de cálculo como un diagrama de flujo o un mapa que muestra cómo se calculan las cosas paso a paso:

- **Primero:** Calculamos un valor intermedio (llamado 𝑍) usando nuestros parámetros y datos de entrada.
- **Luego:** Usamos ese valor intermedio para obtener una predicción (llamada $\hat Y$).
- **Finalmente:** Comparamos la predicción con la respuesta correcta y calculamos cuánto nos equivocamos (esto es la pérdida).

## 3. ¿Cómo mejoramos el modelo?
Queremos ajustar nuestros parámetros (botones) para reducir ese error. Aquí es donde entran las derivadas, que nos dicen cómo cambiar cada parámetro para mejorar el modelo.

## 4. Pasos simples para Actualizar los Parámetros
**1. Calcula el error (Función de Coste):** Primero, calcula cuánto se equivocó el modelo. Esto es como ver cuánto te desviaste del objetivo.

**2. Calcula cómo cambiar los parámetros (Descenso de Gradiente):** Determina cómo ajustar cada parámetro para reducir el error.

**3. Actualiza los Parámetros:** Ajusta cada parámetro un poco basado en el cálculo anterior.

### Ejemplo Práctico
1. Predicción: Supongamos que tienes dos características (como el tamaño y el color de una manzana) y los parámetros que quieres ajustar son como la sensibilidad del sensor a esas características.

2. Comparar: Tu modelo predice si la manzana es roja o verde, pero la respuesta correcta es verde. El modelo se equivocó.

3. Ajustar:
    - Mira cuánto te equivocaste.
    - Ajusta la sensibilidad del sensor (parámetros) para que sea más preciso en la próxima predicción.

## 5. Implementación en el Código:
En la práctica, estos cálculos se hacen automáticamente usando herramientas de programación. Solo necesitas saber que estás ajustando los parámetros basándote en el error del modelo.

<p>&nbsp;</p><p>&nbsp;</p><p>&nbsp;</p>

# La Regresión Logística como Red Neuronal explicado de más extensa

## 1. Introducción a la Regresión Logística

La regresión logística es un algoritmo de aprendizaje supervisado utilizado para problemas de clasificación binaria, donde las etiquetas de salida $Y$ son 0 o 1. Por ejemplo, si queremos clasificar imágenes como "gato" o "no gato", el modelo debe predecir la probabilidad de que la imagen sea de un gato.

El objetivo de la regresión logística es predecir la probabilidad de que $Y$ sea igual a 1 dado un vector de características de entrada $X$. Esta probabilidad se denota como $\hat Y$, donde:

$\hat Y = P(Y = 1 ∣ X)$

El modelo de regresión logística se define como:

$\hat Y = \sigma(W^{t} X + b)$

Donde:

- $X$ es el vector de características de entrada.
- $W$ es el vector de pesos que asigna importancia a cada característica.
- $b$ es el sesgo o intercepto, que permite ajustar el modelo.
- $(W^{t} X + b)$ es la combinación lineal de las características y los parámetros.
- 𝜎(⋅) es la función sigmoide, que convierte el resultado de la combinación lineal en una probabilidad entre 0 y 1.


### Función Sigmoide

La función sigmoide es una función matemática que toma un número real $z$ y lo transforma en un valor entre 0 y 1. Es ideal para modelar probabilidades, ya que garantiza que el valor predicho siempre esté en este rango. Su fórmula es:

$\sigma (z) = \frac{1}{1+e^{-z}}$

Donde $e$ es la base del logaritmo natural (aproximadamente 2.718).

La función sigmoide tiene las siguientes propiedades:
- Cuando $z$ es grande, $\sigma (z)$ se aproxima a 1. 
- Cuando $z$ es pequeño (negativo), $\sigma (z)$ se aproxima a 0.
-  Cuando $z = 0$, $\sigma (z) = 0.5$ es decir, una probabilidad del 50%.


## 2. Función de coste de regresión logística 

### Función de Pérdida

Esta función mide qué tan bien están funcionando los parámetros $W$ (pesos) y $b$ (sesgo) en el conjunto de entrenamiento completo.

Para ello, utilizamos una función de pérdida que mide como de bien las predicciones $\hat Y$ se alinean con los valores reales $Y$ para un ejemplo específico. 

Su fórmula es la siguiente:

$L (\hat Y, Y) = \lfloor Y log(\hat Y) + (1 - Y) log(1 - \hat Y) \rfloor$

- Cuando $Y = 1$: La pérdida es $-log(\hat Y)$, lo que incentiva a que $\hat Y$ sea lo más cercano a 1 posible.

- Cuando $Y = 0$: La pérdida es $-log(1 - \hat Y)$, lo que incentiva a que $\hat Y$ sea lo más cercano a 1 posible.


### Función de Coste $J(W, b)$

Mientras que la función de pérdida mide el error en un solo ejemplo, la función de coste mide el error promedio en todo el conjunto de entrenamiento.

Para el conjunto completo de ejemplos de entrenamiento, la función de coste total $J(W,b)$ se calcula como el promedio de las pérdidas individuales de cada ejemplo:

$J(W,b) = - \frac{1}{m}  \sum_{i=1}^{m}\left [Y^{(i)} log(\hat Y^{(i)}) + (1 - Y^{(i)}) log(1 - \hat Y^{(i)})\right ]$

Donde $m$ es el número total de ejemplos en el entrenamiento.


## 3. Descenso de gradiente

El descenso de gradiente se emplea el objetivo de encontrar los valores de $W$ y $b$ que minimicen la función de costo $J(W, b)$. Es decir, encuentra los parámetros que hacen que las perdicciones del modelo sean lo más precisas posibles. 

Para ello, se empieza con valores iniciales para $W$ y $b$ (que pueden ser cero, por ejemplo) y se actualizan iterativamente tomando pequeños pasos en la dirección que reduce el valor de la función de costo.

### Convexidad de la Función de Costo

La función de costo en la regresión logística es convexa, lo que significa que tiene la forma de un cuenco.

Esto asegura que siempre habrá un único mínimo global (el punto más bajo del cuenco). Por lo tanto, el algoritmo de descenso por gradiente encontrará este mínimo global sin importar el punto de partida inicial.
