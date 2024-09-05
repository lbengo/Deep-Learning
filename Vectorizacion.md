# ¿Qué es la Vectorización?
La vectorización es una técnica que permite realizar operaciones en grandes cantidades de datos de una vez, en lugar de hacerlo elemento por elemento con bucles explicitos. La idea es aprovechar operaciones eficientes que las bibliotecas de programación, como NumPy en Python, ya tienen implementadas.

### ¿Por qué es Importante?

- **Velocidad:** La vectorización puede hacer que tu código sea mucho más rápido. Cuando trabajas con grandes conjuntos de datos, como en el deep learning, este aumento de velocidad es crucial. Los métodos vectorizados suelen ser cientos de veces más rápidos que los métodos que usan bucles explicitos.

- **Eficiencia:** Los cálculos vectorizados se realizan en bloques de datos, lo que permite a los procesadores (CPU o GPU) trabajar de manera más eficiente. Esto implica resultados más rápidos y con menos tiempo de espera.

### Ejemplo de Vectorización

Supongamos que quieres calcular una operación con dos vectores, $Z = W^{t} X + b$. Donde, $W$ y $X$ son vectores (por ejemplo, con 1 millón de elementos).

- **Sin Vectorización (con bucles explícitos):**
    ```python
    Z = 0
    for i in range(len(X)):
     Z += W[i] * X[i]
    Z += B
    ```

    Esto significa que estás usando un bucle para multiplicar cada elemento de $W$ con el elemento correspondiente de $X$ y luego sumando los resultados.


- **Con Vectorización:**
    ```python
    import numpy as np
    Z = np.dot(W, X) + B
    ```

    Aquí, 'np.dot' realiza el producto punto de $W$ y $X$ en una sola operación, y la suma del sesgo $B$ se hace directamente.


**Demostración de Velocidad**

Al comparar el tiempo que tarda en realizar el cálculo usando la vectorización frente a un bucle explícito se encontro que:

- Versión Vectorizada: ~1.5 milisegundos
- Versión con Bucle Explícito: ~500 milisegundos

La versión vectorizada es aproximadamente 300 veces más rápida que la versión con bucle explícito.

<p>&nbsp;</p><p>&nbsp;</p>

## Vectorización de la regresión logística

En la **regresión logística tradicional** se realizan cálculos **individualmente** para cada ejemplo de entrenamiento. Es decir, se calcula $z^{1} = w^{T} x^{1} + b$, $z^{2} = w^{T} x^{2} + b$, ...., $z^{m} = w^{T} x^{m} + b$. Y luego $a^{1} = \sigma(z^{1})$, $a^{2} = \sigma(z^{2})$, ...., $a^{m} = \sigma(z^{m})$, para cada muestra.

Sin embargo, con la **vectorización** se puede calcular $z$ y $a$ al mismo momento, empleando matrices sin la necesidad de bucles.

- **Cálculo de $z$:** para calcular $z$ en todos los ejemplo a la vez se aplica la siguiente operación.

```python
Z = np.dot(wT, x) + b
```

- **Cálculo de $a$:** una vez se tiene $z$, se aplica la función sigmoide para obtener $A$ en una sola operación. 

```python
A = sigmoid(Z)
```

## Vectorización de la regresión logística
