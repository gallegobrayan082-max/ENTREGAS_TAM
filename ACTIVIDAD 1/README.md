# Actividad 1 · Detección de varios dígitos en una imagen

Extiende el clasificador de MNIST del `scratchpad.ipynb` para que reciba una
imagen con **uno o varios dígitos escritos a mano** y devuelva **una detección
por dígito**, con su etiqueta, su confianza y su caja.

Una imagen con un 4 y un 5 devuelve **dos detecciones**: un 4 y un 5. Nunca un
número "45" del que no se pueda saber dónde acaba cada dígito.

Incluye una interfaz web donde se suben las imágenes y se ve el resultado
dibujado sobre la imagen original.

---

## 1. Puesta en marcha

```bash
cd "ACTIVIDAD 1"

uv venv
uv pip install --python .venv/bin/python -r requirements.txt
```

No hace falta pedir la versión de Python: el archivo `.python-version` fija la
**3.13** y `uv` la usa sola. El pin está porque TensorFlow no publica ruedas
para Python 3.14, que es el que trae Fedora 43 de fábrica.

Si usas `pyenv` en vez de `uv`, el mismo archivo le sirve: `pyenv` también lo
lee. Con `pip` a pelo:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Entrenar el modelo

```bash
.venv/bin/python -m src.modelo
```

Descarga MNIST (la primera vez), entrena el MLP y guarda el modelo en
`modelos/mnist_mlp.keras`. Tarda unos cuatro minutos en CPU.

## 3. Levantar el dashboard

```bash
.venv/bin/streamlit run src/app_streamlit.py
```

Y abrir <http://127.0.0.1:8501>. El dashboard trae:

- **Indicadores**: cuántos dígitos se detectaron, la confianza media, el
  tiempo que tardó y el tamaño de la imagen.
- **La imagen anotada**, con una caja por dígito.
- **La tabla de detecciones**: número, dígito, confianza, caja y centro.
- **Modo por lotes**: subiendo varias imágenes sale una tabla resumen con una
  fila por imagen.
- **El selector de modelo** en la barra lateral (ver más abajo).

### Cambiar de modelo

Los modelos se leen de la carpeta `modelos/`, así que **cambiar de modelo es
dejar el archivo ahí**: aparece en el desplegable y se puede usar desde la
propia página, sin tocar código.

Cada modelo va acompañado de una ficha `modelos/<nombre>.json` con su
exactitud en prueba, su pérdida, las épocas y la fecha de entrenamiento; esa
ficha es la que se muestra en el panel lateral.

El detector se adapta solo al modelo: mira la forma de entrada y aplana el
lote a 784 para un modelo denso, o lo deja como imagen de 28x28 (con canal, si
hace falta) para uno convolucional. La clase `fondo` se toma de la salida del
modelo, así que tampoco hace falta tocar nada.

### Los dos modelos

| Modelo | Arquitectura | Entrada | Entrenarlo |
|---|---|---|---|
| `mnist_mlp.keras` | `784 → 64 → 64 → 11` (el del `scratchpad.ipynb`) | vector de 784 | `python -m src.modelo` |
| `mnist_cnn.keras` | `Conv2D(32) → Conv2D(64) → Flatten → 11` (la del `scratchpad_2.ipynb`) | imagen de 28×28×1 | `python -m src.modelo convolucional` |

Los dos se entrenan con el mismo pipeline (los dígitos de MNIST normalizados
como los ve el detector, más los ejemplos de la clase `fondo`, con la misma
variación aleatoria), así que la comparación en el dashboard es justa.

Los números, medidos sobre el mismo conjunto de prueba:

| Modelo | Tipo | Exactitud (dígito recortado) | Mandados a `fondo` | Ejemplos | Tiempo por imagen |
|---|---|---|---|---|---|
| **CNN** | convolucional | **96,77 %** | **0,00 %** | 8/8 | 85 ms |
| MLP | denso | 93,93 % | 1,10 % | 8/8 | 93 ms |

La CNN gana por dos márgenes. Primero, clasifica mejor: **+2,8 puntos**. Y
segundo, y más interesante: **no rechaza ningún dígito real**. La red densa
manda a `fondo` el 1,1 % de los dígitos, porque un recorte que se sale un
píxel de la ventana ya no le encaja; la CNN, que detecta el trazo donde esté,
reconoce el dígito igual y no lo descarta nunca.

El MLP es una red densa: memoriza la posición exacta de los píxeles. La CNN
aprende filtros que detectan el trazo donde esté, y por eso generaliza mejor
a los recortes parciales y a las fotos con ruido. Es también la que va a
usar de verdad el fondo negro: con una representación densa, los ceros del
fondo no ahorran ni una multiplicación.

> En un portátil con 2 núcleos la CNN tarda unos 3 minutos por época. En
> Colab o con GPU, segundos.

### La otra interfaz

También hay una versión con **Flask**, más simple y con API JSON, por si
hace falta consultarla desde otro programa:

```bash
.venv/bin/python -m src.app
```

en <http://127.0.0.1:5000>, con `POST /predecir` devolviendo **una detección
por dígito** en JSON:

```json
{"detecciones": [{"digito": 4, "confianza": 1.0, "caja": [21, 15, 63, 75]},
                 {"digito": 5, "confianza": 1.0, "caja": [105, 12, 165, 72]}]}
```

## 4. Pruebas

```bash
.venv/bin/python -m pytest pruebas/ -q
```

Y para medir el modelo y el detector sobre el conjunto de prueba y sobre las
imágenes de ejemplo:

```bash
.venv/bin/python -m herramientas.evaluar
```

---

## Qué se cambió respecto al `scratchpad.ipynb`

El notebook ya tenía la arquitectura correcta:

```
Input(784) → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)
```

Se conserva tal cual, y se corrigieron tres cosas:

**1. El preprocesamiento.** `tf.keras.datasets.mnist.load_data()` devuelve los
píxeles como enteros en `[0, 255]` y el notebook los entregaba así al modelo.
Aquí se normalizan a `[0, 1]`.

**2. La evaluación.** El notebook entrenaba 30 épocas y no miraba el conjunto
de prueba. Aquí se separa un 10 % para validar y se evalúa sobre el conjunto
de prueba.

**3. Una clase extra: `fondo`.** El notebook tenía diez salidas, una por
dígito. El detector necesita una más (ver más abajo).

## Cómo funciona la detección

El clasificador de MNIST solo sabe responder una cosa: *este parche de 28x28 es
un 4*. Para una imagen con varios dígitos hacen falta tres etapas, y cada una
usa la anterior.

### 1. Proponer candidatos: dónde puede haber un dígito

La imagen se pasa a escala de grises, se decide su polaridad (si el trazo es
oscuro sobre fondo claro, se invierte) y se binariza con el **método de Otsu**.

La tinta se divide en **bloques**: se separa por columnas vacías y por filas
vacías. Dos dígitos uno al lado del otro quedan en columnas distintas, así que
quedan en bloques distintos aunque la tinta llegue a tocarse.

De cada bloque se revisa su forma:

- si no es más ancho que alto y no es una mota, **el bloque es un candidato**;
- si es demasiado ancho o alto, puede esconder varios dígitos, y entonces se
  recorre con una **ventana deslizante** que busca un dígito por ventana.

### 2. Clasificar cada candidato

El recorte se normaliza al formato de MNIST (recortar la tinta, escalar para
que quepa en 20x20, centrar en el lienzo de 28x28) y se pasa al modelo, que
devuelve la distribución de probabilidad de las once clases.

Al final de la normalización el fondo se deja en **negro** y el trazo en
**blanco**: el recorte viene de una foto donde el papel es gris, y el modelo se
entrenó con el fondo en negro.

Los bordes grises de los trazos se conservan a propósito: si se pasa todo a
binario, el modelo pierde **1,8 puntos** de exactitud (93,9 % → 92,0 %
medido), porque en MNIST los bordes de los trazos también tienen gris.

> **Sobre "consume menos recursos".** Con un modelo **denso** el fondo en
> negro no ahorra ni una multiplicación: los ceros se multiplican igual y los
> FLOPs son los mismos. Lo que sí mejora es que la entrada se parezca a la
> del entrenamiento, y de ahí sale la ganancia en exactitud. Con un modelo
> convolucional, o con una representación dispersa, los ceros sí se pueden
> aprovechar.

### 3. Una detección por dígito

Los candidatos que se solapan son el mismo dígito visto de distintas maneras.
Se recorren de mayor a menor confianza y se conserva el primero de cada grupo:
es la **supresión de no máximos** de la detección de objetos.

El recorrido es *greedy* y no transitivo a propósito. Con un agrupamiento por
componentes conexas, un candidato ancho que abarca dos dígitos los unía en el
mismo grupo y el detector reportaba un solo dígito. Con el recorrido greedy, el
4 y el 5 se conservan los dos porque no se solapan entre sí.

## La clase `fondo`: por qué hace falta

Esta es la parte menos obvia de la actividad, y la que más rendimiento aporta.

Un modelo de diez clases no tiene forma de decir "aquí no hay un dígito". Y
responde con una seguridad aparente: **media barra de un 4 se parece
muchísimo a un 1**, así que el detector recibe una ventana que corta el 4 por
la mitad, la clasifica como "1" con el 99,9 % de confianza y reporta un 1 donde
no hay ningún 1.

La solución es una clase extra, la número 10, llamada `fondo`, entrenada con
tres tipos de ejemplos que el detector se encuentra en la práctica:

| Tipo de ejemplo | Qué simula |
|---|---|
| un trozo de un dígito | una ventana que corta el trazo por la mitad |
| dos dígitos en el mismo recorte | una ventana más ancha que un dígito |
| un recorte casi vacío | ruido en la imagen |

Los dígitos positivos y los ejemplos de `fondo` pasan por **el mismo
preprocesamiento y la misma variación aleatoria** (rotación, escala, posición
y ruido). Eso es importante: si solo variaran los dígitos, el modelo aprendería
"forma alterada = dígito" y mandaría a `fondo` cualquier recorte con una forma
un poco distinta, que es exactamente lo que pasaba con los 7 y los 2.

## Estructura

```
ACTIVIDAD 1/
├── scratchpad.ipynb          el notebook original, sin tocar
├── src/
│   ├── datos.py              carga de MNIST (Keras, con respaldo a los IDX)
│   ├── preprocesamiento.py   grises, polaridad, Otsu, normalización a 28x28
│   ├── modelo.py             el MLP, la clase `fondo` y el entrenamiento
│   ├── deteccion.py          candidatos, clasificación y no máximos
│   ├── app_streamlit.py      el dashboard (Streamlit), con selector de modelo
│   ├── app.py                la interfaz web (Flask) y su API JSON
│   ├── templates/index.html  la página de la interfaz de Flask
│   └── static/estilo.css
├── modelos/
│   ├── mnist_mlp.keras       el modelo entrenado
│   └── mnist_mlp.json        su ficha (exactitud, épocas, fecha)
├── herramientas/
│   ├── generar_ejemplos.py   crea imágenes de prueba en `ejemplos/`
│   └── evaluar.py            métricas del modelo y del detector
├── pruebas/                  pruebas del preprocesamiento y del detector
├── modelos/mnist_mlp.keras   el modelo entrenado
└── ejemplos/                 imágenes de ejemplo
```

## Limitaciones y por dónde seguir

- **Varios dígitos pegados entre sí.** Con dos dígitos que se tocan (un "4" y
  un "5" sin hueco entre ellos) el detector acierta: salen un 4 y un 5. Con
  cuatro dígitos pegados en cadena, ya no: aparecen detecciones de más. Es el
  caso más difícil, porque no hay separación posible por columnas y la ventana
  deslizante tiene que adivinar dónde cortar.
- **El modelo es un MLP.** Con una red convolucional el detector mejoraría,
  sobre todo en los recortes parciales: las convoluciones aprenden a detectar el trazo
  donde esté, en vez de memorizar la posición exacta de los píxeles.
- **Un solo dígito por fila.** Los dígitos apilados en vertical se detectan
  como un bloque único.
- **Fondo con textura.** El detector asume que la imagen es el dígito y nada
  más; un fondo con ruido fuerte ensucia las cajas.

## Resultados

Con el modelo entrenado (`modelos/mnist_mlp.keras`, 40 épocas, semilla 2026):

| Métrica | Valor |
|---|---|
| Exactitud sobre el conjunto de prueba de MNIST | 93,56 % |
| Exactitud sobre el dígito **recortado** (como lo ve el detector) | 95,02 % |
| Dígitos perdidos (mandados a la clase `fondo`) | 1,10 % |
| Imágenes de ejemplo detectadas correctamente | 8 de 8 |

El modelo de diez clases del `scratchpad` llega al 97,5 % de exactitud, pero no
puede rechazar un recorte que no es un dígito: ese precio es el que paga la
clase `fondo`, y con él el detector pasa de reportar basura con total aplomo a
no perder dígitos.
