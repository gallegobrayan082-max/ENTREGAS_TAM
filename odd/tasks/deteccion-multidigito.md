# Detección de varios dígitos en una sola imagen

**Objetivo**: extender el clasificador de MNIST del `scratchpad.ipynb` para que
reciba imágenes con uno o varios dígitos y devuelva **una detección por dígito**,
acompañada de una interfaz web donde se puedan subir las imágenes.

**Rama**: `feature/deteccion-multidigito` · **Estrategia de entrega**: PR única
(el trabajo cabe en el presupuesto de revisión) · **Runner de pruebas**: `pytest`

## Specs

- **S1** — "a partir de el construye un modelo que identifique numeros del 1 al 9".
  El modelo se construye sobre el MLP de Keras del `scratchpad.ipynb`
  (`Input(784) → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)`).
  Se conserva la arquitectura y se corrige el preprocesamiento: el notebook
  entrega los píxeles en `[0, 255]` al modelo, y aquí se normalizan a `[0, 1]`.
  El modelo se entrena, se evalúa sobre el conjunto de prueba y se guarda
  en disco para que la interfaz no dependa de reentrenar.
- **S2** — "pueden haber mas de un numero en la imagen por ejemplo 4 y 5
  (identifica ambos como 4 y 5 no como 45)". Una imagen con varios dígitos
  devuelve **una detección por dígito**, con su etiqueta, su confianza y su caja.
  Nunca se devuelve el número como un string concatenado.
- **S3** — "recuerda hacer una interfaz donde podamos subir las imagenes".
  Interfaz web (Flask) con carga de archivos por botón o arrastrar-y-soltar,
  que muestra la imagen original, las cajas detectadas sobre la imagen y una
  lista de dígitos individuales con su confianza.
- **S4** — La salida debe ser usable sin conexión y reproducible: modelo
  entrenado versionado en `modelos/`, semilla fija para el entrenamiento y
  pruebas que validan el comportamiento multi-dígito.

## Tasks

| ID | Spec | Ruta | Descripción | Estado |
|----|------|------|-------------|--------|
| T1 | S1, S4 | inline | `src/datos.py` (carga de MNIST) y `src/modelo.py` (MLP + clase `fondo` + entrenamiento) | ✅ |
| T2 | S2 | inline | `src/preprocesamiento.py` (gris, polaridad, Otsu, normalización MNIST) | ✅ |
| T3 | S2, S4 | inline | `src/deteccion.py` (bloques de tinta, ventanas, NMS greedy) | ✅ |
| T4 | S3 | inline | `src/app.py` + `src/templates/index.html` + `src/static/estilo.css` | ✅ |
| T5 | S2, S4 | inline | `pruebas/test_deteccion.py` + `pruebas/test_preprocesamiento.py` (20 pruebas) | ✅ |
| T6 | S1–S4 | inline | `README.md`, `requirements.txt`, `.gitignore`, cuaderno de la actividad | ✅ |
| T7 | S1–S4 | inline | `herramientas/` (generar ejemplos, evaluar, construir el cuaderno) | ✅ |

## Log

- **L1** (2026-11-10, verbatim): "revisa el notebook scratchpad.ipynb que esta en
  Documents/TAM/proyectos_clase y apartir de el construye un modelo que identifique
  numeros del 1 al 9 teniendo en cuenta que pueden haber mas de un numero en la
  imagen por ejemplo 4 y 5 (identifica ambos como 4 y 5 no como 45) recuerda hacer
  una interfaz donde podamos subir las imagenes"
- **L2** (2026-11-10): el `scratchpad.ipynb` ya está copiado en `ACTIVIDAD 1/`.
  Se trabaja sobre esa copia; el original de `proyectos_clase/` no se toca.
- **L3** (2026-11-10): el entorno tiene Python 3.14 y no existe wheel de
  TensorFlow para esa versión, así que el proyecto usa un venv propio con
  Python 3.13 (`uv venv --python 3.13 .venv`).
- **L4** (2026-11-10): convención del curso verificada en `TALLER 1/` y
  `EXAMEN 1/`: notas, docstrings e identificadores en español.
- **L5** (2026-11-10): la altura del dígito se estimaba sobre los componentes
  conexos y daba 10,5 px para un "5" de 20 px (ese dígito tiene varios
  componentes). Ahora se estima sobre los bloques de tinta.
- **L6** (2026-11-10): el modelo de diez clases no puede rechazar un recorte
  que no es un dígito, y lo hace con una confianza altísima (media barra de un
  4 → "1" con 99,9 %). Se añadió la clase 10 `fondo`, entrenada con recortes
  parciales y pares de dígitos pegados.
- **L7** (2026-11-10): los dígitos positivos y los ejemplos de `fondo` deben
  pasar por el mismo preprocesamiento y la misma variación aleatoria. Con los
  dígitos en crudo y los fragmentos normalizados, el modelo mandaba a `fondo`
  los 7 y los 2 recortados.
- **L8** (2026-11-10): el agrupamiento transitivo (unión-buscar) unía dos
  dígitos distintos a través de un candidato ancho que los cubría a los dos, y
  el detector reportaba un solo dígito. Se cambió por supresión de no máximos
  greedy: 4 y 5 se conservan los dos porque no se solapan entre sí.
- **L9** (2026-11-10): la rejilla de ventanas no llegaba al final del bloque,
  así que el último dígito de un grupo pegado nunca entraba en una ventana
  completa. Se añadió la posición alineada con el borde final.
- **L10** (2026-11-10): el usuario confirma el alcance: las imágenes son solo
  dígitos del 0 al 9, sin números como "11" o "99"; un "67" debe reportarse
  como un 6 y un 7 por separado. Verificado: "67" separado → `[6, 7]`.
- **L11** (2026-11-10): caso límite documentado — dígitos pegados sin ninguna
  separación (0 px entre ellos) producen detecciones extra. Es el peor caso
  del detector y queda anotado como limitación en el README.
