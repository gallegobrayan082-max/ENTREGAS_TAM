"""El modelo: el MLP del `scratchpad.ipynb`, entrenado y guardado en disco.

El notebook original ya tenía la arquitectura correcta:

    Input(784) → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)

Lo que faltaba era el preprocesamiento: `tf.keras.datasets.mnist.load_data()`
devuelve los píxeles como enteros en `[0, 255]`, y el notebook los entregaba
así al modelo. Aquí se normalizan a `[0, 1]`, que es la escala con la que el
modelo converge mejor y la que usa el preprocesamiento de la imagen en la
interfaz. También se separa un 10 % de entrenamiento para validar, y se
evalúa sobre el conjunto de prueba, que en el notebook no se miraba.

**La clase extra: `fondo`.** El detector recorta regiones de una imagen y se
las pasa al modelo. Muchas de esas regiones no contienen un dígito completo:
son trozos de un trazo (una ventana que corta el "4" por la mitad) o dos
dígitos pegados. Un modelo de diez clases no tiene forma de decir "aquí no hay
un dígito", y responde con una confianza altísima: media barra de un 4 se
parece muchísimo a un 1, y el modelo dice "1" con el 99,9 % de seguridad. Con
esa seguridad, el detector reporta basura con total aplomo.

Por eso el modelo tiene once salidas: los dígitos del 0 al 9 y una clase
`fondo` (10) entrenada con recortes parciales y con pares de dígitos pegados.
El detector descarta lo que caiga en esa clase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

RUTA_MODELO = Path(__file__).resolve().parent.parent / "modelos" / "mnist_mlp.keras"
"""Dónde se guarda el modelo entrenado."""

CARPETA_MODELOS = RUTA_MODELO.parent
"""Carpeta donde se guardan los modelos y sus fichas."""

SEMILLA = 2026
EPOQUES = 40
LOTE = 32
VALIDACION = 0.1

CLASE_FONDO = 10
"""Clase que significa "aquí no hay un dígito completo"."""

NUMERO_DE_CLASES = 11
NUMERO_DE_DIGITOS = 10
"""Las clases 0 a 9 son dígitos; la 10 es la clase `fondo`."""
CANTIDAD_DE_FONDO = 45000
"""Cuántos ejemplos de la clase `fondo` se generan (uno por cada dígito real)."""

ROTACION = (-6.0, 6.0)
"""Rango de rotación (grados) de la variación aleatoria."""

ESCALA = (0.93, 1.08)
"""Rango de escala de la variación aleatoria."""

RUIDO_MAXIMO = 0.02
"""Desviación máxima del ruido gaussiano de la variación aleatoria."""

BRILLO = (0.80, 1.0)
"""Rango del factor de brillo de la variación aleatoria."""

NIVEL_FONDO = (0.0, 0.10)
"""Rango del nivel de gris del fondo.

MNIST tiene el fondo en negro puro. Un recorte que viene de una foto lo tiene
en gris, porque el papel no es negro. Entrenar solo con el fondo en negro hace
que el modelo sea frágil: basta con un nivel de gris distinto para que un 3
se lea como un 8.
"""


def construir_modelo():
    """Arma el MLP de tres capas, idéntico al del `scratchpad.ipynb`.

    La única diferencia es la última capa: `NUMERO_DE_CLASES` salidas en vez
    de 10, para poder decir "esto no es un dígito".
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    entrada = layers.Input(shape=(784,), name="entrada")
    cp1 = layers.Dense(64, activation="relu", name="cp1")(entrada)
    cp2 = layers.Dense(64, activation="relu", name="cp2")(cp1)
    salida = layers.Dense(NUMERO_DE_CLASES, activation="softmax", name="salida")(cp2)
    return keras.Model(inputs=entrada, outputs=salida, name="mnist_mlp")


def preprocesar(imagenes: np.ndarray) -> np.ndarray:
    """Pasa las imágenes de MNIST a float32 en `[0, 1]` y las aplana a 784."""
    return (imagenes.astype("float32") / 255.0).reshape(-1, 784)


def _variar_una(imagen: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica una variación aleatoria a una imagen de 28x28.

    Rota un poco, la cambia de escala, la desplaza y le mete ruido. Es lo que
    hace que el modelo tolere las diferencias entre un dígito de MNIST y el
    recorte de ese mismo dígito dentro de una foto: el detector recorta, vuelve
    a escalar y centra, y sin este entrenamiento el modelo confundía un "7"
    recortado con un fragmento de trazo y lo mandaba a la clase `fondo`.
    """
    from PIL import Image

    angulo = float(rng.uniform(*ROTACION))
    escala = float(rng.uniform(*ESCALA))
    brillo = float(rng.uniform(*BRILLO))
    fondo = float(rng.uniform(*NIVEL_FONDO))
    ruido = rng.normal(0, rng.uniform(0.0, RUIDO_MAXIMO), (28, 28))

    lienzo = Image.new("L", (28, 28), color=0)
    lado = max(8, int(round(28 * escala)))
    # `Image.fromarray` con un array de float crearía una imagen de punto
    # flotante de 32 bits, no una imagen de grises: hay que pasar a uint8.
    grande = Image.fromarray((np.clip(imagen, 0, 1) * 255).astype(np.uint8))
    grande = grande.resize((lado, lado), Image.BILINEAR)
    grande = grande.rotate(angulo, resample=Image.BILINEAR, fillcolor=0)
    desplazamiento_x = int(rng.integers(-2, 3))
    desplazamiento_y = int(rng.integers(-2, 3))
    lienzo.paste(
        grande,
        ((28 - lado) // 2 + desplazamiento_x, (28 - lado) // 2 + desplazamiento_y),
    )
    salida = np.asarray(lienzo, dtype=np.float32) / 255.0 * brillo
    # el fondo gris de una foto: el papel no es negro
    salida = fondo + (1.0 - fondo) * salida
    salida = salida + ruido
    return np.clip(salida, 0.0, 1.0)


def _variar(imagenes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica `_variar_una` a cada imagen de un lote."""
    variadas = np.zeros((len(imagenes), 28, 28), dtype=np.float32)
    for i, imagen in enumerate(imagenes):
        variadas[i] = _variar_una(imagen, rng)
    return variadas


def _normalizar_como_mnist(imagen: np.ndarray) -> np.ndarray:
    """Deja una imagen (uint8) en el formato que ve el modelo: 28x28 en `[0, 1]`.

    Recorta la tinta, la escala para que quepa en 20x20 y la centra, igual que
    hace el preprocesamiento en tiempo de ejecución.
    """
    from . import preprocesamiento as prep

    gris = imagen.astype(np.float32) / 255.0
    caja = prep.caja_de_tinta(gris > 0.5)
    if caja is None:
        return np.zeros((28, 28), dtype=np.float32)
    return prep.normalizar_mnist(gris, caja)


def _trozo_de_digito(digito: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Recorta una parte de un dígito: el recorte de una ventana que lo corta."""
    filas, columnas = np.nonzero(digito > 127)
    if filas.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    y0, y1 = int(filas.min()), int(filas.max()) + 1
    x0, x1 = int(columnas.min()), int(columnas.max()) + 1

    alto = max(3, int(round((y1 - y0) * rng.uniform(0.15, 0.45))))
    ancho = max(3, int(round((x1 - x0) * rng.uniform(0.15, 0.45))))
    fy = int(rng.integers(y0, max(y0 + 1, y1 - alto + 1)))
    fx = int(rng.integers(x0, max(x0 + 1, x1 - ancho + 1)))
    return digito[fy : fy + alto, fx : fx + ancho]


def _dos_digitos_juntos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pega dos dígitos uno al lado del otro, como en un recorte que agarra dos."""
    lienzo = np.zeros((28, 56), dtype=np.uint8)
    lienzo[:, :28] = a
    lienzo[:, 28:] = b
    return lienzo


def generar_ejemplos_de_fondo(
    x: np.ndarray, rng: np.random.Generator, cantidad: int = CANTIDAD_DE_FONDO
) -> np.ndarray:
    """Construye los ejemplos de la clase `fondo`.

    Son las tres situaciones que el detector se encuentra en la práctica:

    1. **un trozo de un dígito**: una ventana que corta el trazo por la mitad.
       Es el caso más importante, porque el modelo sin esta clase confunde
       un fragmento con un 1 y lo reporta con el 99,9 % de confianza.
    2. **dos dígitos en el mismo recorte**: la ventana es más ancha que un
       dígito y agarra parte del siguiente.
    3. **un recorte vacío**: apenas hay tinta.
    """
    ejemplos = np.zeros((cantidad, 28, 28), dtype=np.uint8)
    relleno = 0
    intentos = 0
    while relleno < cantidad and intentos < cantidad * 3:
        intentos += 1
        modo = rng.integers(3)
        if modo == 0:
            trozo = _trozo_de_digito(x[rng.integers(len(x))], rng)
        elif modo == 1:
            trozo = _dos_digitos_juntos(
                x[rng.integers(len(x))], x[rng.integers(len(x))]
            )
        else:
            # recorte con muy poca tinta: una esquina de una imagen
            digito = x[rng.integers(len(x))]
            filas, columnas = np.nonzero(digito > 127)
            if filas.size == 0:
                continue
            alto = int(rng.integers(4, 10))
            ancho = int(rng.integers(4, 10))
            fy = int(rng.integers(0, max(1, 28 - alto)))
            fx = int(rng.integers(0, max(1, 28 - ancho)))
            trozo = digito[fy : fy + alto, fx : fx + ancho]
        if trozo.size == 0:
            continue
        ejemplos[relleno] = (_normalizar_como_mnist(trozo) * 255).astype(np.uint8)
        relleno += 1
    return ejemplos[:relleno]


def _preparar_digitos(x: np.ndarray) -> np.ndarray:
    """Pasa los dígitos de MNIST por el mismo preprocesamiento del detector.

    Esto es lo más importante del entrenamiento: los dígitos positivos y los
    ejemplos de `fondo` tienen que verse igual de "recortados". Si los
    dígitos se entrenan tal cual vienen del dataset y los fragmentos ya
    normalizados, el modelo aprende que "forma recortada = fondo" y manda a
    `fondo` cualquier dígito real que el detector recorte de una foto.
    """
    recortes = np.zeros((len(x), 28, 28), dtype=np.float32)
    for i, imagen in enumerate(x):
        recortes[i] = _normalizar_como_mnist(imagen)
    return recortes


def _aumentar_con_fondo(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Prepara el conjunto de entrenamiento.

    Se juntan los dígitos (normalizados como los ve el detector) con los
    ejemplos de la clase `fondo`, y a todos se les aplica la misma variación
    aleatoria. La variación tiene que ser la misma en las dos clases: si solo
    variaran los dígitos, el modelo aprendería "forma alterada = dígito" y
    mandaría a `fondo` cualquier recorte con una forma un poco distinta.

    Devuelve las imágenes ya en `[0, 1]` y aplanadas a 784 columnas.
    """
    fondo = generar_ejemplos_de_fondo(x, rng)
    etiquetas_fondo = np.full(len(fondo), CLASE_FONDO, dtype=np.uint8)

    digitos = _preparar_digitos(x)
    x_total = np.concatenate([digitos, fondo.astype(np.float32) / 255.0])
    y_total = np.concatenate([y, etiquetas_fondo])
    orden = rng.permutation(len(x_total))
    x_total, y_total = x_total[orden], y_total[orden]

    x_total = _variar(x_total, rng)
    return x_total.reshape(-1, 784), y_total


def entrenar(
    epochs: int = EPOQUES,
    ruta: Path | str = RUTA_MODELO,
    verbose: int = 1,
    nombre_modelo: str | None = None,
) -> tuple[object, dict[str, float]]:
    """Entrena el clasificador, lo evalúa y lo guarda.

    Devuelve el modelo y un diccionario con la pérdida y la exactitud sobre el
    conjunto de prueba.
    """
    import tensorflow as tf

    from .datos import cargar_mnist

    tf.keras.utils.set_random_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)

    x_ent, y_ent, x_pru, y_pru = cargar_mnist()
    x_ent, y_ent = _aumentar_con_fondo(x_ent, y_ent, rng)

    modelo = construir_modelo()
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.fit(
        x_ent,
        y_ent,
        epochs=epochs,
        batch_size=LOTE,
        validation_split=VALIDACION,
        verbose=verbose,
    )
    perdida, exactitud = modelo.evaluate(preprocesar(x_pru), y_pru, verbose=0)

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(ruta)
    metricas = {"perdida_prueba": float(perdida), "exactitud_prueba": float(exactitud)}
    _guardar_ficha(
        ruta,
        exactitud=metricas["exactitud_prueba"],
        perdida=metricas["perdida_prueba"],
        epocas=epochs,
        nombre=nombre_modelo or ruta.stem.replace("_", " "),
    )
    return modelo, metricas


def cargar_modelo(ruta: Path | str = RUTA_MODELO):
    """Carga el modelo entrenado desde disco."""
    from tensorflow import keras

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. "
            "Entrénalo con: python -m src.modelo"
        )
    return keras.models.load_model(ruta)


# --------------------------------------------------------------- registro


@dataclass(frozen=True)
class Descriptor:
    """La ficha de un modelo: lo que el dashboard necesita para mostrarlo."""

    nombre: str
    ruta: Path
    tipo: str
    """`denso` (perceptrón multicapa) o `convolucional`."""

    clases: int
    """Cuántas salidas tiene: los dígitos del 0 al 9 y la clase `fondo`."""

    entrada: str
    """Qué recibe: `784` (vector) o `28x28` (imagen)."""

    tamano: int
    """Tamaño del archivo, en bytes."""

    exactitud: float | None = None
    """Exactitud en el conjunto de prueba, si la ficha la trae."""

    perdida: float | None = None
    """Pérdida en el conjunto de prueba, si la ficha la trae."""

    untrained_note: str = ""
    """Nota de la ficha (épocas, fecha, etc.)."""


def _tipo_de_forma(forma) -> str:
    """Deduce si el modelo es denso o convolucional por su forma de entrada."""
    if isinstance(forma, (list, tuple)) and len(forma) == 3:
        return "convolucional"
    return "denso"


def _forma_de_entrada(modelo) -> str:
    forma = getattr(modelo, "input_shape", None)
    if isinstance(forma, (list, tuple)) and len(forma) == 3:
        return "x".join(str(dimension) for dimension in forma[1:])
    return "784"


def _leer_ficha(ruta: Path) -> dict:
    """Lee la ficha JSON que acompaña a un modelo, si existe."""
    ficha = ruta.with_suffix(".json")
    if not ficha.exists():
        return {}
    try:
        return json.loads(ficha.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def describir_modelo(ruta: Path | str) -> Descriptor:
    """Carga el modelo y arma su ficha para el dashboard."""
    ruta = Path(ruta)
    modelo = cargar_modelo(ruta)
    ficha = _leer_ficha(ruta)

    forma_salida = getattr(modelo, "output_shape", None)
    clases = int(forma_salida[-1]) if forma_salida else 11

    return Descriptor(
        nombre=ficha.get("nombre", ruta.stem.replace("_", " ")),
        ruta=ruta,
        tipo=_tipo_de_forma(getattr(modelo, "input_shape", None)),
        clases=clases,
        entrada=_forma_de_entrada(modelo),
        tamano=ruta.stat().st_size,
        exactitud=ficha.get("exactitud_prueba"),
        perdida=ficha.get("perdida_prueba"),
        untrained_note=ficha.get("nota", ""),
    )


def modelos_disponibles(carpeta: Path | str = CARPETA_MODELOS) -> list[Descriptor]:
    """Todos los modelos disponibles en la carpeta, listos para elegir.

    Es lo que permite cambiar de modelo desde el dashboard: cuando llegue la
    red convolucional, basta con dejarla en `modelos/` para que aparezca en
    el desplegable.
    """
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []

    descriptores: list[Descriptor] = []
    for ruta in sorted(carpeta.glob("*.keras")):
        try:
            descriptores.append(describir_modelo(ruta))
        except Exception:  # noqa: BLE001 - un modelo roto no debe tumbar la página
            continue
    return descriptores


def _guardar_ficha(
    ruta: Path,
    exactitud: float,
    perdida: float,
    epocas: int,
    nombre: str,
) -> None:
    """Guarda la ficha JSON que acompaña al modelo."""
    ficha = {
        "nombre": nombre,
        "exactitud_prueba": round(exactitud, 6),
        "perdida_prueba": round(perdida, 6),
        "epocas": epocas,
        "entrenado": date.today().isoformat(),
    }
    ruta.with_suffix(".json").write_text(
        json.dumps(ficha, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":  # pragma: no cover - uso manual
    modelo, metricas = entrenar()
    print(f"Modelo guardado en {RUTA_MODELO}")
    print(f"Pérdida en prueba: {metricas['perdida_prueba']:.4f}")
    print(f"Exactitud en prueba: {metricas['exactitud_prueba']:.4%}")
