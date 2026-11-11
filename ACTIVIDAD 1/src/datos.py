"""Carga del dataset MNIST.

Se usa la vía oficial de Keras (`tf.keras.datasets.mnist.load_data`). Si esa
descarga falla (sin red, o el mirror caído) se recurre a los archivos IDX
originales en GitHub, que son los mismos que usa el cuaderno del Parcial 1.

El MNIST tiene 10 clases: 0 a 9. El enunciado pide del 1 al 9, pero el
clasificador es el mismo para las diez, así que el 0 también se reconoce.
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

URL_IDX = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/",
}

RUTA_CACHE = Path(__file__).resolve().parent.parent / "datos_cache"


def _leer_idx(ruta: str) -> np.ndarray:
    """Lee un archivo IDX comprimido (el formato original del MNIST).

    Los primeros bytes del archivo son un encabezado: dos bytes de código
    (0x0008) y dos bytes con el número de dimensiones (1 o 3).
    """
    with gzip.open(ruta, "rb") as archivo:
        dato = archivo.read()
    codigo = int.from_bytes(dato[0:2], "big")
    if codigo != 8:
        raise ValueError(f"{ruta}: se esperaba un archivo IDX de código 8, llegó {codigo}")
    dimensiones = int.from_bytes(dato[2:4], "big")
    if dimensiones == 1:
        return np.frombuffer(dato, dtype=np.uint8, offset=4)
    if dimensiones == 3:
        return np.frombuffer(dato, dtype=np.uint8, offset=4).reshape(-1, 28, 28)
    raise ValueError(f"{ruta}: número de dimensiones no soportado: {dimensiones}")


def _descargar_idx(nombre: str) -> tuple[np.ndarray, np.ndarray]:
    """Descarga y lee los cuatro archivos IDX del MNIST."""
    base = URL_IDX["mnist"]
    RUTA_CACHE.mkdir(parents=True, exist_ok=True)
    imagenes: list[np.ndarray] = []
    etiquetas: list[np.ndarray] = []
    for sufijo, destino in (
        ("train-images-idx3-ubyte.gz", imagenes),
        ("train-labels-idx1-ubyte.gz", etiquetas),
        ("t10k-images-idx3-ubyte.gz", imagenes),
        ("t10k-labels-idx1-ubyte.gz", etiquetas),
    ):
        local = RUTA_CACHE / sufijo
        if not local.exists():
            urllib.request.urlretrieve(base + sufijo, local)  # noqa: S310
        destino.append(_leer_idx(str(local)))
    return np.concatenate(imagenes[:2]), np.concatenate(etiquetas[:2])


def cargar_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve `(x_ent, y_ent, x_pru, y_pru)` con las formas de MNIST.

    x: imágenes uint8 de 28x28 (60000 de entrenamiento, 10000 de prueba).
    y: etiquetas uint8 con la clase de cada imagen (0 a 9).
    """
    try:
        from tensorflow.keras.datasets import mnist

        (x_ent, y_ent), (x_pru, y_pru) = mnist.load_data()
        return x_ent, y_ent, x_pru, y_pru
    except Exception:  # noqa: BLE001 - sin red, se usa la vía alternativa
        x, y = _descargar_idx("mnist")
        return x[:60000], y[:60000], x[60000:], y[60000:]
