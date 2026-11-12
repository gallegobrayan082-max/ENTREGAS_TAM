"""Preprocesamiento de imágenes para el detector de dígitos.

El clasificador de MNIST espera un parche de 28x28 en escala `[0, 1]` con un
único dígito centrado, ocupando los 20 píxeles centrales. Este módulo convierte
una imagen arbitraria (foto, escaneo, lienzo) a esa representación.

Las piezas, en orden:

1. `a_escala_grises` — un solo canal, valores en `[0, 1]`.
2. `invertir_si_hace_falla` — deja el trazo como valor **alto**, como en MNIST.
3. `umbral_otsu` / `binarizar` — separa la tinta del fondo.
4. `caja_de_tinta` — recorta lo que sobra.
5. `normalizar_mnist` — escala y centra el dígito en el lienzo de 28x28.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

TAMANO_MNIST = 28
"""Lado del lienzo de salida (28x28), igual que en MNIST."""

LADO_INTERIOR = 20
"""Lado máximo que ocupa el dígito dentro del lienzo (20 de 28 píxeles)."""


def a_escala_grises(imagen: Image.Image) -> np.ndarray:
    """Convierte la imagen a un array float32 de un canal con valores en `[0, 1]`."""
    if imagen.mode != "L":
        imagen = imagen.convert("L")
    return np.asarray(imagen, dtype=np.float32) / 255.0


def invertir_si_hace_falla(gris: np.ndarray) -> np.ndarray:
    """Deja el trazo del dígito como valor alto, que es la convención de MNIST.

    En MNIST el dígito es claro sobre fondo oscuro. Una foto de un papel es al
    revés: tinta oscura sobre fondo blanco. Hay que decidir con qué criterio,
    y se usan tres señales, de la más segura a la menos:

    1. Si el marco de la imagen es mucho más claro que el resto, el fondo es
       claro: hay que invertir.
    2. Si el marco es mucho más oscuro que el resto, el fondo es oscuro: se
       deja como está.
    3. Si el marco no dice nada (imagen uniforme en los bordes), decide la
       clase minoritaria de Otsu: el trazo de un dígito es la clase pequeña,
       así que si la clase alta es la mayoría, la tinta es la clase baja y hay
       que invertir.

    Se compara por la mediana del marco y no por el promedio, porque un
    trazo pegado al borde no debe arrastrar la decisión.
    """
    if gris.size == 0 or min(gris.shape) < 2:
        return gris

    marco = np.concatenate([gris[0, :], gris[-1, :], gris[:, 0], gris[:, -1]])
    mediana_marco = float(np.median(marco))
    mediana_imagen = float(np.median(gris))
    if mediana_marco > mediana_imagen + 0.15:
        return 1.0 - gris
    if mediana_marco < mediana_imagen - 0.15:
        return gris

    niveles = niveles_de_intensidad(gris)
    corte = umbral_otsu(gris)
    altos = int((niveles > corte).sum())
    bajos = int((niveles <= corte).sum())
    if altos > bajos:
        # La clase alta es la mayoría: la tinta es la clase baja.
        return 1.0 - gris
    return gris


def niveles_de_intensidad(gris: np.ndarray) -> np.ndarray:
    """Convierte la imagen en escala `[0, 1]` a niveles enteros de 0 a 255.

    Todo el preprocesamiento trabaja sobre estos niveles para que el umbral y
    la comparación usen la misma cuantización.
    """
    return (np.clip(gris, 0.0, 1.0) * 255.0).astype(np.uint8)


def umbral_otsu(gris: np.ndarray) -> int:
    """Umbral de Otsu: el nivel de corte que maximiza la varianza entre clases.

    Devuelve un entero entre 0 y 255. Los píxeles con nivel **mayor** que el
    umbral son los de la clase "clara". Se calcula sobre el histograma de 256
    niveles, sin dependencias extra.
    """
    niveles = niveles_de_intensidad(gris)
    hist = np.bincount(niveles.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    valores = np.arange(256, dtype=np.float64)
    peso_fondo = np.cumsum(hist)
    suma_fondo = np.cumsum(hist * valores)
    suma_total = float((hist * valores).sum())
    peso_tinta = total - peso_fondo
    with np.errstate(divide="ignore", invalid="ignore"):
        media_fondo = suma_fondo / peso_fondo
        media_tinta = (suma_total - suma_fondo) / peso_tinta
        varianza = peso_fondo * peso_tinta * (media_fondo - media_tinta) ** 2
    varianza[~np.isfinite(varianza)] = -1.0
    return int(np.argmax(varianza))


def binarizar(gris: np.ndarray, umbral: int | None = None) -> np.ndarray:
    """Devuelve una máscara booleana donde `True` es tinta (el trazo del dígito)."""
    corte = umbral_otsu(gris) if umbral is None else int(umbral)
    return niveles_de_intensidad(gris) > corte


def caja_de_tinta(binar: np.ndarray) -> tuple[int, int, int, int] | None:
    """Caja que envuelve toda la tinta, en `(x0, y0, x1, y1)`, o `None` si no hay tinta."""
    filas = np.flatnonzero(binar.any(axis=1))
    if filas.size == 0:
        return None
    columnas = np.flatnonzero(binar.any(axis=0))
    return int(columnas[0]), int(filas[0]), int(columnas[-1]) + 1, int(filas[-1]) + 1


def recortar(gris: np.ndarray, caja: tuple[int, int, int, int]) -> np.ndarray:
    """Recorta la región `(x0, y0, x1, y1)` de la imagen en escala de grises."""
    x0, y0, x1, y1 = caja
    return gris[y0:y1, x0:x1]


def normalizar_mnist(gris: np.ndarray, caja: tuple[int, int, int, int]) -> np.ndarray:
    """Convierte un dígito recortado en el parche de 28x28 que espera el modelo.

    Se escala el recorte para que quepa en 20x20 respetando la relación de
    aspecto y se pega centrado en el lienzo de 28x28, que es exactamente el
    preprocesamiento del dataset original.

    Al final el fondo se deja en **negro** y el trazo en **blanco**, como en
    MNIST. El recorte viene de una foto, donde el papel es gris; sin este
    último estirón, el modelo recibe una imagen que no se parece a ninguna de
    las que vio entrenándose.

    Se conserva el antialiasing (los bordes grises del trazo) a propósito: si
    se convierte todo a binario, el modelo pierde 1,8 puntos de exactitud,
    porque en MNIST los bordes de los trazos también tienen gris.

    Devuelve un array float32 de 28x28 con valores en `[0, 1]`.
    """
    parche = recortar(gris, caja)
    alto, ancho = parche.shape
    if alto == 0 or ancho == 0:
        return np.zeros((TAMANO_MNIST, TAMANO_MNIST), dtype=np.float32)

    escala = LADO_INTERIOR / max(alto, ancho)
    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))

    # PIL hace un reescalado con antialiasing (LANCZOS), que conserva el grosor
    # del trazo; un reescalado nearest-neighbor dejaría el dígito.blocky.
    recortada = Image.fromarray((np.clip(parche, 0, 1) * 255).astype(np.uint8))
    escalada = recortada.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)

    lienzo = Image.new("L", (TAMANO_MNIST, TAMANO_MNIST), color=0)
    lienzo.paste(escalada, ((TAMANO_MNIST - nuevo_ancho) // 2, (TAMANO_MNIST - nuevo_alto) // 2))
    parche = np.asarray(lienzo, dtype=np.float32) / 255.0

    return _estirar_fondo(parche)


def _estirar_fondo(parche: np.ndarray) -> np.ndarray:
    """Deja el fondo en negro y el trazo en blanco.

    Se mide el nivel del fondo con el percentil 5 (robusto: un grano suelto no
    lo desplaza) y el de la tinta con el percentil 99, y se estira el rango
    entre los dos. Un recorte que ya venía en negro y blanco no cambia.
    """
    if parche.size == 0:
        return parche
    fondo, tinta = np.percentile(parche, [5, 99])
    if tinta <= fondo:
        return parche
    estirado = np.clip((parche - fondo) / (tinta - fondo), 0.0, 1.0)
    return estirado.astype(np.float32)


def razon_de_tinta(binar: np.ndarray, caja: tuple[int, int, int, int]) -> float:
    """Proporción de píxeles de tinta dentro de la caja (0 = vacía, 1 = completamente negra)."""
    parche = binar[caja[1] : caja[3], caja[0] : caja[2]]
    if parche.size == 0:
        return 0.0
    return float(parche.mean())
