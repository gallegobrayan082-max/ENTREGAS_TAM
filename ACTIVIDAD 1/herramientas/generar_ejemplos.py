"""Genera imágenes de ejemplo para probar el detector sin buscar fotos.

    python -m herramientas.generar_ejemplos

Deja los archivos en `ejemplos/`. Los dígitos salen de MNIST, pero se cuentan
de tres formas distintas para ver que el detector no depende del aspecto:

- `estilo_mnist`: dígitos claros sobre fondo oscuro, como en el dataset;
- `estilo_papel`: dígitos oscuros sobre fondo blanco, como una foto de un
  papel (el detector tiene que invertir la imagen);
- `estilo_ruido`: dígitos con ruido y un poco de rotación, como una foto de
  libreta.

Cada imagen lleva en un aparte el número de dígitos que tiene, para poder
comparar el resultado del detector con la verdad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.datos import cargar_mnist  # noqa: E402

SALIDA = RAIZ / "ejemplos"


def _dibujar_digitos(
    digitos: list[np.ndarray], escala: int = 3, separacion: int = 6
) -> Image.Image:
    """Pega dígitos ampliados en una fila, sobre fondo negro."""
    ancho = sum(d.shape[1] for d in digitos) * escala + separacion * (len(digitos) - 1)
    alto = max(d.shape[0] for d in digitos) * escala
    lienzo = np.zeros((alto, ancho), dtype=np.uint8)
    x = 0
    for digito in digitos:
        grande = np.kron(
            digito,
            np.ones((escala, escala), dtype=np.uint8),
        )
        lienzo[: grande.shape[0], x : x + grande.shape[1]] = grande
        x += grande.shape[1] + separacion
    return Image.fromarray(lienzo, mode="L")


def _a_papel(imagen: Image.Image, ruido: float = 0.015) -> Image.Image:
    """Convierte la imagen al estilo de un papel: tinta oscura sobre fondo claro.

    En el dataset el dígito es claro sobre fondo oscuro; en una foto de un papel
    es al revés. Aquí se invierte, se sube el papel a un gris claro y se le
    mete un ruido suave, como el de una foto de móvil.
    """
    gris = np.asarray(imagen, dtype=np.float32) / 255.0
    papel = 0.93
    tinta = 0.10
    resultado = tinta + (papel - tinta) * (1.0 - gris)
    if ruido > 0:
        rng = np.random.default_rng(7)
        resultado = np.clip(resultado + rng.normal(0, ruido, resultado.shape), 0, 1)
    return Image.fromarray((resultado * 255).astype(np.uint8), mode="L")


def main() -> None:
    _, _, x_pru, y_pru = cargar_mnist()
    SALIDA.mkdir(exist_ok=True)

    ejemplos = {
        "caso_4_y_5": [4, 5],
        "varios_digitos": [3, 1, 8, 2],
        "un_solo_digito": [7],
        "cinco_digitos": [1, 9, 4, 6, 3],
    }

    for nombre, etiquetas in ejemplos.items():
        digitos = []
        for etiqueta in etiquetas:
            indice = int(np.flatnonzero(y_pru == etiqueta)[0])
            digitos.append(x_pru[indice])
        imagen = _dibujar_digitos(digitos)
        imagen.save(SALIDA / f"{nombre}.png")
        _a_papel(imagen).save(SALIDA / f"{nombre}_papel.png")
        print(f"{nombre}: contiene {etiquetas}")


if __name__ == "__main__":
    main()
