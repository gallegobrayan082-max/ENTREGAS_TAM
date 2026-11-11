"""Evalúa el modelo y el detector sobre el conjunto de prueba y los ejemplos.

    python -m herramientas.evaluar

Mide tres cosas:

1. La exactitud del clasificador sobre los dígitos de prueba de MNIST, tal
   como los ve el detector (recortados a su tinta y centrados).
2. Cuántos dígitos reales caen en la clase `fondo` (falsos rechazos).
3. Qué detecta el detector en las imágenes de `ejemplos/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import preprocesamiento as prep  # noqa: E402
from src.datos import cargar_mnist  # noqa: E402
from src.deteccion import DetectorDeDigitos  # noqa: E402
from src.modelo import CLASE_FONDO, cargar_modelo  # noqa: E402

ESPERADOS = {
    "caso_4_y_5": [4, 5],
    "varios_digitos": [3, 1, 8, 2],
    "un_solo_digito": [7],
    "cinco_digitos": [1, 9, 4, 6, 3],
}


def evaluar_clasificador(modelo) -> None:
    """Exactitud y falsos rechazos sobre el conjunto de prueba de MNIST."""
    _, _, x_pru, y_pru = cargar_mnist()

    # (a) el dígito tal cual, como en el dataset original
    plano = modelo.predict(x_pru.reshape(-1, 784).astype("float32") / 255.0, verbose=0)
    # (b) el dígito como lo ve el detector: recortado a su tinta y centrado
    recortes = np.stack(
        [
            prep.normalizar_mnist(
                imagen.astype(np.float32) / 255.0,
                prep.caja_de_tinta(prep.binarizar(imagen.astype(np.float32) / 255.0)),
            )
            for imagen in x_pru
        ]
    )
    ajustados = modelo.predict(recortes.reshape(-1, 784), verbose=0)

    for nombre, probabilidades in (("sin recortar", plano), ("recortado", ajustados)):
        argmax = probabilidades.argmax(axis=1)
        exactitud = float((argmax == y_pru).mean())
        rechazos = float((argmax == CLASE_FONDO).mean())
        print(
            f"  {nombre:12} exactitud {exactitud:7.2%}   "
            f"mandados a fondo {rechazos:6.2%}   "
            f"error entre dígitos {exactitud + rechazos - 1:6.2%}"
        )


def evaluar_detector(detector) -> None:
    """Qué detecta el detector en las imágenes de ejemplo."""
    aciertos = 0
    total = 0
    for nombre, verdad in ESPERADOS.items():
        for sufijo in ("", "_papel"):
            ruta = RAIZ / "ejemplos" / f"{nombre}{sufijo}.png"
            if not ruta.exists():
                continue
            detecciones = detector.detectar(Image.open(ruta))
            detectados = [d.digito for d in detecciones]
            bien = detectados == verdad
            aciertos += int(bien)
            total += 1
            marca = "OK   " if bien else "FALLA"
            print(
                f"  {marca} {ruta.name:26} verdad={verdad} "
                f"detectados={detectados} "
                f"conf={[round(d.confianza, 3) for d in detecciones]}"
            )
    print(f"  -> {aciertos}/{total} imágenes de ejemplo correctas")


def main() -> None:
    modelo = cargar_modelo()
    print("Clasificador:")
    evaluar_clasificador(modelo)
    print("Detector:")
    evaluar_detector(DetectorDeDigitos(modelo))


if __name__ == "__main__":
    main()
