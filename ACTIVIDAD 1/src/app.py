"""Interfaz web para subir imágenes y ver los dígitos detectados.

Se ejecuta con:

    python -m src.app

y queda en http://127.0.0.1:5000.

La interfaz recibe una imagen, la pasa por el detector y devuelve **una fila por
dígito** con su etiqueta, su confianza y su caja, además de la imagen con las
cajas dibujadas.
"""

from __future__ import annotations

import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from .deteccion import DetectorDeDigitos, Parametros, anotar
from .modelo import cargar_modelo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB por imagen

_modelo = None


def obtener_modelo():
    """Carga el modelo una sola vez y lo reutiliza en cada petición."""
    global _modelo
    if _modelo is None:
        _modelo = cargar_modelo()
    return _modelo


@app.get("/")
def indice():
    """Página principal con el formulario de carga."""
    return render_template("index.html")


@app.post("/predecir")
def predecir():
    """Recibe la imagen y devuelve las detecciones en JSON."""
    archivo = request.files.get("archivo")
    if archivo is None or not archivo.filename:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    try:
        imagen = ImageOps.exif_transpose(Image.open(io.BytesIO(archivo.read())))
        imagen.load()
    except Exception:  # noqa: BLE001 - un archivo que no es imagen no es un fallo del servidor
        return jsonify({"error": "El archivo no es una imagen válida."}), 400

    try:
        umbral = float(request.form.get("umbral", 0.85))
    except ValueError:
        umbral = 0.85
    umbral = min(max(umbral, 0.05), 0.99)

    detector = DetectorDeDigitos(
        obtener_modelo(),
        Parametros(umbral_confianza=umbral),
    )
    detecciones = detector.detectar(imagen)

    buffer = io.BytesIO()
    anotar(imagen, detecciones).save(buffer, format="PNG")
    anotada = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    return jsonify(
        {
            "detecciones": [d.a_dict() for d in detecciones],
            "anotada": anotada,
            "ancho": imagen.width,
            "alto": imagen.height,
            "umbral": umbral,
        }
    )


if __name__ == "__main__":  # pragma: no cover - uso manual
    app.run(host="127.0.0.1", port=5000, debug=False)
