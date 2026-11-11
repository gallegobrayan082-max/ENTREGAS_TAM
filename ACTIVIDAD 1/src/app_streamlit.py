"""Interfaz web con Streamlit: subir una imagen y ver los dígitos detectados.

    streamlit run src/app_streamlit.py

Se muestran **una detección por dígito**: cada uno con su etiqueta, su
confianza y su caja, dibujados sobre la imagen original.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# `streamlit run` ejecuta el archivo como script, así que las importaciones
# relativas no funcionan: se agrega la raíz del proyecto al `sys.path`.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.deteccion import DetectorDeDigitos, Parametros, anotar  # noqa: E402
from src.modelo import cargar_modelo  # noqa: E402

st.set_page_config(
    page_title="Detección de dígitos",
    page_icon="🔢",
    layout="wide",
)


@st.cache_resource
def obtener_detector() -> DetectorDeDigitos:
    """Carga el modelo una vez y lo reutiliza entre recargas de la página.

    `cache_resource` y no `cache_data` porque el modelo de Keras no se puede
    serializar para cachearlo por valor.
    """
    return DetectorDeDigitos(cargar_modelo())


def _abrir_ejemplos() -> dict[str, bytes]:
    """Las imágenes de `ejemplos/`, para probarlo sin subir nada."""
    carpeta = RAIZ / "ejemplos"
    if not carpeta.exists():
        return {}
    return {
        archivo.stem.replace("_", " "): archivo.read_bytes()
        for archivo in sorted(carpeta.glob("*.png"))
    }


def _detectar(datos: bytes, umbral: float) -> tuple[Image.Image, list]:
    """Procesa los bytes de una imagen y devuelve la imagen y las detecciones."""
    imagen = Image.open(io.BytesIO(datos))
    imagen.load()
    detector = obtener_detector()
    detector.parametros = Parametros(umbral_confianza=umbral)
    return imagen, detector.detectar(imagen)


# ------------------------------------------------------------------- página

st.title("🔢 Detección de varios dígitos en una imagen")
st.write(
    "Sube una imagen con uno o varios dígitos escritos a mano. "
    "Cada dígito se reporta **por separado**: un 4 y un 5 salen como dos "
    "detecciones, nunca como el número 45."
)

barra_izquierda, barra_derecha = st.columns([1, 1.2], gap="large")

with barra_izquierda:
    umbral = st.slider("Confianza mínima", 5, 99, 85, 1) / 100

    subida = st.file_uploader(
        "Sube una imagen",
        type=["png", "jpg", "jpeg", "bmp", "webp", "gif"],
        help="PNG, JPG, BMP… con uno o varios dígitos.",
    )

    ejemplos = _abrir_ejemplos()
    datos_ejemplo = None
    if ejemplos:
        with st.expander("O prueba con un ejemplo"):
            nombre = st.selectbox("Ejemplo", list(ejemplos))
            datos_ejemplo = ejemplos[nombre]

    if subida is not None:
        datos = subida.getvalue()
    elif datos_ejemplo is not None:
        datos = datos_ejemplo
    else:
        datos = None

if datos is None:
    st.info("Sube una imagen o elige un ejemplo para empezar.")
    st.stop()

with barra_derecha:
    try:
        imagen, detecciones = _detectar(datos, umbral)
    except Exception as error:  # noqa: BLE001 - un archivo inválido no es un fallo
        st.error(f"No se pudo procesar la imagen: {error}")
        st.stop()

    anotada = anotar(imagen, detecciones)
    buffer = io.BytesIO()
    anotada.save(buffer, format="PNG")

    st.image(anotada, caption="Cajas sobre la imagen original", width="stretch")

    total = len(detecciones)
    if total == 0:
        st.warning(
            "No se encontró ningún dígito con esa confianza. "
            "Prueba a bajarla un poco."
        )
    else:
        st.success(f"**{total}** dígito{'s' if total > 1 else ''} detectado{'s' if total > 1 else ''}.")

        st.markdown("#### Dígitos detectados")
        for indice, deteccion in enumerate(detecciones, 1):
            columna_izquierda, columna_media, columna_derecha = st.columns([0.4, 2, 1])
            with columna_izquierda:
                st.markdown(
                    f"<div style='font-size:2.4rem;font-weight:700;color:#4ade80;"
                    f"line-height:1'>{deteccion.digito}</div>",
                    unsafe_allow_html=True,
                )
            with columna_media:
                st.progress(deteccion.confianza)
            with columna_derecha:
                st.caption(
                    f"#{indice} · {deteccion.confianza:.1%} · caja {deteccion.caja}"
                )

    st.download_button(
        "Descargar imagen anotada",
        data=buffer.getvalue(),
        file_name="deteccion.png",
        mime="image/png",
    )
