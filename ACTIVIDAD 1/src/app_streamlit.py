"""Dashboard de detección de dígitos, en Streamlit.

    streamlit run src/app_streamlit.py

Trae las dos cosas que se pidieron:

1. **Un dashboard** con indicadores, la imagen anotada y la tabla de
   detecciones, más un modo por lotes para varias imágenes.
2. **Cambiar de modelo desde la propia página.** Los modelos se leen de la
   carpeta `modelos/`, así que cuando se añada la red convolucional aparecerá
   en el desplegable sin tocar este archivo: el detector se adapta solo a la
   forma de entrada del modelo (vector de 784 para el denso, imagen de 28x28
   para el convolucional).
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# `streamlit run` ejecuta el archivo como script, así que las importaciones
# relativas no funcionan: se agrega la raíz del proyecto al `sys.path`.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.deteccion import DetectorDeDigitos, Parametros, anotar  # noqa: E402
from src.modelo import (  # noqa: E402
    CARPETA_MODELOS,
    cargar_modelo,
    modelos_disponibles,
)

st.set_page_config(
    page_title="Detección de dígitos",
    page_icon="🔢",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _cargar(ruta: str):
    """Carga el modelo una vez y lo reutiliza.

    Se cachea **el modelo**, no el detector: el detector es barato de crear y
    lleva dentro los umbrales de la petición, que no deben compartirse entre
    sesiones.
    """
    return cargar_modelo(ruta)


def _detector(ruta: str, umbral: float) -> DetectorDeDigitos:
    return DetectorDeDigitos(_cargar(ruta), Parametros(umbral_confianza=umbral))


def _ejemplos() -> dict[str, bytes]:
    """Las imágenes de `ejemplos/`, para probarlo sin subir nada."""
    carpeta = RAIZ / "ejemplos"
    if not carpeta.exists():
        return {}
    return {
        archivo.stem.replace("_", " "): archivo.read_bytes()
        for archivo in sorted(carpeta.glob("*.png"))
    }


def _analizar(datos: bytes, detector: DetectorDeDigitos):
    """Procesa una imagen y devuelve (imagen, detecciones, milisegundos)."""
    imagen = Image.open(io.BytesIO(datos))
    imagen.load()
    inicio = time.perf_counter()
    detecciones = detector.detectar(imagen)
    milisegundos = (time.perf_counter() - inicio) * 1000
    return imagen, detecciones, milisegundos


def _anotada_en_bytes(imagen: Image.Image, detecciones) -> bytes:
    buffer = io.BytesIO()
    anotar(imagen, detecciones).save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------------- página

st.title("🔢 Detección de varios dígitos en una imagen")
st.write(
    "Sube una imagen con uno o varios dígitos escritos a mano. "
    "Cada dígito se reporta **por separado**: un 4 y un 5 salen como dos "
    "detecciones, nunca como el número 45."
)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Modelo")

    descriptores = modelos_disponibles(CARPETA_MODELOS)
    if not descriptores:
        st.error(
            "No hay ningún modelo en `modelos/`. "
            "Entrena uno con `python -m src.modelo`."
        )
        st.stop()

    def _etiqueta(descriptor) -> str:
        exactitud = (
            f" · {descriptor.exactitud:.1%}" if descriptor.exactitud is not None else ""
        )
        return f"{descriptor.nombre} ({descriptor.tipo}){exactitud}"

    rutas = {d.ruta.stem: d for d in descriptores}
    seleccion = st.radio(
        "Elige el modelo con el que se va a detectar",
        list(rutas),
        format_func=lambda clave: _etiqueta(rutas[clave]),
    )
    descriptor = rutas[seleccion]

    with st.container(border=True):
        st.markdown("**Ficha del modelo**")
        st.markdown(
            f"""
- **Tipo:** {descriptor.tipo}
- **Recibe:** {descriptor.entrada}
- **Clases:** {descriptor.clases} (los dígitos 0-9 y `fondo`)
- **Tamaño:** {descriptor.tamano / 1024:.0f} KB
- **Exactitud en prueba:** {f"{descriptor.exactitud:.2%}" if descriptor.exactitud else "sin medir"}
"""
        )

    st.header("Detección")
    umbral = st.slider("Confianza mínima", 5, 99, 85, 1) / 100
    st.caption(
        "Por debajo de este valor se descarta la detección. "
        "Los recortes que salen de un bloque de tinta usan un umbral menor, "
        "porque ahí la geometría ya garantiza que hay un dígito."
    )

    detector = _detector(str(descriptor.ruta), umbral)

# ------------------------------------------------------------------- entrada

columna_subida, columna_resultado = st.columns([1, 1.3], gap="large")

with columna_subida:
    subidas = st.file_uploader(
        "Sube una o varias imágenes",
        type=["png", "jpg", "jpeg", "bmp", "webp", "gif"],
        accept_multiple_files=True,
        help="PNG, JPG, BMP… con uno o varios dígitos.",
    )

    ejemplos = _ejemplos()
    if ejemplos:
        with st.expander("O prueba con un ejemplo"):
            nombre = st.selectbox("Ejemplo", list(ejemplos))
            datos_ejemplo = ejemplos[nombre]

    if not subidas:
        if "datos_ejemplo" in locals():
            subidas = [("ejemplo", datos_ejemplo)]
        else:
            st.info("Sube una imagen o elige un ejemplo para empezar.")
            st.stop()

# ------------------------------------------------------------------ resultado

with columna_resultado:
    if len(subidas) > 1:
        st.subheader("Lote")
        filas = []
        for nombre, datos in subidas:
            try:
                imagen, detecciones, milisegundos = _analizar(datos, detector)
            except Exception as error:  # noqa: BLE001
                st.error(f"{nombre}: {error}")
                continue
            filas.append(
                {
                    "imagen": nombre,
                    "dígitos": " ".join(str(d.digito) for d in detecciones),
                    "cantidad": len(detecciones),
                    "confianza media": round(
                        sum(d.confianza for d in detecciones) / max(1, len(detecciones)), 4
                    ),
                    "tiempo (ms)": round(milisegundos, 1),
                }
            )
        if filas:
            st.dataframe(filas, width="stretch", hide_index=True)
    else:
        nombre, datos = subidas[0]
        try:
            imagen, detecciones, milisegundos = _analizar(datos, detector)
        except Exception as error:  # noqa: BLE001 - un archivo inválido no es un fallo
            st.error(f"No se pudo procesar la imagen: {error}")
            st.stop()

        bytes_anotada = _anotada_en_bytes(imagen, detecciones)

        indicador_1, indicador_2, indicador_3, indicador_4 = st.columns(4)
        indicador_1.metric("Dígitos detectados", len(detecciones))
        indicador_2.metric(
            "Confianza media",
            f"{sum(d.confianza for d in detecciones) / max(1, len(detecciones)):.1%}",
        )
        indicador_3.metric("Tiempo", f"{milisegundos:.0f} ms")
        indicador_4.metric("Imagen", f"{imagen.width}×{imagen.height}")

        st.image(
            Image.open(io.BytesIO(bytes_anotada)),
            caption="Cajas sobre la imagen original",
            width="stretch",
        )

        if detecciones:
            st.markdown("#### Dígitos detectados")
            tabla = [
                {
                    "#": indice,
                    "dígito": deteccion.digito,
                    "confianza": round(deteccion.confianza, 4),
                    "caja (x0, y0, x1, y1)": str(deteccion.caja),
                    "centro": (
                        round(deteccion.centro[0]),
                        round(deteccion.centro[1]),
                    ),
                }
                for indice, deteccion in enumerate(detecciones, 1)
            ]
            st.dataframe(tabla, width="stretch", hide_index=True)
        else:
            st.warning(
                "No se encontró ningún dígito con esa confianza. "
                "Prueba a bajarla un poco."
            )

        st.download_button(
            "Descargar imagen anotada",
            data=bytes_anotada,
            file_name="deteccion.png",
            mime="image/png",
        )
