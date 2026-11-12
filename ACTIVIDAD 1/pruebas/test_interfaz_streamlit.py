"""Pruebas de la interfaz de Streamlit.

Se ejecutan con el propio framework de Streamlit (`AppTest`), que corre la
aplicación en proceso: así se comprueba que el dashboard se dibuja, que no
lanza excepciones y que el ejemplo de un 4 y un 5 sale como dos dígitos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RUTA_APP = Path(__file__).resolve().parent.parent / "src" / "app_streamlit.py"

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    """La aplicación ya ejecutada una vez, con el ejemplo por defecto."""
    if not RUTA_APP.exists():
        pytest.skip("no está la interfaz de Streamlit")
    instancia = AppTest.from_file(str(RUTA_APP), default_timeout=300)
    instancia.run()
    return instancia


def test_la_aplicacion_no_lanza_excepciones(app):
    assert app.exception == []


def test_tiene_los_controles_de_la_pagina(app):
    assert [s.label for s in app.slider] == ["Confianza mínima"]
    assert len(app.file_uploader) == 1
    assert [s.label for s in app.selectbox] == ["Ejemplo"]


def test_se_puede_elegir_el_modelo(app):
    """El selector de modelo es el que permite cambiar entre el MLP y la CNN."""
    assert len(app.radio) == 1
    opciones = app.radio[0].options
    assert len(opciones) >= 1
    # la etiqueta que se ve incluye el tipo y la exactitud
    assert any("denso" in opcion for opcion in opciones)
    assert all("%" in opcion for opcion in opciones)


def test_cambiar_de_modelo_en_el_dashboard(app):
    """Al cambiar de modelo, la página se vuelve a dibujar sin errores."""
    opciones = app.radio[0].options
    if len(opciones) < 2:
        pytest.skip("solo hay un modelo disponible")

    app.radio[0].set_value(app.radio[0].options[1]).run()
    assert not app.exception
    # sigue detectando los dos dígitos del ejemplo
    assert {m.label: m.value for m in app.metric}["Dígitos detectados"] == "2"


def test_la_ficha_del_modelo_muestra_su_tipo_y_su_exactitud(app):
    textos = " ".join(bloque.value for bloque in app.markdown)
    assert "Tipo:" in textos
    assert "denso" in textos
    assert "Exactitud en prueba" in textos


def test_el_ejemplo_del_4_y_el_5_sale_como_dos_digitos(app):
    """El requisito de fondo: un 4 y un 5 son dos detecciones, no un 45."""
    indicadores = {m.label: m.value for m in app.metric}
    assert indicadores["Dígitos detectados"] == "2"
    assert indicadores["Confianza media"] == "100.0%"

    tabla = app.dataframe[0].value
    assert list(tabla["dígito"]) == [4, 5]


def test_el_deslizador_cambia_el_umbral(app):
    app.slider[0].set_value(50).run()
    assert not app.exception
    assert app.slider[0].value == 50
