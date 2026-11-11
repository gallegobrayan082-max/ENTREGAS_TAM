"""Pruebas de la interfaz de Streamlit.

Se ejecutan con el propio framework de Streamlit (`AppTest`), que corre la
aplicación en proceso: así se comprueba que la página se dibuja, que no
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


def test_el_ejemplo_del_4_y_el_5_sale_como_dos_digitos(app):
    """El requisito de fondo: un 4 y un 5 son dos detecciones, no un 45."""
    assert len(app.success) == 1
    assert "2" in app.success[0].value

    digitos = [
        bloque.value
        for bloque in app.markdown
        if bloque.value.startswith("<div") and "font-size" in bloque.value
    ]
    assert len(digitos) == 2
    assert "4" in digitos[0]
    assert "5" in digitos[1]


def test_el_deslizador_cambia_el_umbral(app):
    app.slider[0].set_value(50).run()
    assert not app.exception
    assert app.slider[0].value == 50
