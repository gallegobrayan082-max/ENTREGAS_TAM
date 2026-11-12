"""Pruebas del registro de modelos y de la adaptación del detector.

El dashboard tiene que poder cambiar de modelo (hoy el MLP denso, mañana la
red convolucional) sin tocar el detector. Eso se apoya en dos cosas:

- el detector adapta el lote de parches a la forma de entrada del modelo;
- el registro de modelos lee la carpeta `modelos/` y arma la ficha de cada uno.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.deteccion import DetectorDeDigitos, Parametros
from src.modelo import (
    CARPETA_MODELOS,
    NUMERO_DE_CLASES,
    describir_modelo,
    modelos_disponibles,
)


class ModeloFalso:
    """Un modelo de mentira con la forma de entrada y salida que se le pida."""

    def __init__(self, forma_entrada, forma_salida, distribuciones=None):
        self.input_shape = forma_entrada
        self.output_shape = forma_salida
        self.distribuciones = distribuciones or []
        self.recibido = None

    def predict(self, entradas, verbose=0):
        self.recibido = np.asarray(entradas)
        if self.distribuciones:
            return np.asarray(self.distribuciones[: len(entradas)])
        return np.tile(np.linspace(0, 1, self.output_shape[-1]), (len(entradas), 1))


def imagen_con_un_trazo() -> Image.Image:
    gris = np.zeros((40, 40), dtype=np.float32)
    gris[10:30, 18:22] = 1.0
    gris[26:30, 14:26] = 1.0
    return Image.fromarray((gris * 255).astype(np.uint8), mode="L")


def test_el_detector_adapta_la_entrada_a_un_modelo_denso():
    """El MLP denso recibe un vector de 784."""
    modelo = ModeloFalso((None, 784), (None, NUMERO_DE_CLASES))
    detector = DetectorDeDigitos(modelo, Parametros())
    detector.detectar(imagen_con_un_trazo())

    assert modelo.recibido is not None
    assert modelo.recibido.shape == (len(modelo.recibido), 784)


def test_el_detector_adapta_la_entrada_a_un_modelo_convolucional():
    """La CNN recibe la imagen de 28x28, sin aplanar."""
    modelo = ModeloFalso((None, 28, 28), (None, NUMERO_DE_CLASES))
    detector = DetectorDeDigitos(modelo, Parametros())
    detector.detectar(imagen_con_un_trazo())

    assert modelo.recibido is not None
    assert modelo.recibido.shape[1:] == (28, 28)


def test_el_detector_adapta_la_entrada_a_una_cnn_con_canal():
    """La CNN del `scratchpad_2.ipynb` recibe (28, 28, 1): el canal se añade."""
    modelo = ModeloFalso((None, 28, 28, 1), (None, NUMERO_DE_CLASES))
    detector = DetectorDeDigitos(modelo, Parametros())
    detector.detectar(imagen_con_un_trazo())

    assert modelo.recibido is not None
    assert modelo.recibido.shape[1:] == (28, 28, 1)


def test_un_modelo_sin_clase_fondo_no_rechaza_por_fondo():
    """Un modelo de diez clases (solo dígitos) sigue funcionando.

    La clase `fondo` es una convención: si el modelo tiene una salida más que
    dígitos, esa salida es `fondo`; si tiene exactamente diez, no hay clase
    `fondo` y el detector se apoya solo en la geometría y en la confianza.
    """
    modelo = ModeloFalso((None, 784), (None, 10))
    detector = DetectorDeDigitos(modelo, Parametros())
    assert detector._tiene_fondo is False

    # con diez clases, el "9" no se puede confundir con el fondo
    detector = DetectorDeDigitos(ModeloFalso((None, 784), (None, 11)), Parametros())
    assert detector._tiene_fondo is True
    assert detector._indice_fondo == 10


def test_la_ultima_clase_del_modelo_es_el_fondo():
    """El detector no fija el índice de `fondo`: lo toma de la salida del modelo."""
    modelo = ModeloFalso((None, 784), (None, NUMERO_DE_CLASES))
    detector = DetectorDeDigitos(modelo, Parametros())
    assert detector._indice_fondo == NUMERO_DE_CLASES - 1


def test_el_registro_encuentra_los_modelos_de_la_carpeta():
    descriptores = modelos_disponibles(CARPETA_MODELOS)
    assert descriptores, "debería haber al menos un modelo en modelos/"
    assert any(d.tipo == "denso" for d in descriptores)


def test_la_cnn_se_reconoce_como_convolucional():
    """La CNN recibe (28, 28, 1): tiene cuatro dimensiones, no tres."""
    from src.modelo import _tipo_de_forma

    assert _tipo_de_forma((None, 784)) == "denso"
    assert _tipo_de_forma((None, 28, 28)) == "convolucional"
    assert _tipo_de_forma((None, 28, 28, 1)) == "convolucional"


def test_la_ficha_del_modelo_trae_su_exactitud():
    for descriptor in modelos_disponibles(CARPETA_MODELOS):
        assert descriptor.clases == NUMERO_DE_CLASES
        # la forma de entrada tiene que ser coherente con el tipo
        if descriptor.tipo == "denso":
            assert descriptor.entrada == "784"
        else:
            assert descriptor.entrada.startswith("28x28")
        assert descriptor.exactitud is not None
        assert 0.8 < descriptor.exactitud <= 1.0
    assert modelos_disponibles(CARPETA_MODELOS), "no hay modelos"


def test_un_modelo_inexistente_da_error_claro():
    from src.modelo import cargar_modelo

    with pytest.raises(FileNotFoundError):
        cargar_modelo(Path("/tmp") / "no_existe.keras")
