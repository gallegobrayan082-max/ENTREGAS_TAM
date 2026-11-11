"""Pruebas del detector de dígitos.

Hay dos tipos de prueba:

- **unitarias**: usan un modelo falso que devuelve probabilidades fijas, así
  se puede comprobar la lógica de filtrado y de supresión sin entrenar nada;
- **de integración**: usan el modelo real sobre imágenes compuestas con dígitos
  de MNIST, que es donde se comprueba el requisito de fondo: una imagen con un
  4 y un 5 tiene que devolver **dos** detecciones, 4 y 5.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.deteccion import Deteccion, DetectorDeDigitos, Parametros, anotar
from src.preprocesamiento import caja_de_tinta

RUTA_MODELO = "modelos/mnist_mlp.keras"


class ModeloFalso:
    """Modelo de mentira: devuelve las probabilidades que se le configuren."""

    def __init__(self, probabilidades):
        self.probabilidades = np.asarray(probabilidades, dtype=float)

    def predict(self, entradas, verbose=0):
        if len(self.probabilidades) < len(entradas):
            raise AssertionError(
                f"el modelo falso recibió {len(entradas)} entradas "
                f"pero solo hay {len(self.probabilidades)} probabilidades"
            )
        return self.probabilidades[: len(entradas)]


def distribucion(digito: int, confianza: float, fondo: float = 0.0) -> list[float]:
    """Una distribución de probabilidad con toda la masa en `digito`.

    La clase 10 es la clase `fondo`, la que dice "aquí no hay un dígito".
    """
    valores = [0.0] * 11
    valores[digito] = confianza
    valores[10] = fondo
    return valores


def imagen_con_un_bloque(lado: int = 20) -> Image.Image:
    """Imagen oscura con un trazo claro en el centro, parecido a un "1"."""
    gris = np.zeros((40, 40), dtype=np.float32)
    gris[10 : 10 + lado, 18:22] = 1.0  # el trazo vertical
    gris[10 + lado - 4 : 10 + lado, 14:26] = 1.0  # la base
    return Image.fromarray((gris * 255).astype(np.uint8), mode="L")


# --------------------------------------------------------------- unitarias


def test_el_filtro_de_confianza_descarta_las_predicciones_bajas():
    """Una confianza por debajo del mínimo no se reporta."""
    detector = DetectorDeDigitos(
        ModeloFalso([distribucion(7, 0.10)] * 50),
        Parametros(umbral_confianza=0.85, umbral_bloque=0.30),
    )
    assert detector.detectar(imagen_con_un_bloque()) == []


def test_un_bloque_limpio_se_acepta_aunque_el_modelo_no_este_seguro():
    """La geometría ya dice que hay un dígito: el modelo solo dice cuál.

    Un recorte que sale de un bloque de tinta es una mancha entera, así que no
    hace falta que el modelo confirme que hay un dígito. Solo que el dígito que
    propone tenga algo de probabilidad.
    """
    detector = DetectorDeDigitos(
        ModeloFalso([distribucion(7, 0.45)] * 50),
        Parametros(umbral_confianza=0.85, umbral_bloque=0.30),
    )
    detecciones = detector.detectar(imagen_con_un_bloque())
    assert len(detecciones) == 1
    assert detecciones[0].digito == 7


def test_la_clase_fondo_descarta_un_recorte_que_no_es_un_digito():
    """Si el modelo dice `fondo`, el recorte no se reporta."""
    detector = DetectorDeDigitos(
        ModeloFalso([distribucion(10, 0.99, fondo=0.99)] * 50),
        Parametros(umbral_confianza=0.85, umbral_bloque=0.30),
    )
    assert detector.detectar(imagen_con_un_bloque()) == []


def test_la_supresion_de_no_maximos_deja_una_sola_deteccion():
    """Todas las ventanas dicen lo mismo: el resultado es un único dígito."""
    detector = DetectorDeDigitos(
        ModeloFalso([distribucion(3, 0.99)] * 500),
        Parametros(umbral_confianza=0.85),
    )
    detecciones = detector.detectar(imagen_con_un_bloque())
    assert len(detecciones) == 1
    assert detecciones[0].digito == 3


def test_el_orden_de_lectura_es_de_izquierda_a_derecha():
    detector = DetectorDeDigitos(ModeloFalso([]))
    detecciones = [
        Deteccion(digito=4, confianza=0.9, caja=(100, 10, 120, 30)),
        Deteccion(digito=5, confianza=0.9, caja=(10, 12, 30, 32)),
    ]
    ordenadas = detector._en_orden_de_lectura(detecciones)
    assert [d.digito for d in ordenadas] == [5, 4]


def test_anotar_dibuja_una_caja_por_deteccion():
    imagen = imagen_con_un_bloque()
    detecciones = [Deteccion(digito=8, confianza=0.9, caja=(10, 10, 30, 30))]
    anotada = anotar(imagen, detecciones)
    assert anotada.size == imagen.size
    original = imagen.convert("RGB")
    # el borde de la caja se dibuja, el interior no se toca
    assert anotada.getpixel((10, 20)) != original.getpixel((10, 20))
    assert anotada.getpixel((20, 20)) == original.getpixel((20, 20))


# ------------------------------------------------------------ integración


@pytest.fixture(scope="module")
def modelo():
    from src.modelo import cargar_modelo

    try:
        return cargar_modelo()
    except FileNotFoundError as error:
        pytest.skip(f"falta el modelo entrenado: {error}")


@pytest.fixture(scope="module")
def digitos_limpios(modelo):
    """Una imagen de MNIST por dígito, de las que el modelo clasifica bien."""
    from src.datos import cargar_mnist

    _, _, x_pru, y_pru = cargar_mnist()
    predicciones = modelo.predict(x_pru.reshape(-1, 784) / 255.0, verbose=0)
    correctas = predicciones.argmax(axis=1) == y_pru

    elegidas = {}
    for etiqueta in (4, 5):
        indices = np.flatnonzero(correctas & (y_pru == etiqueta))
        elegidas[etiqueta] = x_pru[indices[0]]
    return elegidas


def componer(digitos: list[np.ndarray], separacion: int = 8) -> Image.Image:
    """Pega dígitos de MNIST en una fila, como en una hoja con varios números."""
    ancho = sum(d.shape[1] for d in digitos) + separacion * (len(digitos) - 1)
    alto = max(d.shape[0] for d in digitos)
    lienzo = np.zeros((alto, ancho), dtype=np.uint8)
    x = 0
    for digito in digitos:
        lienzo[: digito.shape[0], x : x + digito.shape[1]] = digito
        x += digito.shape[1] + separacion
    return Image.fromarray(lienzo, mode="L")


def componer_juntos(digitos: list[np.ndarray]) -> Image.Image:
    """Pega dígitos pegados: recorta cada uno a su tinta y los une sin hueco."""
    recortes = []
    for digito in digitos:
        binar = digito > 127
        caja = caja_de_tinta(binar)
        recortes.append(digito[caja[1] : caja[3], caja[0] : caja[2]])
    return componer(recortes, separacion=0)


def test_una_imagen_con_un_4_y_un_5_devuelve_dos_detecciones(modelo, digitos_limpios):
    detector = DetectorDeDigitos(modelo, Parametros(umbral_confianza=0.85))
    imagen = componer([digitos_limpios[4], digitos_limpios[5]], separacion=8)

    detecciones = detector.detectar(imagen)

    assert len(detecciones) == 2
    assert {d.digito for d in detecciones} == {4, 5}
    # y además van de izquierda a derecha: el 4 primero
    assert detecciones[0].digito == 4


def test_los_digitos_pegados_no_se_concatenan(modelo, digitos_limpios):
    """El 4 y el 5 juntos siguen siendo dos detecciones, no un solo número."""
    detector = DetectorDeDigitos(modelo, Parametros(umbral_confianza=0.85))
    imagen = componer_juntos([digitos_limpios[4], digitos_limpios[5]])

    detecciones = detector.detectar(imagen)

    assert {d.digito for d in detecciones} == {4, 5}


def test_una_imagen_con_un_solo_digito_devuelve_una_deteccion(modelo, digitos_limpios):
    detector = DetectorDeDigitos(modelo, Parametros(umbral_confianza=0.85))
    imagen = componer([digitos_limpios[5]], separacion=0)

    detecciones = detector.detectar(imagen)

    assert len(detecciones) == 1
    assert detecciones[0].digito == 5


def test_una_imagen_vacia_no_devuelve_nada(modelo):
    detector = DetectorDeDigitos(modelo, Parametros(umbral_confianza=0.85))
    vacia = Image.new("L", (60, 60), color=0)
    assert detector.detectar(vacia) == []
