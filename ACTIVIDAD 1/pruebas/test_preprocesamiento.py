"""Pruebas del preprocesamiento.

No necesitan el modelo entrenado: solo numpy y Pillow.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import preprocesamiento as prep


def imagen_con_cuadrado(fondo: float, tinta: float, lado: int = 8) -> np.ndarray:
    """Lienzo de 32x32 con un cuadrado de `lado`x`lado` en la esquina superior izquierda."""
    gris = np.full((32, 32), fondo, dtype=np.float32)
    gris[:lado, :lado] = tinta
    return gris


def test_umbral_otsu_separa_oscuro_de_claro():
    gris = np.concatenate(
        [np.full((20, 40), 0.9, dtype=np.float32), np.full((20, 40), 0.1, dtype=np.float32)]
    )
    corte = prep.umbral_otsu(gris)
    assert 0 <= corte <= 255
    binar = prep.binarizar(gris)
    assert binar[:20].all()
    assert not binar[20:].any()


def test_binarizar_marca_la_tinta():
    gris = imagen_con_cuadrado(fondo=0.1, tinta=0.9)
    binar = prep.binarizar(gris)
    assert binar[:8, :8].all()
    assert not binar[8:, 8:].any()


def test_invierte_cuando_el_fondo_es_claro():
    """Una foto de papel: tinta oscura sobre fondo blanco."""
    gris = imagen_con_cuadrado(fondo=0.95, tinta=0.10)
    invertida = prep.invertir_si_hace_falla(gris)
    assert invertida[0, 0] > 0.8  # la tinta pasa a ser el valor alto
    assert invertida[-1, -1] < 0.2  # el fondo pasa a ser el valor bajo


def test_no_invierte_una_imagen_estilo_mnist():
    """MNIST ya trae el dígito claro sobre fondo oscuro: no hay que tocarlo."""
    rng = np.random.default_rng(0)
    gris = np.zeros((32, 32), dtype=np.float32)
    gris[8:24, 12:20] = 1.0
    gris = np.clip(gris + rng.normal(0, 0.02, gris.shape), 0, 1).astype(np.float32)
    assert prep.invertir_si_hace_falla(gris) is gris


def test_caja_de_tinta_encuentra_el_trazado():
    gris = imagen_con_cuadrado(fondo=0.0, tinta=1.0, lado=6)
    binar = prep.binarizar(gris)
    caja = prep.caja_de_tinta(binar)
    assert caja == (0, 0, 6, 6)


def test_caja_de_tinta_sin_tinta_devuelve_none():
    binar = np.zeros((20, 20), dtype=bool)
    assert prep.caja_de_tinta(binar) is None


def test_normalizar_mnist_devuelve_28x28_en_rango():
    gris = imagen_con_cuadrado(fondo=0.0, tinta=1.0, lado=10)
    parche = prep.normalizar_mnist(gris, (0, 0, 10, 10))
    assert parche.shape == (28, 28)
    assert parche.dtype == np.float32
    assert 0.0 <= parche.min() and parche.max() <= 1.0


def test_normalizar_mnist_escala_el_digito_a_20_pixeles():
    """El recorte debe ocupar 20 de los 28 píxeles del lado más largo."""
    gris = np.zeros((40, 20), dtype=np.float32)
    gris[5:35, 2:18] = 1.0  # un trazo de 30x16
    parche = prep.normalizar_mnist(gris, (2, 5, 18, 35))
    filas = np.flatnonzero(parche.any(axis=1))
    assert filas.max() - filas.min() + 1 == 20


def test_normalizar_mnist_centra_el_digito():
    gris = np.zeros((40, 40), dtype=np.float32)
    gris[10:20, 10:20] = 1.0
    parche = prep.normalizar_mnist(gris, (10, 10, 20, 20))
    centro_filas = (np.flatnonzero(parche.any(axis=1)).min() + np.flatnonzero(parche.any(axis=1)).max()) / 2
    assert abs(centro_filas - 14) <= 1.5


def test_razon_de_tinta():
    binar = np.zeros((10, 10), dtype=bool)
    binar[:5] = True  # la mitad superior
    assert prep.razon_de_tinta(binar, (0, 0, 10, 10)) == pytest.approx(0.5)
