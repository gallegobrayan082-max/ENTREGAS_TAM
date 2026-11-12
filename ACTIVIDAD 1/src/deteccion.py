"""Detección de varios dígitos en una sola imagen.

El clasificador de MNIST solo sabe responder una cosa: *este parche de 28x28 es
un 4*. Para una imagen que puede contener varios dígitos hacen falta tres
etapas, y el orden importa porque cada una usa la anterior.

**1. Proponer candidatos** — dónde puede haber un dígito.

La tinta de la imagen se divide en *bloques*: se separa por columnas vacías y
por filas vacías, así que dos dígitos escritos uno al lado del otro quedan en
bloques distintos aunque esténPEGados con un pelo. Cada bloque se revisa:

- si tiene forma de dígito (no es más ancho que alto, no es una mota), es un
  candidato tal cual;
- si es demasiado ancho o alto, puede esconder varios dígitos, y entonces se
  divide por los valles del perfil de tinta y, si todavía sobra, se recorre con
  una ventana deslizante que busca un dígito por ventana.

**2. Clasificar cada candidato** — el recorte se normaliza al formato de MNIST
y se pasa al modelo, que devuelve la distribución de probabilidad de las diez
clases.

**3. Una detección por dígito** — los candidatos que se solapan son el mismo
dígito visto de distintas maneras: se agrupan y se queda con el mejor de cada
grupo. Aquí hay un detalle importante: una ventana que corta un dígito por la
mitad produce un *fragmento* (media barra de un 4) que el modelo clasifica con
mucha confianza, porque un fragmento se parece muchísimo a un 1. Por eso, al
elegir el representante del grupo no se mira solo la confianza, sino también
que el recorte sea **completo**.

El resultado es una lista de detecciones, cada una con su etiqueta, su
confianza y su caja. Un "45" son **dos detecciones**: un 4 y un 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import preprocesamiento as prep
from .modelo import NUMERO_DE_CLASES, NUMERO_DE_DIGITOS


@dataclass(frozen=True)
class Deteccion:
    """Un dígito encontrado en la imagen.

    `caja` usa la convención `(x0, y0, x1, y1)` en píxeles de la imagen
    original, con `x1`/`y1` exclusivos.
    """

    digito: int
    confianza: float
    caja: tuple[int, int, int, int]

    @property
    def centro(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.caja
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def ancho(self) -> int:
        return self.caja[2] - self.caja[0]

    @property
    def alto(self) -> int:
        return self.caja[3] - self.caja[1]

    @property
    def area(self) -> int:
        return self.ancho * self.alto

    def a_dict(self) -> dict[str, object]:
        """Versión serializable, para la respuesta JSON de la interfaz."""
        return {
            "digito": self.digito,
            "confianza": round(self.confianza, 4),
            "caja": list(self.caja),
        }


@dataclass
class Parametros:
    """Ajustes del detector. Los valores por defecto son los que usa la interfaz."""

    umbral_confianza: float = 0.85
    """Confianza mínima (softmax) para aceptar una detección."""

    umbral_bloque: float = 0.30
    """Confianza mínima para un recorte limpio, cuya forma ya garantiza el dígito."""

    ancho_maximo_relativo: float = 1.0
    """Un dígito nunca es más ancho que alto: más ancho que esto son varios."""

    alto_maximo_relativo: float = 1.5
    """Tolerancia vertical: un dígito puede ser algo más alto que la media."""

    alto_minimo_relativo: float = 0.45
    """Por debajo de esta fracción de la altura estimada, el bloque es una mota."""

    ancho_minimo_relativo: float = 0.12
    """Ancho mínimo del bloque, como fracción de la altura media."""

    margen_ventana: float = 1.25
    """La ventana es un poco más grande que el dígito, para no cortarlo."""

    paso_relativo: float = 0.32
    """Cada cuánto se mueve la ventana, como fracción del tamaño del dígito."""

    escalas_ventana: tuple[float, ...] = (1.0, 0.78, 0.58)
    """Tamaños de ventana probados, como múltiplos de la altura estimada."""

    iou_maximo: float = 0.30
    """Dos detecciones con IoU mayor que esto son el mismo dígito."""

    distancia_relativa: float = 0.65
    """Distancia mínima entre centros, como fracción de la altura del dígito."""

    area_minima: int = 12
    """Área mínima en píxeles para que una mancha cuente como dígito."""

    tinta_minima: float = 0.02
    """Proporción mínima de tinta en el recorte del candidato."""

    tinta_maxima: float = 0.85
    """Proporción máxima de tinta: por encima es un borrón, no un dígito."""


class DetectorDeDigitos:
    """Detecta todos los dígitos de una imagen.

    Uso típico:

    ```python
    detector = DetectorDeDigitos(cargar_modelo())
    detecciones = detector.detectar(Image.open("foto.png"))
    for d in detecciones:
        print(d.digito, d.confianza, d.caja)
    ```
    """

    def __init__(self, modelo, parametros: Parametros | None = None) -> None:
        self.modelo = modelo
        self.parametros = parametros or Parametros()

    # ------------------------------------------------------------------ público

    def detectar(self, imagen: Image.Image) -> list[Deteccion]:
        """Devuelve una detección por dígito, en orden de lectura."""
        gris = prep.invertir_si_hace_falla(prep.a_escala_grises(imagen))
        binar = prep.binarizar(gris)
        if not binar.any():
            return []

        bloques = self._bloques_de_tinta(binar)
        altura = self._estimar_altura_digito(bloques)
        if altura is None:
            return []

        cajas = self._candidatos(binar, bloques, altura)
        if not cajas:
            return []

        detecciones = self._clasificar(gris, binar, cajas)
        detecciones = self._una_deteccion_por_digito(detecciones)
        return self._en_orden_de_lectura(detecciones)

    def detectar_desde_ruta(self, ruta: str | Path) -> list[Deteccion]:
        """Atajo: detecta los dígitos de una imagen guardada en disco."""
        with Image.open(ruta) as imagen:
            return self.detectar(imagen)

    # ------------------------------------------------------------- candidatos

    def _estimar_altura_digito(self, bloques: list[tuple[int, int, int, int]]) -> float | None:
        """Estima la altura (en píxeles) de un dígito en la imagen.

        Se mide sobre los **bloques de tinta**, no sobre los componentes
        conexos: un mismo dígito puede tener varios componentes (el "5" de
        MNIST, con la barra de arriba separada del trazo de abajo), y con los
        componentes la estimación se quedaba con la mitad de la altura real.

        De los bloques se toma el más alto de los que tienen forma de dígito
        (no más anchos que altos), porque una mancha o un guion no deben
        bajar la escala de todo lo demás.
        """
        if not bloques:
            return None
        con_forma = [
            caja[3] - caja[1]
            for caja in bloques
            if (caja[2] - caja[0]) <= 1.5 * (caja[3] - caja[1])
            and (caja[3] - caja[1]) >= 3
        ]
        if con_forma:
            return float(max(con_forma))
        return float(np.median([caja[3] - caja[1] for caja in bloques]))

    def _candidatos(
        self,
        binar: np.ndarray,
        bloques: list[tuple[int, int, int, int]],
        altura: float,
    ) -> list[tuple[tuple[int, int, int, int], bool]]:
        """Genera las cajas candidatas a contener un dígito.

        Cada caja viene con una bandera: si el recorte es *limpio* (sale
        directamente de un bloque de tinta, así que la geometría ya garantiza
        que hay una sola mancha y no un fragmento) o si viene de una ventana
        deslizante (donde no se puede fiar de la geometría).
        """
        candidatos: list[tuple[tuple[int, int, int, int], bool]] = []
        for bloque in bloques:
            if self._parece_un_digito(bloque, altura):
                candidatos.append((bloque, True))
            else:
                # El bloque es demasiado grande para ser un solo dígito: puede
                # esconder varios. Se busca dígito por dígito con ventanas.
                candidatos.extend((caja, False) for caja in self._ventanas(binar, bloque, altura))
        return candidatos

    def _bloques_de_tinta(self, binar: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Divide la tinta en bloques, separando por columnas y filas vacías.

        Es una segmentación rectangular: dos dígitos que están uno al lado del
        otro quedan en columnas distintas, así que quedan en bloques
        distintos aunque la tinta llegue a tocarse.
        """
        segmentos_columna = _segmentos_con_tinta(binar.any(axis=0))
        segmentos_fila = _segmentos_con_tinta(binar.any(axis=1))

        bloques: list[tuple[int, int, int, int]] = []
        for x0, x1 in segmentos_columna:
            for y0, y1 in segmentos_fila:
                sub = binar[y0:y1, x0:x1]
                if not sub.any():
                    continue
                tinta = prep.caja_de_tinta(sub)
                if tinta is None:
                    continue
                bloques.append(
                    (x0 + tinta[0], y0 + tinta[1], x0 + tinta[2], y0 + tinta[3])
                )
        return bloques

    def _ventanas(
        self,
        binar: np.ndarray,
        bloque: tuple[int, int, int, int],
        altura: float,
    ) -> list[tuple[int, int, int, int]]:
        """Recorre un bloque con una ventana deslizante buscando un dígito cada vez.

        Es el último recurso: se usa cuando dos dígitos están tan juntos que
        ni el perfil ni los bloques logran separarlos.
        """
        x0, y0, x1, y1 = bloque
        sub = binar[y0:y1, x0:x1]
        alto_bloque, ancho_bloque = sub.shape

        cajas: list[tuple[int, int, int, int]] = []
        for escala in self.parametros.escalas_ventana:
            lado = max(6, int(round(altura * escala)))
            paso = max(2, int(round(altura * escala * self.parametros.paso_relativo)))
            for y in _posiciones(alto_bloque, lado, paso):
                for x in _posiciones(ancho_bloque, lado, paso):
                    ventana = sub[y : y + lado, x : x + lado]
                    if not ventana.any():
                        continue
                    tinta = prep.caja_de_tinta(ventana)
                    if tinta is None:
                        continue
                    ancho = tinta[2] - tinta[0]
                    alto = tinta[3] - tinta[1]
                    if ancho > self.parametros.ancho_maximo_relativo * altura:
                        continue
                    if alto > self.parametros.alto_maximo_relativo * altura:
                        continue
                    if ancho < self.parametros.ancho_minimo_relativo * altura:
                        continue
                    if alto < self.parametros.alto_minimo_relativo * altura:
                        continue
                    cajas.append(
                        (
                            x0 + x + tinta[0],
                            y0 + y + tinta[1],
                            x0 + x + tinta[2],
                            y0 + y + tinta[3],
                        )
                    )
        return cajas

    def _parece_un_digito(self, caja: tuple[int, int, int, int], altura: float) -> bool:
        """¿El bloque tiene la forma de un solo dígito?"""
        ancho = caja[2] - caja[0]
        alto = caja[3] - caja[1]
        if alto < self.parametros.alto_minimo_relativo * altura:
            return False
        if alto > self.parametros.alto_maximo_relativo * altura:
            return False
        if ancho > self.parametros.ancho_maximo_relativo * altura:
            return False
        if ancho < self.parametros.ancho_minimo_relativo * altura:
            return False
        return ancho * alto >= self.parametros.area_minima

    # ----------------------------------------------------------- clasificación

    @property
    def _numero_de_clases(self) -> int:
        """Cuántas clases tiene el modelo."""
        forma = getattr(self.modelo, "output_shape", None)
        if isinstance(forma, (list, tuple)) and forma and forma[-1]:
            return int(forma[-1])
        return NUMERO_DE_CLASES

    @property
    def _tiene_fondo(self) -> bool:
        """¿El modelo tiene la clase `fondo`?

        La convención es que el modelo tiene una salida más que dígitos: las
        diez de los dígitos y una última, `fondo`, para los recortes que no
        contienen un dígito completo. Un modelo entrenado solo sobre dígitos
        (sin esa clase) también sirve: el detector deja de usar el rechazo
        por `fondo` y se apoya solo en la geometría y en la confianza.
        """
        return self._numero_de_clases > NUMERO_DE_DIGITOS

    @property
    def _indice_fondo(self) -> int:
        """La clase `fondo` es la última del modelo."""
        return self._numero_de_clases - 1

    def _adaptar_entrada(self, parches: np.ndarray) -> np.ndarray:
        """Adapta el lote de parches a lo que el modelo espera recibir.

        Un modelo **denso** (el MLP) recibe un vector de 784; uno
        **convolucional** recibe la imagen de 28x28, con o sin canal. El
        detector sirve para los dos sin cambiar una línea, que es lo que
        permite cambiar de modelo desde el dashboard.
        """
        forma = getattr(self.modelo, "input_shape", None)
        if isinstance(forma, (list, tuple)) and len(forma) >= 3:
            if len(forma) == 4 and forma[3] == 1:
                return parches[..., np.newaxis]
            return parches
        return parches.reshape(-1, 784)

    def _clasificar(
        self,
        gris: np.ndarray,
        binar: np.ndarray,
        candidatos: list[tuple[tuple[int, int, int, int], bool]],
    ) -> list[Deteccion]:
        """Normaliza cada candidato y lo pasa por el modelo.

        Para los recortes *limpios* (una mancha de tinta que ya es un dígito
        entero) el modelo solo decide **qué** dígito es: no hace falta que
        confirme que hay un dígito, eso ya lo dijo la geometría. Por eso se
        lee la probabilidad entre los dígitos y se ignora la clase `fondo`.

        Para los recortes que salen de una ventana deslizante sí hace falta la
        clase `fondo`: ahí la geometría no garantiza nada y el modelo es el
        único que puede decir "esto es medio 4".
        """
        p = self.parametros
        vectores: list[np.ndarray] = []
        cajas_finales: list[tuple[int, int, int, int]] = []
        limpios: list[bool] = []

        for caja, limpio in candidatos:
            if prep.razon_de_tinta(binar, caja) < p.tinta_minima:
                continue
            if prep.razon_de_tinta(binar, caja) > p.tinta_maxima:
                continue
            vectores.append(prep.normalizar_mnist(gris, caja))
            cajas_finales.append(caja)
            limpios.append(limpio)

        if not vectores:
            return []

        # El formato de entrada depende del modelo: vector para el denso,
        # imagen de 28x28 para el convolucional.
        probabilidades = np.asarray(
            self.modelo.predict(self._adaptar_entrada(np.stack(vectores)), verbose=0)
        )

        digitos = probabilidades[:, :NUMERO_DE_DIGITOS]
        indice_fondo = self._indice_fondo

        detecciones: list[Deteccion] = []
        for i, caja in enumerate(cajas_finales):
            if limpios[i]:
                # La geometría ya garantiza que hay un dígito: el modelo solo
                # decide cuál es.
                etiqueta = int(digitos[i].argmax())
                confianza = float(digitos[i].max())
                if confianza < p.umbral_bloque:
                    continue
            else:
                etiqueta = int(probabilidades[i].argmax())
                confianza = float(probabilidades[i].max())
                if self._tiene_fondo and etiqueta == indice_fondo:
                    # El modelo dice que aquí no hay un dígito completo: el
                    # recorte es un fragmento o agarra más de un dígito.
                    continue
                if confianza < p.umbral_confianza:
                    continue
            detecciones.append(
                Deteccion(digito=etiqueta, confianza=confianza, caja=caja)
            )
        return detecciones

    # ---------------------------------------------------------- post-proceso

    def _una_deteccion_por_digito(self, detecciones: list[Deteccion]) -> list[Deteccion]:
        """Deja una sola detección por dígito (supresión de no máximos).

        Dos candidatos que se solapan son el mismo dígito visto de distintas
        maneras: la caja exacta de la tinta y una ventana que lo rodea. Se
        recorren de mayor a menor confianza y se conserva el primero de cada
        grupo; los que se solapan con uno ya conservado se descartan.

        El recorrido es **greedy y no transitivo** a propósito. Con un
        agrupamiento por componentes conexas, un candidato grande que abarca
        dos dígitos (un recorte ancho) unía a los dos dígitos en el mismo
        grupo y el detector reportaba un solo dígito. Con el recorrido greedy,
        el "4" y el "5" se conservan los dos porque no se solapan entre sí.
        """
        conservadas: list[Deteccion] = []
        for deteccion in sorted(detecciones, key=lambda d: -d.confianza):
            if all(not _se_solapan(deteccion, otra, self.parametros) for otra in conservadas):
                conservadas.append(deteccion)
        return conservadas

    def _en_orden_de_lectura(self, detecciones: list[Deteccion]) -> list[Deteccion]:
        """Ordena de izquierda a derecha, agrupando en filas cuando hace falta."""
        if not detecciones:
            return []
        ordenadas = sorted(detecciones, key=lambda d: (d.caja[1], d.caja[0]))
        filas: list[list[Deteccion]] = []
        for deteccion in ordenadas:
            if filas and _misma_fila(deteccion, filas[-1][-1]):
                filas[-1].append(deteccion)
            else:
                filas.append([deteccion])
        return [d for fila in filas for d in sorted(fila, key=lambda d: d.caja[0])]


# --------------------------------------------------------------------- apoyo


def _posiciones(largo: int, lado: int, paso: int) -> list[int]:
    """Posiciones de origen de una ventana de `lado` que recorre un tramo.

    Además del avance regular, se incluye la última posición alineada con el
    borde final. Sin ella, la rejilla se queda corta y el último dígito del
    bloque (el "5" de un "45" pegado) nunca entra en ninguna ventana completa.
    """
    if largo <= lado:
        return [0]
    posiciones = list(range(0, largo - lado + 1, paso))
    alineada = largo - lado
    if posiciones[-1] != alineada:
        posiciones.append(alineada)
    return posiciones


def _segmentos_con_tinta(presente: np.ndarray) -> list[tuple[int, int]]:
    """Agrupa los índices con tinta en tramos contiguos.

    `presente` es un vector de booleanos: `True` donde hay tinta. Devuelve los
    tramos como `(inicio, fin)`, con `fin` exclusivo.
    """
    tramos: list[tuple[int, int]] = []
    inicio = None
    for indice, hay in enumerate(presente):
        if hay and inicio is None:
            inicio = indice
        elif not hay and inicio is not None:
            tramos.append((inicio, indice))
            inicio = None
    if inicio is not None:
        tramos.append((inicio, len(presente)))
    return tramos


def _iou(caja_a: tuple[int, int, int, int], caja_b: tuple[int, int, int, int]) -> float:
    """Intersección sobre unión de dos cajas."""
    x0 = max(caja_a[0], caja_b[0])
    y0 = max(caja_a[1], caja_b[1])
    x1 = min(caja_a[2], caja_b[2])
    y1 = min(caja_a[3], caja_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    interseccion = (x1 - x0) * (y1 - y0)
    area_a = (caja_a[2] - caja_a[0]) * (caja_a[3] - caja_a[1])
    area_b = (caja_b[2] - caja_b[0]) * (caja_b[3] - caja_b[1])
    union = area_a + area_b - interseccion
    return interseccion / union if union > 0 else 0.0


def _se_solapan(a: Deteccion, b: Deteccion, parametros: Parametros) -> bool:
    """¿Son estas dos detecciones el mismo dígito?"""
    if _iou(a.caja, b.caja) > parametros.iou_maximo:
        return True
    distancia = math.hypot(a.centro[0] - b.centro[0], a.centro[1] - b.centro[1])
    return distancia < parametros.distancia_relativa * min(a.alto, b.alto)


def _misma_fila(a: Deteccion, b: Deteccion) -> bool:
    """¿Comparten fila vertical? Se usa para el orden de lectura."""
    arriba = max(a.caja[1], b.caja[1])
    abajo = min(a.caja[3], b.caja[3])
    solape = abajo - arriba
    if solape <= 0:
        return False
    return solape >= 0.4 * min(a.alto, b.alto)


def anotar(imagen: Image.Image, detecciones: list[Deteccion]) -> Image.Image:
    """Devuelve una copia de la imagen con las cajas y las etiquetas dibujadas."""
    anotada = imagen.convert("RGB").copy()
    dibujo = ImageDraw.Draw(anotada)
    for deteccion in detecciones:
        x0, y0, x1, y1 = deteccion.caja
        color = (34, 197, 94)
        dibujo.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
        etiqueta = f"{deteccion.digito} ({deteccion.confianza:.0%})"
        dibujo.text((x0 + 2, max(0, y0 - 12)), etiqueta, fill=color)
    return anotada
