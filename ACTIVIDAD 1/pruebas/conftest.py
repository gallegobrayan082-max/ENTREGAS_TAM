"""Configuración común de las pruebas.

Agrega la raíz del proyecto al `sys.path` para que se pueda importar `src`
sin instalar el paquete.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
