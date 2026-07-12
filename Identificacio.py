
"""
pruebaIdentificacionV7_10.py

Segmenta las señales de aceleración X/Y e identifica comportamientos
anómalos mediante comparación con la base de datos de patrones normales.

"""

import matplotlib
matplotlib.use("TkAgg")

import re
import sqlite3
import io
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from tksheet import Sheet


##################################################
# DATOS BÁSICOS
##################################################

Fs = 1000

experimento = 125
bloque = 0

# True  -> carga todos los timeblock*.txt de la carpeta seleccionada
# False -> carga solo timeblock{bloque}.txt
CARGAR_TODOS_LOS_BLOQUES = False

CONVERTIR_A_MS2 = False
G0 = 9.80665

APLICAR_RESTA_MEDIA = True

RUTA_SCRIPT = Path(__file__).resolve().parent
RUTA_DATOS = RUTA_SCRIPT.parent


##################################################
# FILTRADO
##################################################

APLICAR_FILTRO_PASO_BAJO = True

# Se usan dos filtrados:
# - suave: segmentación inicial y detección de candidatos;
# - validación: confirmación y ajuste de límites de desplazamiento.
FILTRO_SEGMENTACION_SUAVE_HZ = 150.0
FILTRO_VALIDACION_DESPLAZAMIENTO_HZ = 20.0

ORDEN_FILTRO = 2


##################################################
# SEGMENTACIÓN SOLO XY
##################################################

# El eje Z NO se usa para segmentar.
# Z solo se muestra en la tercera gráfica.
EJES_SEGMENTACION = ["x", "y"]
IDX_EJE = {
    "x": 0,
    "y": 1,
    "z": 2,
}

# Umbral de desplazamiento.
# No se calcula a partir de UMBRAL_REPOSO, sino a partir del máximo absoluto
# medido en el experimento de referencia en reposo.
#
#   UMBRAL_REPOSO[eje] = FACTOR_UMBRAL_REPOSO * MAX_ABS_REPOSO[eje]
#   UMBRAL_DESPLAZAMIENTO[eje] = FACTOR_UMBRAL_DESPLAZAMIENTO * MAX_ABS_REPOSO[eje]
#
# Criterio final:
#   |a| <= UMBRAL_REPOSO[eje]                         -> reposo
#   UMBRAL_REPOSO[eje] < |a| < UMBRAL_DESPLAZAMIENTO[eje] -> oscilatorio
#   |a| >= UMBRAL_DESPLAZAMIENTO[eje]                 -> desplazamiento
FACTOR_UMBRAL_DESPLAZAMIENTO = 6


# Banda muerta para cruces por cero.
# Evita que el ruido cerca de 0 cree cortes falsos.
#
# Estos valores iniciales se sustituyen automáticamente por el umbral de reposo
# calculado a partir del experimento de referencia.
SIGN_DEADBAND = {
    "x": 0.006,
    "y": 0.006,
}

# ------------------------------------------------
# UMBRAL DE REPOSO DESDE EXPERIMENTO DE REFERENCIA
# ------------------------------------------------
# Para el experimento de referencia:
# 1) se cargan X/Y,
# 2) se quita la media a cada eje,
# 3) NO se filtra la señal de referencia,
# 4) se calcula el máximo absoluto de cada eje sobre la señal centrada:
#       max_abs_x = max(abs(x - mean(x)))
#       max_abs_y = max(abs(y - mean(y)))
# 5) se multiplica cada valor por FACTOR_UMBRAL_REPOSO.
#
# No se usa pico a pico.
EXPERIMENTO_REFERENCIA_REPOSO = 149
FACTOR_UMBRAL_REPOSO = 1.7

# Aquí se guarda el máximo absoluto real de la señal de referencia en reposo
# tras quitar la media, pero SIN aplicar filtro.
# Se usa como base común para calcular UMBRAL_REPOSO y UMBRAL_DESPLAZAMIENTO.
MAX_ABS_REPOSO = {
    "x": SIGN_DEADBAND["x"],
    "y": SIGN_DEADBAND["y"],
}

# Aquí se guarda el umbral de reposo calculado.
UMBRAL_REPOSO = SIGN_DEADBAND.copy()


VENTANA_ZC_MS = 100
MIN_CRUCES_OSC = 5

# Validación de candidatos:
# Los candidatos a desplazamiento detectados con el filtro suau se revisan
# con la señal filtrada agresivamente. Si no superan UMBRAL_DESPLAZAMIENTO
# en la señal de validación, se reclasifican como oscilatorios.
MARGEN_VALIDACION_DESPLAZAMIENTO_MS = 0

# Duraciones mínimas.
MIN_DUR_NORMAL_MS = 50
MIN_DUR_OSC_MS = 40

# Corrección final: un desplazamiento demasiado corto se considera oscilatorio.
# Se aplica al final del postprocesado de los segmentos rojos, para medir la
# duració real del segmento ya ajustado/recortado.
CONVERTIR_DESPLAZAMIENTO_CORTO_A_OSCILATORIO = False
MAX_DUR_DESPLAZAMIENTO_CORTO_MS = 50

# Fusión de huecos dentro de un mismo tipo.
MAX_GAP_NORMAL_MS = 35
MAX_GAP_OSC_MS = 120
FUSIONAR_OSCILATORIOS_CERCANOS_FINAL = True
MAX_GAP_OSC_FINAL_MS = 120

# División adicional de desplazamientos por valle cercano a cero.
# Sirve para separar dos lóbulos de desplazamiento cuando entre ambos la señal
# vuelve claramente cerca de cero, aunque no exista un cruce de signo estable.
DIVIDIR_NORMAL_POR_VALLE_CERCA_CERO = True
MIN_DUR_VALLE_CERO_NORMAL_MS = 5
VENTANA_LOBULO_VALLE_MS = 90
FACTOR_VALLE_CERO_NORMAL = 1.00
FACTOR_PICO_LOBULO_VALLE_NORMAL = 1.00

# División híbrida de desplazamientos.
# La señal de 20 Hz valida que existe desplazamiento y detecta valles internos;
# la señal de filtro suau da el punto fino de corte, porque conserva mejor los pasos
# por cero que pueden desaparecer con el filtrado agresivo.
DIVIDIR_DESPLAZAMIENTO_HIBRIDO = True
VENTANA_LOBULO_DIVISION_HIBRIDA_MS = 120
VENTANA_BUSQUEDA_CORTE_SUAVE_MS = 35
MIN_SEPARACION_CORTES_HIBRIDA_MS = 35
FACTOR_VALLE_ABS_20HZ = 0.60
FACTOR_CAIDA_RELATIVA_VALLE_20HZ = 0.72
FACTOR_PICO_LOBULO_DIVISION_HIBRIDA = 0.80

# Margen añadido a cada segmento.
EXPAND_MS = 10

# Retoque de bordes:
# una vez detectado un segmento normal por umbral, sus bordes se desplazan
# al cruce por cero más cercano. Esto no cambia el criterio de detección,
# solo el instante exacto de inicio/fin del segmento.
AJUSTAR_LIMITES_A_CRUCE_CERO = True
MAX_EXTENSION_CRUCE_CERO_MS = 120

# También ajusta a cero los segmentos oscilatorios amarillos.
# Importante para evitar que el amarillo empiece cuando ya hay amplitud alta.
AJUSTAR_OSCILATORIOS_A_CRUCE_CERO = True


# Prioridad de lóbulos de movimiento frente a oscilación:
# Un tramo con cruces por cero puede parecer oscilatorio por el ringing,
# pero si dentro hay un lóbulo claro de movimiento, se mantiene como normal.
PRIORIDAD_LOBULO_NORMAL_SOBRE_OSCILACION = True

# Para decidir si un tramo detectado por umbral dentro de una zona oscilatoria
# es realmente un lóbulo de movimiento y no una oscilación sostenida.
MIN_DUR_LOBULO_NORMAL_EN_OSC_MS = 45
# Se permite hasta 3 cruces porque algunos lóbulos reales tienen
# un pequeño rebote al entrar/salir de una zona oscilatoria.
MAX_CRUCES_LOBULO_NORMAL_EN_OSC = 3
FACTOR_PICO_LOBULO_NORMAL_EN_OSC = 1.00

# Si el lóbulo tiene un pico muy claro, se mantiene como movimiento normal
# aunque tenga algún cruce adicional provocado por el rizado de entrada/salida.
PERMITIR_LOBULO_CLARO_AUNQUE_TENGA_CRUCES = True
FACTOR_PICO_LOBULO_CLARO_EN_OSC = 2.50

# Si un lóbulo normal cae dentro de un amarillo, se recorta el amarillo
# para que el lóbulo quede visualmente como segmento normal independiente.
RECORTAR_OSCILATORIO_CON_LOBULOS_NORMALES = True


# Repòs visual.
MIN_DURACION_REPOSO_VISUAL_MS = 120

# Escala común en la gráfica de aceleración:
# X, Y y Z comparten los mismos límites verticales.
MISMA_ESCALA_Y_ACELERACION = True

# Escala vertical:
# True  -> usa el máximo absoluto real de X/Y/Z, no recorta picos.
# False -> usa percentil 99.7, útil si quieres ignorar picos muy aislados.
USAR_MAXIMO_REAL_ESCALA_Y = True
FACTOR_MARGEN_ESCALA_Y = 1.10

# Etiquetas de reposo. Se crean una sola vez en la segmentación y se reutilizan
# después en la clasificación para no renombrar los segmentos.
MOSTRAR_ETIQUETAS_REPOSO = True

# Detección visual de impactos.
# Un impacto NO corta ni modifica los segmentos de reposo, oscilatorio o
# desplazamiento. Solo se pinta encima en naranja cuando el módulo de la
# aceleración supera el umbral indicado.
DETECTAR_IMPACTOS = True
UMBRAL_IMPACTO_G = 4.0
USAR_SENAL_SIN_FILTRAR_PARA_IMPACTOS = True
MIN_DUR_IMPACTO_MS = 1
MAX_GAP_IMPACTO_MS = 5
EXPAND_IMPACTO_MS = 3


# ------------------------------------------------
# RECORTE DE INICIO DE SEGMENTOS NORMALES
# ------------------------------------------------
# Corrige casos donde el ajuste al cruce por cero mueve demasiado hacia atrás
# el inicio de un segmento normal.
#
# Lógica:
# 1) Se busca dentro del segmento el primer punto donde |a| supera el
#    umbral de desplazamiento de forma sostenida.
# 2) Si antes de ese punto hay un prefijo suficientemente largo sin movimiento
#    fuerte, se recorta el inicio.
# 3) El nuevo inicio se recoloca en el cruce por cero cercano a ese primer
#    movimiento sostenido, pero con una búsqueda mucho más corta que el ajuste
#    general a cero.
RECORTAR_INICIO_NORMAL_POR_ACTIVIDAD_SOSTENIDA = True

# Tiempo mínimo de tramo inicial por debajo del umbral de desplazamiento para permitir recorte.
MIN_PREFIJO_REPOSO_RECORTE_MS = 45

# Ventana usada para decidir que el movimiento ya es sostenido.
VENTANA_ACTIVIDAD_SOSTENIDA_MS = 18

# Porcentaje mínimo de mostres dentro de la ventana que deben superar el umbral de desplazamiento.
FRACCION_ACTIVIDAD_SOSTENIDA = 0.8

# Búsqueda hacia atrás del cruce por cero desde el inicio sostenido.
# Es mucho menor que MAX_EXTENSION_CRUCE_CERO_MS para evitar volver demasiado atrás.
MAX_BUSQUEDA_CERO_RECORTE_MS = 35

# Recorte adicional de inicio de desplazamientos:
# después de ajustar un segmento normal al cruce por cero, el inicio no debe
# quedar antes del último cruce por cero previo al primer punto candidato a
# desplazamiento. Así se evita que el segmento rojo empiece demasiado pronto
# por culpa del margen o de un cruce de cero anterior.
RECORTAR_INICIO_NORMAL_AL_CERO_PREVIO_CANDIDATO = True

# Para el recorte anterior, no se permite buscar un cero demasiado lejano.
# El inicio se recoloca en el último punto cercano a cero justo antes del
# primer candidato real a desplazamiento.
MAX_BUSQUEDA_CERO_PREVIO_CANDIDATO_MS = 100


##################################################
# CLASIFICACIÓN POR DTW INDEPENDIENTE X/Y
##################################################

USAR_CLASIFICACION_DTW = True

# Cada eje se clasifica por separado: segmentos X con señal X, segmentos Y con señal Y.
EJES_CLASIFICACION_DTW = ["x", "y"]


LONGITUD_DTW = 120

# Umbral de distància por eje.
DISTANCIA_CLUSTER_DTW = {"x": 0.20, "y": 0.20}


USAR_DTW = True
DTW_BANDA_PORC = 0.15

# La forma se compara con DTW sobre la señal z-normalizada, pero además
# se penalizan diferencias físicas entre segmentos.
# Esto evita que dos lóbulos con forma parecida, pero con amplitud o duración
# claramente distintas, acaben en el mismo grupo.
PESO_FORMA_DTW = 1.00
PESO_RMS_DTW = 0
PESO_PICO_DTW = 0.25
PESO_DURACION_DTW = 0.25

# Si dos segmentos tienen pico dominante de signo contrario, se añade una
# penalización fuerte. Normalmente no debería hacer falta, pero protege
# contra agrupaciones de movimientos opuestos.
PENALIZAR_SIGNO_PICO_DTW = True
PENALIZACION_SIGNO_PICO_DTW = 1.00

MOSTRAR_MATRIZ_DISTANCIAS_DTW = True

CMAP_MATRIZ_DISTANCIAS = "viridis"
ANOTAR_VALORES_MATRIZ = True
MAX_SEGMENTOS_ANOTAR_MATRIZ = 9999
DECIMALES_MATRIZ_DISTANCIAS = 2


##################################################
# IDENTIFICACIÓN CON BASE DE DATOS
##################################################

# Base de dades generada con el programa de entrenamiento.
NOMBRE_BD_PATRONES_NORMALES = "patrones_normales_dtw_V7.sqlite"


# ------------------------------------------------------------
# Comparación frecuencial por bandas
# ------------------------------------------------------------

# En desplazamientos, la FORMA solo se usa para asignar el segmento
# al grupo de desplazamiento más parecido.
# Si se deja en None, se usa DISTANCIA_CLUSTER_DTW[eje].
UMBRAL_FORMA_CLASIFICACION_DESPLAZAMIENTO_BD = {"x": None, "y": None}

# Longitud común para comparar espectros.
LONGITUD_ESPECTRO_IDENTIFICACION_BD = 120

# Freqüència máxima usada para comparar y dibujar espectros de segmentos.
# None -> usa todo el espectro hasta Fs/2.
FREQ_MAX_COMPARACION_IDENTIFICACION_BD = 500.0


# Características que se validan en oscilatorio y reposo.

# Criterio de asignación:
# - Los desplazamientos buscan el grupo de desplazamiento más cercano.
# - Los oscilatorios NO buscan grupo: se comparan directamente con el
#   patrón oscilatorio de la BD del mismo eje.
# - Los reposos NO buscan grupo: se comparan directamente con el
#   patrón reposo de la BD del mismo eje.
TIPOS_COMPARACION_DIRECTA_BD = {"oscilatorio", "reposo"}

# Si True, se avisa por consola de los motivos de anomalía.
IMPRIMIR_RESUMEN_ANOMALIAS_BD = True


##################################################
# TEXTOS VISIBLES EN VALENCIÀ
##################################################

def etiqueta_tipus_patro(tipo_patron):
    """Traduïx el tipus intern del patró només per a mostrar-lo."""
    traduccions = {
        "desplazamiento": "Desplaçament",
        "oscilatorio": "Oscil·latori",
        "reposo": "Repòs",
        "senyal_completa": "Senyal completa",
        "normal": "Normal",
    }
    tipus = str(tipo_patron or "")
    return traduccions.get(tipus, tipus.capitalize())


def etiqueta_caracteristica_bd(nombre):
    """Nom llegible en valencià per als avisos i les taules."""
    traduccions = {
        "duracion_s": "duració",
        "rms_original": "RMS original",
        "pico_pico_original": "pic-pic original",
        "rizado_pico_pico_original": "ondulació pic pic original",
        "offset_abs_original": "òfset absolut original",
        "dtw_distancia": "distància DTW",
    }
    return traduccions.get(str(nombre), str(nombre))


def etiqueta_columna_bd(columna):
    """Capçalera valenciana per al visor de la base de dades."""
    traduccions = {
        "id": "ID",
        "tipo_patron": "Tipus de patró",
        "eje": "Eix",
        "n_segmentos": "Nombre de segments",
        "duracion_s_min": "Duració mín. [s]",
        "duracion_s_max": "Duració màx. [s]",
        "rms_original_max": "RMS original màx.",
        "pico_pico_original_max": "Pic-pic original màx.",
        "rizo_pico_pico_original_max": "Ondulació pic pic original màx.",
        "rizado_pico_pico_original_max": "Ondulació pic pic original màx.",
        "offset_abs_original_max": "Òfset absolut original màx.",
    }

    columna = str(columna)

    if columna in traduccions:
        return traduccions[columna]

    coincidencia = re.fullmatch(
        r"energia_banda_(\d+)_(\d+)_original_max",
        columna
    )
    if coincidencia:
        f_ini, f_fin = coincidencia.groups()
        return f"Energia {f_ini}-{f_fin} Hz original màx."

    return columna


##################################################
# RUTAS
##################################################

def obtener_ruta_base(num_experimento):
    if 64 < num_experimento <= 113:
        return (
            RUTA_DATOS
            / "pruebas_lab"
            / "pruebasExperimentalesDocumentadas"
            / f"Experimento{num_experimento}"
        )

    elif 114 <= num_experimento <= 119 or 122 < num_experimento:
        return (
            RUTA_DATOS
            / "pruebas_planta"
            / f"Experimento{num_experimento}"
        )

    else:
        raise ValueError(
            f"Experiència {num_experimento} fora del rang previst. "
            "Rangs vàlids: 65-113 per al pòrtic, 114-119 i >122 per a la màquina real."
        )


def extraer_numero_bloque(path):
    m = re.search(r"timeblock(\d+)\.txt", path.name)

    if m is None:
        return -1

    return int(m.group(1))


##################################################
# CARGA DE DATOS DESDE CARPETAS
##################################################

def leer_aceleracion_archivo(archivo):
    data = np.loadtxt(archivo)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 5:
        raise ValueError(
            f"{archivo.name} ha de tindre almenys 5 columnes: "
            "temps, temps, accX, accY, accZ"
        )

    acc = data[:, 2:5].copy()

    if CONVERTIR_A_MS2:
        acc = acc * G0

    return acc


def obtener_archivos_timeblock_carpeta(carpeta):
    if CARGAR_TODOS_LOS_BLOQUES:
        archivos = sorted(
            carpeta.glob("timeblock*.txt"),
            key=extraer_numero_bloque
        )

        if len(archivos) == 0:
            raise FileNotFoundError(f"No hi ha cap timeblock*.txt en:\n{carpeta}")

        return archivos

    archivo = carpeta / f"timeblock{bloque}.txt"

    if not archivo.exists():
        raise FileNotFoundError(f"No s'ha trobat:\n{archivo}")

    return [archivo]


def filtrar_paso_bajo(acc, frecuencia_corte_hz):
    if not APLICAR_FILTRO_PASO_BAJO:
        return acc.copy()

    nyq = Fs / 2.0
    Wn = frecuencia_corte_hz / nyq

    if Wn <= 0 or Wn >= 1:
        raise ValueError("La freqüència de tall ha d’estar entre 0 i Fs/2.")

    sos = butter(
        ORDEN_FILTRO,
        Wn,
        btype="lowpass",
        output="sos"
    )

    try:
        return sosfiltfilt(sos, acc, axis=0)
    except ValueError:
        print("[AVÍS] Senyal massa curta per a filtrar. Es retorna sense filtrar.")
        return acc.copy()


def cargar_datos_carpeta(carpeta):
    archivos = obtener_archivos_timeblock_carpeta(carpeta)

    bloques = []

    print(f"\nCarregant la carpeta: {carpeta.name}")

    for archivo in archivos:
        acc = leer_aceleracion_archivo(archivo)
        bloques.append(acc)
        print(f"  {archivo.name}: {len(acc)} mostres")

    acc_sin_filtrar = np.vstack(bloques)

    if APLICAR_RESTA_MEDIA:
        media_global = np.mean(acc_sin_filtrar, axis=0, keepdims=True)
        acc_sin_filtrar = acc_sin_filtrar - media_global
    else:
        media_global = np.zeros((1, 3))

    acc_filtrada_suave = filtrar_paso_bajo(
        acc_sin_filtrar,
        FILTRO_SEGMENTACION_SUAVE_HZ
    )
    acc_filtrada_agresiva = filtrar_paso_bajo(
        acc_sin_filtrar,
        FILTRO_VALIDACION_DESPLAZAMIENTO_HZ
    )

    t = np.arange(len(acc_sin_filtrar)) / Fs

    unidad_acc = "m/s²" if CONVERTIR_A_MS2 else "g"

    print(f"  Total de mostres: {len(t)}")
    print(f"  Duració: {t[-1]:.3f} s")

    if APLICAR_RESTA_MEDIA:
        print(
            "  Mitjana restada [X,Y,Z]: "
            f"{media_global[0, 0]:.8f}, "
            f"{media_global[0, 1]:.8f}, "
            f"{media_global[0, 2]:.8f} {unidad_acc}"
        )

    if APLICAR_FILTRO_PASO_BAJO:
        print(
            "  Filtres de pas baix: "
            f"suau {FILTRO_SEGMENTACION_SUAVE_HZ:.1f} Hz | "
            f"validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.1f} Hz | "
            f"ordre {ORDEN_FILTRO}"
        )

    return t, acc_sin_filtrar, acc_filtrada_suave, acc_filtrada_agresiva, unidad_acc


##################################################
# UMBRAL DE REPOSO DESDE EXPERIMENTO DE REFERENCIA
##################################################

def leer_aceleracion_archivo_referencia(archivo):
    """
    Lee la aceleración del experimento de referencia sin aplicar resta de media.
    Para calcular el umbral de reposo se quitará la media explícitamente
    después de concatenar todos los bloques.
    """
    data = np.loadtxt(archivo)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 5:
        raise ValueError(
            f"{archivo.name} ha de tindre almenys 5 columnes: "
            "temps, temps, accX, accY, accZ"
        )

    acc = data[:, 2:5].copy()

    if CONVERTIR_A_MS2:
        acc = acc * G0

    return acc


def obtener_carpetas_con_timeblocks(ruta_base):
    """
    Devuelve carpetas que contienen timeblock*.txt.
    Sirve para calcular automáticamente el umbral de reposo del experimento
    de referencia sin pedir selección por consola.
    """
    ruta_base = Path(ruta_base)

    if list(ruta_base.glob("timeblock*.txt")):
        return [ruta_base]

    carpetas_directas = sorted([
        p for p in ruta_base.iterdir()
        if p.is_dir() and list(p.glob("timeblock*.txt"))
    ])

    if len(carpetas_directas) > 0:
        return carpetas_directas

    carpetas_recursivas = sorted({
        p.parent for p in ruta_base.rglob("timeblock*.txt")
    })

    return carpetas_recursivas


def cargar_datos_referencia_experimento(num_experimento):
    ruta_ref = obtener_ruta_base(num_experimento)

    if not ruta_ref.exists():
        raise FileNotFoundError(
            f"No existeix la ruta de l'experiment de referència {num_experimento}:\n{ruta_ref}"
        )

    carpetas_ref = obtener_carpetas_con_timeblocks(ruta_ref)

    if len(carpetas_ref) == 0:
        raise FileNotFoundError(
            f"No s'han trobat fitxers timeblock*.txt en l'experiment de referència {num_experimento}:\n{ruta_ref}"
        )

    bloques = []
    n_archivos = 0

    for carpeta in carpetas_ref:
        archivos = sorted(
            carpeta.glob("timeblock*.txt"),
            key=extraer_numero_bloque
        )

        for archivo in archivos:
            bloques.append(leer_aceleracion_archivo_referencia(archivo))
            n_archivos += 1

    if n_archivos == 0:
        raise FileNotFoundError(
            f"No s'ha pogut carregar cap timeblock*.txt de l'experiment {num_experimento}."
        )

    acc_ref = np.vstack(bloques)

    return acc_ref, ruta_ref, carpetas_ref, n_archivos


def calcular_umbral_reposo_desde_experimento(num_experimento):
    """
    Calcula el umbral de reposo a partir del máximo absoluto de X/Y
    tras quitar la media de cada eje.

    La señal de referencia de reposo NO se filtra. No usa pico a pico.
    """
    acc_ref, ruta_ref, carpetas_ref, n_archivos = cargar_datos_referencia_experimento(
        num_experimento
    )

    media_ref = np.mean(acc_ref, axis=0)
    acc_ref_centrada = acc_ref - media_ref

    max_abs_x = float(np.max(np.abs(acc_ref_centrada[:, IDX_EJE["x"]])))
    max_abs_y = float(np.max(np.abs(acc_ref_centrada[:, IDX_EJE["y"]])))

    max_abs = {
        "x": max_abs_x,
        "y": max_abs_y,
    }

    umbral = {
        "x": FACTOR_UMBRAL_REPOSO * max_abs_x,
        "y": FACTOR_UMBRAL_REPOSO * max_abs_y,
    }

    umbral_desplazamiento = {
        "x": FACTOR_UMBRAL_DESPLAZAMIENTO * max_abs_x,
        "y": FACTOR_UMBRAL_DESPLAZAMIENTO * max_abs_y,
    }

    info = {
        "ruta_ref": ruta_ref,
        "n_carpetas": len(carpetas_ref),
        "n_archivos": n_archivos,
        "media_x": float(media_ref[IDX_EJE["x"]]),
        "media_y": float(media_ref[IDX_EJE["y"]]),
        "max_abs_x": max_abs_x,
        "max_abs_y": max_abs_y,
        "max_abs": max_abs,
        "umbral": umbral,
        "umbral_desplazamiento": umbral_desplazamiento,
    }

    return umbral, info


def actualizar_umbral_reposo_desde_experimento_referencia():
    """
    Actualiza MAX_ABS_REPOSO, UMBRAL_REPOSO y SIGN_DEADBAND con el umbral
    calculado desde el experimento de referencia en reposo.

    La referencia se centra quitando la media, pero NO se filtra.
    """
    global SIGN_DEADBAND, UMBRAL_REPOSO, MAX_ABS_REPOSO

    umbral, info = calcular_umbral_reposo_desde_experimento(
        EXPERIMENTO_REFERENCIA_REPOSO
    )

    MAX_ABS_REPOSO = info["max_abs"].copy()
    UMBRAL_REPOSO = umbral.copy()
    SIGN_DEADBAND = umbral.copy()

    unidad_acc = "m/s²" if CONVERTIR_A_MS2 else "g"

    print("\nLlindar de repòs calculat a partir de l'experiment de referència")
    print("-" * 80)
    print(f"Experiència de referència: {EXPERIMENTO_REFERENCIA_REPOSO}")
    print(f"Ruta de referència:        {info['ruta_ref']}")
    print(f"Carpetes utilitzades:      {info['n_carpetas']}")
    print(f"Fitxers utilitzats:        {info['n_archivos']}")
    print(f"Mitjana X aplicada:        {info['media_x']:.8f} {unidad_acc}")
    print(f"Mitjana Y aplicada:        {info['media_y']:.8f} {unidad_acc}")
    print(f"Màx. abs. X ref. sense filtrar: {info['max_abs_x']:.8f} {unidad_acc}")
    print(f"Màx. abs. Y ref. sense filtrar: {info['max_abs_y']:.8f} {unidad_acc}")
    print(f"Factor de repòs:           {FACTOR_UMBRAL_REPOSO:.3f}")
    print(f"Factor de desplaçament:    {FACTOR_UMBRAL_DESPLAZAMIENTO:.3f}")
    print(f"Llindar de repòs X:        {UMBRAL_REPOSO['x']:.8f} {unidad_acc}")
    print(f"Llindar de repòs Y:        {UMBRAL_REPOSO['y']:.8f} {unidad_acc}")
    print(f"Llindar de desplaç. X:     {info['umbral_desplazamiento']['x']:.8f} {unidad_acc}")
    print(f"Llindar de desplaç. Y:     {info['umbral_desplazamiento']['y']:.8f} {unidad_acc}")
    print("-" * 80)


def umbral_desplazamiento_eje(eje):
    """
    Umbral real para clasificar desplazamiento.

    Se calcula directamente desde el máximo absoluto del experimento de
    referencia en reposo, no desde UMBRAL_REPOSO.

    UMBRAL_REPOSO[eje] = FACTOR_UMBRAL_REPOSO * MAX_ABS_REPOSO[eje]
    UMBRAL_DESPLAZAMIENTO[eje] = FACTOR_UMBRAL_DESPLAZAMIENTO * MAX_ABS_REPOSO[eje]
    """
    return FACTOR_UMBRAL_DESPLAZAMIENTO * MAX_ABS_REPOSO[eje]


def umbral_impacto_unidades_actuales():
    """
    Devuelve el umbral de impacto en las unidades actuales de la señal.
    Si la señal está en g, el umbral es 4 g.
    Si la señal se ha convertido a m/s², el umbral equivalente es 4*g0.
    """
    if CONVERTIR_A_MS2:
        return UMBRAL_IMPACTO_G * G0

    return UMBRAL_IMPACTO_G


def detectar_impactos_modulo(acc):
    """
    Detecta impactos como intervalos donde el módulo de la aceleración supera
    UMBRAL_IMPACTO_G.

    El módulo se calcula con los tres ejes disponibles:
        |a| = sqrt(ax² + ay² + az²)

    Esta detección es únicamente visual: los impactos se pintan encima de la
    segmentación existente y no parten ni modifican los segmentos de reposo,
    oscilación o desplazamiento.
    """
    if not DETECTAR_IMPACTOS:
        return []

    acc = np.asarray(acc, dtype=float)
    th = umbral_impacto_unidades_actuales()

    if acc.ndim == 1:
        modulo = np.abs(acc)
    else:
        modulo = np.sqrt(np.sum(acc[:, :3] ** 2, axis=1))

    mask_impacto = modulo >= th
    segmentos = mascara_a_segmentos(mask_impacto)

    segmentos = fusionar_segmentos_por_hueco(
        segmentos,
        int(MAX_GAP_IMPACTO_MS * Fs / 1000)
    )

    segmentos = eliminar_segmentos_cortos(
        segmentos,
        int(MIN_DUR_IMPACTO_MS * Fs / 1000)
    )

    segmentos = aplicar_margenes_segmentos(
        segmentos,
        len(modulo),
        int(EXPAND_IMPACTO_MS * Fs / 1000)
    )

    return segmentos


##################################################
# FFT
##################################################

def calcular_espectro_fft(acc):
    N = len(acc)

    if N < 4:
        raise ValueError("La senyal és massa curta per a calcular la FFT.")

    ventana = np.hanning(N)
    ganancia = np.mean(ventana)

    f = np.fft.rfftfreq(N, d=1 / Fs)
    espectro = np.zeros((len(f), 3))

    for eje in range(3):
        x = acc[:, eje] - np.mean(acc[:, eje])
        xw = x * ventana

        amp = np.abs(np.fft.rfft(xw)) / (N * ganancia)

        if len(amp) > 2:
            amp[1:-1] *= 2

        espectro[:, eje] = amp

    return f, espectro


##################################################
# UTILIDADES DE SEGMENTACIÓN
##################################################

def rms_movil(x, ventana_muestras):
    ventana_muestras = max(1, int(ventana_muestras))
    kernel = np.ones(ventana_muestras) / ventana_muestras
    return np.sqrt(np.convolve(x * x, kernel, mode="same"))


def signo_estable(x, deadband):
    s = np.zeros(len(x), dtype=int)
    s[x > deadband] = 1
    s[x < -deadband] = -1

    ultimo = 0

    for i in range(len(s)):
        if s[i] != 0:
            ultimo = s[i]
        else:
            s[i] = ultimo

    return s


def calcular_cruces_y_rms_ventana(x, eje):
    """
    Calcula, para cada muestra, el número de cruces por cero y el RMS dentro
    de una ventana centrada. Estos valores se usan para debug y para decidir
    si una zona con mucha actividad debe reclasificarse como oscilatoria.
    """
    ventana = int(VENTANA_ZC_MS * Fs / 1000)
    ventana = max(3, ventana)

    s = signo_estable(x, SIGN_DEADBAND[eje])

    cruces = np.zeros(len(x), dtype=float)
    cruces[1:] = (
        (s[1:] != s[:-1])
        & (s[1:] != 0)
        & (s[:-1] != 0)
    )

    cruces_ventana = np.convolve(cruces, np.ones(ventana), mode="same")
    rms_ventana = rms_movil(x, ventana)

    return cruces_ventana, rms_ventana


def mascara_a_segmentos(mask):
    segmentos = []
    N = len(mask)
    i = 0

    while i < N:
        if mask[i]:
            j = i

            while j < N and mask[j]:
                j += 1

            segmentos.append((i, j - 1))
            i = j
        else:
            i += 1

    return segmentos


def fusionar_segmentos_por_hueco(segmentos, max_hueco_muestras):
    if len(segmentos) == 0:
        return []

    segmentos = sorted(segmentos)
    fusionados = [segmentos[0]]

    for ini, fin in segmentos[1:]:
        ini_prev, fin_prev = fusionados[-1]
        hueco = ini - fin_prev - 1

        if hueco <= max_hueco_muestras:
            fusionados[-1] = (ini_prev, max(fin_prev, fin))
        else:
            fusionados.append((ini, fin))

    return fusionados


def eliminar_segmentos_cortos(segmentos, min_duracion_muestras):
    return [
        (ini, fin)
        for ini, fin in segmentos
        if fin - ini + 1 >= min_duracion_muestras
    ]


def aplicar_margenes_segmentos(segmentos, N, margen_muestras):
    segmentos_margen = []

    for ini, fin in segmentos:
        ini_m = max(0, ini - margen_muestras)
        fin_m = min(N - 1, fin + margen_muestras)

        if fin_m >= ini_m:
            segmentos_margen.append((ini_m, fin_m))

    return segmentos_margen


def convertir_desplazamientos_cortos_a_oscilatorios(segmentos_normales, segmentos_osc):
    """
    Reclasifica como oscilatorios los segmentos de desplazamiento cuya duración
    final sea menor que MAX_DUR_DESPLAZAMIENTO_CORTO_MS.

    Se aplica después del postprocesado de los rojos, porque así la duració se
    mide sobre el segmento final ya fusionado, dividido, ajustado a cero y
    recortado.
    """
    if not CONVERTIR_DESPLAZAMIENTO_CORTO_A_OSCILATORIO:
        return segmentos_normales, segmentos_osc, []

    max_dur_muestras = int(MAX_DUR_DESPLAZAMIENTO_CORTO_MS * Fs / 1000)
    normales_finales = []
    normales_convertidos = []

    for ini, fin in segmentos_normales:
        dur = fin - ini + 1

        if dur < max_dur_muestras:
            normales_convertidos.append((ini, fin))
        else:
            normales_finales.append((ini, fin))

    segmentos_osc = sorted(segmentos_osc + normales_convertidos)

    segmentos_osc = fusionar_oscilatorios_cercanos_sin_pisar_normales(
        segmentos_osc,
        normales_finales
    )

    return normales_finales, segmentos_osc, normales_convertidos


def restar_segmentos(segmentos_base, segmentos_a_quitar):
    resultado = []

    for ini, fin in segmentos_base:
        piezas = [(ini, fin)]

        for q_ini, q_fin in segmentos_a_quitar:
            nuevas_piezas = []

            for p_ini, p_fin in piezas:
                if q_fin < p_ini or q_ini > p_fin:
                    nuevas_piezas.append((p_ini, p_fin))
                else:
                    if q_ini > p_ini:
                        nuevas_piezas.append((p_ini, q_ini - 1))

                    if q_fin < p_fin:
                        nuevas_piezas.append((q_fin + 1, p_fin))

            piezas = nuevas_piezas

        resultado.extend(piezas)

    return resultado


def hay_cruce_real_por_cero(a, b):
    """
    Devuelve True si entre dos mostres consecutivas hay cruce real por 0
    o una de ellas toca 0.
    """
    if a == 0 or b == 0:
        return True
    return (a < 0 and b > 0) or (a > 0 and b < 0)


def encontrar_inicio_en_cero(x, idx_inicio, deadband, idx_min=0, max_extension=None):
    """
    Ajusta el inicio hacia atrás hasta el cruce por 0 más cercano.
    Si no encuentra cruce real, usa la última muestra dentro de la banda
    muerta alrededor de 0.
    """
    if len(x) < 2:
        return idx_inicio

    if max_extension is None:
        lim_inf = idx_min
    else:
        lim_inf = max(idx_min, idx_inicio - max_extension)

    for k in range(idx_inicio, max(lim_inf, 1) - 1, -1):
        if hay_cruce_real_por_cero(x[k - 1], x[k]):
            return k

    for k in range(idx_inicio, lim_inf - 1, -1):
        if abs(x[k]) <= deadband:
            return k

    return idx_inicio


def encontrar_fin_en_cero(x, idx_fin, deadband, idx_max=None, max_extension=None):
    """
    Ajusta el final hacia delante hasta el cruce por 0 más cercano.
    Si no encuentra cruce real, usa la primera muestra dentro de la banda
    muerta alrededor de 0.
    """
    if len(x) < 2:
        return idx_fin

    if idx_max is None:
        idx_max = len(x) - 1

    if max_extension is None:
        lim_sup = idx_max
    else:
        lim_sup = min(idx_max, idx_fin + max_extension)

    for k in range(max(1, idx_fin + 1), min(lim_sup + 1, len(x))):
        if hay_cruce_real_por_cero(x[k - 1], x[k]):
            return k - 1

    for k in range(idx_fin, lim_sup + 1):
        if abs(x[k]) <= deadband:
            return k

    return idx_fin


def ajustar_limites_segmentos_a_cruce_cero(x, segmentos, eje, forzar=False):
    """
    Mantiene el criterio de detección, pero una vez detectado el segmento,
    mueve su inicio y su fin al paso por 0 más cercano.
    """
    if (not AJUSTAR_LIMITES_A_CRUCE_CERO and not forzar) or len(segmentos) == 0:
        return segmentos

    deadband = SIGN_DEADBAND[eje]
    max_extension = int(MAX_EXTENSION_CRUCE_CERO_MS * Fs / 1000)

    segmentos = sorted(segmentos)
    ajustados = []

    for i, (ini, fin) in enumerate(segmentos):
        prev_fin = ajustados[-1][1] if len(ajustados) > 0 else -1
        next_ini = segmentos[i + 1][0] if i < len(segmentos) - 1 else len(x)

        ini_aj = encontrar_inicio_en_cero(
            x=x,
            idx_inicio=ini,
            deadband=deadband,
            idx_min=prev_fin + 1,
            max_extension=max_extension
        )

        fin_aj = encontrar_fin_en_cero(
            x=x,
            idx_fin=fin,
            deadband=deadband,
            idx_max=next_ini - 1,
            max_extension=max_extension
        )

        ini_aj = max(prev_fin + 1, ini_aj)
        fin_aj = min(next_ini - 1, fin_aj)

        if fin_aj < ini_aj:
            ini_aj, fin_aj = ini, fin

        ajustados.append((ini_aj, fin_aj))

    return ajustados


def dividir_normales_por_cruce_cero(x, segmentos, eje):
    """
    Divide segmentos rojos solo si el cruce por cero separa dos lóbulos grandes.
    No divide por rizado pequeño cerca de cero.
    """
    segmentos_refinados = []

    th = umbral_desplazamiento_eje(eje)
    deadband = SIGN_DEADBAND[eje]

    ventana_busqueda = int(90 * Fs / 1000)
    min_len = int(MIN_DUR_NORMAL_MS * Fs / 1000)

    for ini, fin in segmentos:
        s = signo_estable(x[ini:fin + 1], deadband)
        cortes = []

        for k in range(1, len(s)):
            if s[k] != s[k - 1] and s[k] != 0 and s[k - 1] != 0:
                idx = ini + k

                izquierda = x[max(ini, idx - ventana_busqueda):idx]
                derecha = x[idx:min(fin + 1, idx + ventana_busqueda)]

                if len(izquierda) == 0 or len(derecha) == 0:
                    continue

                amp_izq = np.max(np.abs(izquierda))
                amp_der = np.max(np.abs(derecha))

                if amp_izq >= th and amp_der >= th:
                    cortes.append(idx)

        if len(cortes) == 0:
            segmentos_refinados.append((ini, fin))
            continue

        inicio_actual = ini

        for corte in cortes:
            fin_actual = corte - 1

            if fin_actual - inicio_actual + 1 >= min_len:
                segmentos_refinados.append((inicio_actual, fin_actual))

            inicio_actual = corte

        if fin - inicio_actual + 1 >= min_len:
            segmentos_refinados.append((inicio_actual, fin))

    return segmentos_refinados


def dividir_normales_por_valle_cerca_cero(x, segmentos, eje):
    """
    Divide segmentos de desplazamiento cuando dentro del segmento hay una
    vuelta clara a cero entre dos lóbulos fuertes.

    A diferencia de dividir_normales_por_cruce_cero(), esta función no exige
    un cruce real de signo. Basta con que exista un pequeño tramo cercano a
    cero, siempre que a izquierda y derecha haya amplitud suficiente para
    considerarlos dos lóbulos de desplazamiento independientes.
    """
    if not DIVIDIR_NORMAL_POR_VALLE_CERCA_CERO:
        return segmentos

    if len(segmentos) == 0:
        return segmentos

    th_valle = FACTOR_VALLE_CERO_NORMAL * UMBRAL_REPOSO[eje]
    th_lobulo = FACTOR_PICO_LOBULO_VALLE_NORMAL * umbral_desplazamiento_eje(eje)

    min_valle = max(1, int(MIN_DUR_VALLE_CERO_NORMAL_MS * Fs / 1000))
    ventana_lobulo = max(1, int(VENTANA_LOBULO_VALLE_MS * Fs / 1000))
    min_len = max(1, int(MIN_DUR_NORMAL_MS * Fs / 1000))

    segmentos_refinados = []

    for ini, fin in segmentos:
        ini = int(ini)
        fin = int(fin)

        if fin - ini + 1 < 2 * min_len + min_valle:
            segmentos_refinados.append((ini, fin))
            continue

        mask_valle = np.abs(x[ini:fin + 1]) <= th_valle
        valles_locales = mascara_a_segmentos(mask_valle)

        if len(valles_locales) == 0:
            segmentos_refinados.append((ini, fin))
            continue

        inicio_actual = ini
        hubo_corte = False

        for v_ini_local, v_fin_local in valles_locales:
            v_ini = ini + int(v_ini_local)
            v_fin = ini + int(v_fin_local)
            dur_valle = v_fin - v_ini + 1

            if dur_valle < min_valle:
                continue

            if v_ini - inicio_actual < min_len:
                continue
            if fin - v_fin < min_len:
                continue

            zona_izq = x[max(inicio_actual, v_ini - ventana_lobulo):v_ini]
            zona_der = x[v_fin + 1:min(fin + 1, v_fin + 1 + ventana_lobulo)]

            if len(zona_izq) == 0 or len(zona_der) == 0:
                continue

            amp_izq = np.max(np.abs(zona_izq))
            amp_der = np.max(np.abs(zona_der))

            if amp_izq < th_lobulo or amp_der < th_lobulo:
                continue

            fin_izq = v_ini - 1
            ini_der = v_fin + 1

            if fin_izq - inicio_actual + 1 >= min_len:
                segmentos_refinados.append((inicio_actual, fin_izq))
                hubo_corte = True
                inicio_actual = ini_der

        if hubo_corte:
            if fin - inicio_actual + 1 >= min_len:
                segmentos_refinados.append((inicio_actual, fin))
        else:
            segmentos_refinados.append((ini, fin))

    return segmentos_refinados


def obtener_candidatos_corte_suave(x_suave, ini, fin, eje):
    """
    Devuelve posibles puntos de corte dentro de un desplazamiento usando la
    señal filtrada a filtro suave.

    Incluye:
    - cruces reales de signo;
    - tramos cercanos a cero, aunque no haya cambio exacto de signo.
    """
    deadband = SIGN_DEADBAND[eje]
    ini = int(max(0, ini))
    fin = int(min(len(x_suave) - 1, fin))

    if fin <= ini:
        return []

    candidatos = []

    s = signo_estable(x_suave[ini:fin + 1], deadband)

    for k in range(1, len(s)):
        if s[k] != s[k - 1] and s[k] != 0 and s[k - 1] != 0:
            candidatos.append(ini + k)

    mask_cerca_cero = np.abs(x_suave[ini:fin + 1]) <= deadband
    zonas_cerca_cero = mascara_a_segmentos(mask_cerca_cero)

    for z_ini_local, z_fin_local in zonas_cerca_cero:
        z_ini = ini + int(z_ini_local)
        z_fin = ini + int(z_fin_local)
        zona = np.abs(x_suave[z_ini:z_fin + 1])

        if len(zona) == 0:
            continue

        idx_min = z_ini + int(np.argmin(zona))
        candidatos.append(idx_min)

    if len(candidatos) == 0:
        return []

    return sorted(set(int(c) for c in candidatos if ini < c < fin))


def dividir_normales_hibrido_valle_20hz_corte_suave(
        x_suave,
        x_validacion,
        segmentos,
        eje
):
    """
    Divide desplazamientos confirmados cuando hay dos lóbulos separados por un
    valle interno.

    Criterio híbrido:
    - 20 Hz: decide si el valle es suficientemente claro para permitir cortar;
    - filtro suave: fija el instante exacto de corte, porque conserva mejor los pasos
      por cero que el filtrado a 20 Hz puede suavizar.

    Esto evita dos errores opuestos:
    - no cortar por cualquier rizado de filtro suave;
    - no unir dos lóbulos reales porque 20 Hz haya suavizado el cruce por cero.
    """
    if not DIVIDIR_DESPLAZAMIENTO_HIBRIDO:
        return segmentos

    if x_validacion is None or len(segmentos) == 0:
        return segmentos

    N = min(len(x_suave), len(x_validacion))
    if N == 0:
        return segmentos

    th_desp = umbral_desplazamiento_eje(eje)
    th_valle_abs = max(
        UMBRAL_REPOSO[eje],
        FACTOR_VALLE_ABS_20HZ * th_desp
    )
    th_pico = FACTOR_PICO_LOBULO_DIVISION_HIBRIDA * th_desp

    ventana_lobulo = max(1, int(VENTANA_LOBULO_DIVISION_HIBRIDA_MS * Fs / 1000))
    radio_corte = max(1, int(VENTANA_BUSQUEDA_CORTE_SUAVE_MS * Fs / 1000))
    min_len = max(1, int(MIN_DUR_NORMAL_MS * Fs / 1000))
    min_sep = max(1, int(MIN_SEPARACION_CORTES_HIBRIDA_MS * Fs / 1000))

    segmentos_refinados = []

    for ini, fin in sorted((int(a), int(b)) for a, b in segmentos):
        ini = max(0, ini)
        fin = min(N - 1, fin)

        if fin - ini + 1 < 2 * min_len:
            segmentos_refinados.append((ini, fin))
            continue

        candidatos = obtener_candidatos_corte_suave(
            x_suave=x_suave,
            ini=ini,
            fin=fin,
            eje=eje
        )

        if len(candidatos) == 0:
            segmentos_refinados.append((ini, fin))
            continue

        inicio_actual = ini
        ultimo_corte_aceptado = None
        hubo_corte = False

        for idx_candidato in candidatos:
            idx_candidato = int(idx_candidato)

            if idx_candidato - inicio_actual < min_len:
                continue

            if fin - idx_candidato + 1 < min_len:
                continue

            if ultimo_corte_aceptado is not None:
                if idx_candidato - ultimo_corte_aceptado < min_sep:
                    continue

            v_ini = max(inicio_actual, idx_candidato - radio_corte)
            v_fin = min(fin, idx_candidato + radio_corte)
            abs_valle_20 = np.abs(x_validacion[v_ini:v_fin + 1])

            if len(abs_valle_20) == 0:
                continue

            pos_min_local = int(np.argmin(abs_valle_20))
            idx_valle_20 = v_ini + pos_min_local
            amp_valle_20 = float(abs_valle_20[pos_min_local])

            min_en_borde = pos_min_local == 0 or pos_min_local == len(abs_valle_20) - 1

            izq_ini = max(inicio_actual, idx_candidato - ventana_lobulo)
            izq_fin = idx_candidato - 1
            der_ini = idx_candidato
            der_fin = min(fin, idx_candidato + ventana_lobulo)

            if izq_fin < izq_ini or der_fin < der_ini:
                continue

            zona_izq_20 = np.abs(x_validacion[izq_ini:izq_fin + 1])
            zona_der_20 = np.abs(x_validacion[der_ini:der_fin + 1])

            if len(zona_izq_20) == 0 or len(zona_der_20) == 0:
                continue

            pico_izq_20 = float(np.max(zona_izq_20))
            pico_der_20 = float(np.max(zona_der_20))

            if pico_izq_20 < th_pico or pico_der_20 < th_pico:
                continue

            referencia_lobulos = max(1e-12, min(pico_izq_20, pico_der_20))

            valle_por_amplitud = amp_valle_20 <= th_valle_abs
            valle_por_caida_relativa = (
                amp_valle_20
                <= FACTOR_CAIDA_RELATIVA_VALLE_20HZ * referencia_lobulos
            )

            if not (valle_por_amplitud or valle_por_caida_relativa):
                continue

            if min_en_borde and not valle_por_amplitud:
                continue

            candidatos_locales = [
                c for c in candidatos
                if abs(c - idx_valle_20) <= radio_corte
                and inicio_actual + min_len <= c <= fin - min_len + 1
            ]

            if len(candidatos_locales) > 0:
                idx_corte = min(
                    candidatos_locales,
                    key=lambda c: (abs(c - idx_valle_20), abs(x_suave[c]))
                )
            else:
                idx_corte = idx_candidato

            if idx_corte - inicio_actual < min_len:
                continue
            if fin - idx_corte + 1 < min_len:
                continue

            segmentos_refinados.append((inicio_actual, idx_corte - 1))
            inicio_actual = idx_corte
            ultimo_corte_aceptado = idx_corte
            hubo_corte = True

        if hubo_corte:
            if fin - inicio_actual + 1 >= min_len:
                segmentos_refinados.append((inicio_actual, fin))
        else:
            segmentos_refinados.append((ini, fin))

    return segmentos_refinados

def segmentos_solapan(seg_a, seg_b):
    ini_a, fin_a = seg_a
    ini_b, fin_b = seg_b
    return not (fin_a < ini_b or fin_b < ini_a)


def solapa_con_alguno(segmento, lista_segmentos):
    return any(segmentos_solapan(segmento, otro) for otro in lista_segmentos)


def contar_cruces_cero_segmento(x, ini, fin, eje):
    """
    Cuenta cruces por cero estables dentro de un segmento.
    Sirve para distinguir un lóbulo de movimiento de una oscilación sostenida.
    """
    if fin <= ini:
        return 0

    s = signo_estable(x[ini:fin + 1], SIGN_DEADBAND[eje])

    cruces = (
        (s[1:] != s[:-1])
        & (s[1:] != 0)
        & (s[:-1] != 0)
    )

    return int(np.sum(cruces))


def filtrar_lobulos_normales_dentro_de_oscilacion(x, segmentos_normales, segmentos_osc, eje):
    """
    Mantiene como movimiento normal los lóbulos claros aunque solapen con una
    zona marcada como oscilatoria.

    El problema que corrige:
    - el detector amarillo usa cruces por cero en ventana;
    - un transitorio de movimiento con ringing puede cumplir esa condición;
    - por eso antes se pintaba como amarillo algo que visualmente es un lóbulo.

    Regla:
    - si el segmento normal no solapa con amarillo, se mantiene;
    - si solapa con amarillo, solo se mantiene si parece un lóbulo:
      duració mínima, pico suficiente y pocos cruces internos.
    """
    if not PRIORIDAD_LOBULO_NORMAL_SOBRE_OSCILACION:
        return segmentos_normales

    min_dur = int(MIN_DUR_LOBULO_NORMAL_EN_OSC_MS * Fs / 1000)
    th_pico = FACTOR_PICO_LOBULO_NORMAL_EN_OSC * umbral_desplazamiento_eje(eje)

    resultado = []

    for ini, fin in segmentos_normales:
        segmento = (ini, fin)

        if not solapa_con_alguno(segmento, segmentos_osc):
            resultado.append(segmento)
            continue

        dur = fin - ini + 1
        pico = float(np.max(np.abs(x[ini:fin + 1])))
        n_cruces = contar_cruces_cero_segmento(x, ini, fin, eje)

        es_lobulo_base = (
            dur >= min_dur
            and pico >= th_pico
            and n_cruces <= MAX_CRUCES_LOBULO_NORMAL_EN_OSC
        )

        es_lobulo_claro = (
            PERMITIR_LOBULO_CLARO_AUNQUE_TENGA_CRUCES
            and dur >= min_dur
            and pico >= FACTOR_PICO_LOBULO_CLARO_EN_OSC * umbral_desplazamiento_eje(eje)
        )

        if es_lobulo_base or es_lobulo_claro:
            resultado.append(segmento)

    return resultado


def encontrar_primer_indice_actividad_sostenida(x, ini, fin, eje):
    """
    Busca el primer punto dentro de un segmento donde |x| supera umbral_desplazamiento_eje(eje)
    de forma sostenida durante una pequeña ventana temporal.

    Esto evita que un único pico o ruido aislado marque el inicio real del lóbulo.
    """
    th = umbral_desplazamiento_eje(eje)
    ventana = max(1, int(VENTANA_ACTIVIDAD_SOSTENIDA_MS * Fs / 1000))
    n_min = max(1, int(np.ceil(FRACCION_ACTIVIDAD_SOSTENIDA * ventana)))

    ini = int(max(0, ini))
    fin = int(min(len(x) - 1, fin))

    if fin <= ini:
        return None

    mask_fuerte = np.abs(x[ini:fin + 1]) >= th

    if len(mask_fuerte) == 0:
        return None

    if len(mask_fuerte) < ventana:
        indices = np.where(mask_fuerte)[0]
        if len(indices) == 0:
            return None
        return ini + int(indices[0])

    acumulada = np.concatenate([[0], np.cumsum(mask_fuerte.astype(int))])

    for k in range(0, len(mask_fuerte) - ventana + 1):
        n_activos = acumulada[k + ventana] - acumulada[k]

        if n_activos >= n_min:
            indices_locales = np.where(mask_fuerte[k:k + ventana])[0]
            return ini + k + int(indices_locales[0])

    indices = np.where(mask_fuerte)[0]

    if len(indices) == 0:
        return None

    return ini + int(indices[0])


def recortar_inicio_segmentos_normales_por_actividad_sostenida(x, segmentos, eje):
    """
    Recorta el inicio de segmentos normales si el ajuste a cruce por cero los ha
    extendido demasiado hacia atrás.

    Solo actúa cuando hay un prefijo suficientemente largo sin movimiento fuerte.
    Después recoloca el nuevo inicio cerca del cruce por cero inmediatamente
    anterior al primer movimiento sostenido.
    """
    if not RECORTAR_INICIO_NORMAL_POR_ACTIVIDAD_SOSTENIDA:
        return segmentos

    if len(segmentos) == 0:
        return segmentos

    min_prefijo = int(MIN_PREFIJO_REPOSO_RECORTE_MS * Fs / 1000)
    max_busqueda_cero = int(MAX_BUSQUEDA_CERO_RECORTE_MS * Fs / 1000)
    min_dur_normal = int(MIN_DUR_NORMAL_MS * Fs / 1000)
    deadband = SIGN_DEADBAND[eje]

    recortados = []

    for ini, fin in segmentos:
        idx_sostenido = encontrar_primer_indice_actividad_sostenida(
            x=x,
            ini=ini,
            fin=fin,
            eje=eje
        )

        if idx_sostenido is None:
            recortados.append((ini, fin))
            continue

        prefijo = idx_sostenido - ini

        if prefijo < min_prefijo:
            recortados.append((ini, fin))
            continue

        ini_min = max(ini, idx_sostenido - max_busqueda_cero)

        nuevo_ini = encontrar_inicio_en_cero(
            x=x,
            idx_inicio=idx_sostenido,
            deadband=deadband,
            idx_min=ini_min,
            max_extension=max_busqueda_cero
        )

        if nuevo_ini <= ini:
            nuevo_ini = idx_sostenido

        if fin - nuevo_ini + 1 >= min_dur_normal:
            recortados.append((int(nuevo_ini), fin))
        else:
            recortados.append((ini, fin))

    return recortados


def buscar_ultimo_paso_por_cero_previo(x, idx_referencia, lim_inf, deadband):
    """
    Busca el último instante de paso por cero antes de idx_referencia.

    Prioridad:
    1) cruce real de signo entre dos mostres consecutivas;
    2) si no hay cruce real, último punto dentro de la banda de cero.

    Devuelve un índice de muestra. Si no encuentra nada, devuelve None.
    """
    if len(x) < 2:
        return None

    idx_referencia = int(np.clip(idx_referencia, 0, len(x) - 1))
    lim_inf = int(max(0, min(lim_inf, idx_referencia)))

    for k in range(idx_referencia, max(lim_inf, 1) - 1, -1):
        if hay_cruce_real_por_cero(x[k - 1], x[k]):
            return int(k)

    for k in range(idx_referencia, lim_inf - 1, -1):
        if abs(x[k]) <= deadband:
            return int(k)

    return None


def recortar_inicio_normales_al_cero_previo_candidato(x, segmentos, eje, mask_candidato_normal):
    """
    Recoloca el inicio de cada segmento de desplazamiento en el último paso por
    cero anterior al primer punto candidato a desplazamiento.

    Esta función puede mover el inicio hacia atrás, incluso si esa parte había
    sido marcada inicialmente como oscilatoria. Después, el postprocesado de
    solapes recorta el oscilatorio para que no pise al desplazamiento.
    """
    if not RECORTAR_INICIO_NORMAL_AL_CERO_PREVIO_CANDIDATO:
        return segmentos

    if len(segmentos) == 0:
        return segmentos

    deadband = SIGN_DEADBAND[eje]
    min_dur_normal = int(MIN_DUR_NORMAL_MS * Fs / 1000)
    max_busqueda = int(MAX_BUSQUEDA_CERO_PREVIO_CANDIDATO_MS * Fs / 1000)

    segmentos = sorted((int(ini), int(fin)) for ini, fin in segmentos)
    corregidos = []

    for ini, fin in segmentos:
        if fin <= ini:
            corregidos.append((ini, fin))
            continue

        zona = mask_candidato_normal[ini:fin + 1]
        indices = np.where(zona)[0]

        if len(indices) == 0:
            corregidos.append((ini, fin))
            continue

        idx_candidato = ini + int(indices[0])

        prev_fin = corregidos[-1][1] if len(corregidos) > 0 else -1
        lim_inf = max(prev_fin + 1, idx_candidato - max_busqueda, 0)

        nuevo_ini = buscar_ultimo_paso_por_cero_previo(
            x=x,
            idx_referencia=idx_candidato,
            lim_inf=lim_inf,
            deadband=deadband
        )

        if nuevo_ini is None:
            nuevo_ini = idx_candidato

        nuevo_ini = int(max(prev_fin + 1, nuevo_ini))

        if fin - nuevo_ini + 1 >= min_dur_normal:
            corregidos.append((nuevo_ini, fin))
        else:
            corregidos.append((ini, fin))

    return corregidos

def segmento_solapa_con_alguno(seg, segmentos):
    ini, fin = seg

    for ini_b, fin_b in segmentos:
        if ini <= fin_b and ini_b <= fin:
            return True

    return False


def fusionar_oscilatorios_cercanos_sin_pisar_normales(segmentos_osc, segmentos_normales):
    """
    Fusiona segmentos oscilatorios cercanos, pero solo si el hueco entre ambos
    no contiene un segmento normal. Así se evita que un grupo oscilatorio absorba
    un desplazamiento normal situado en medio.
    """
    if not FUSIONAR_OSCILATORIOS_CERCANOS_FINAL:
        return segmentos_osc

    if len(segmentos_osc) <= 1:
        return segmentos_osc

    max_hueco = int(MAX_GAP_OSC_FINAL_MS * Fs / 1000)
    segmentos_osc = sorted(segmentos_osc)
    segmentos_normales = sorted(segmentos_normales)

    fusionados = []
    ini_actual, fin_actual = segmentos_osc[0]

    for ini, fin in segmentos_osc[1:]:
        hueco = ini - fin_actual - 1
        hueco_seg = (fin_actual + 1, ini - 1)

        hay_normal_en_hueco = (
            hueco_seg[0] <= hueco_seg[1]
            and segmento_solapa_con_alguno(hueco_seg, segmentos_normales)
        )

        if hueco <= max_hueco and not hay_normal_en_hueco:
            fin_actual = max(fin_actual, fin)
        else:
            fusionados.append((ini_actual, fin_actual))
            ini_actual, fin_actual = ini, fin

    fusionados.append((ini_actual, fin_actual))

    return fusionados

def validar_segmentos_desplazamiento_con_filtro_agresivo(
        x_validacion,
        segmentos_candidatos,
        eje
):
    """
    Valida los candidatos a desplazamiento con la señal filtrada a baja
    frecuencia.

    Criterio actual:
    ya no se confirma el candidato completo solo porque en alguna muestra
    supere el umbral. Se recorta el candidato a los tramos donde la señal de
    20 Hz supera realmente el umbral de desplazamiento. Después, el
    postprocesado ajusta esos tramos al primer cruce por cero de la propia
    señal de 20 Hz.

    Resultado:
    - la parte confirmada entra como desplazamiento;
    - la parte del candidato que no se confirma en 20 Hz pasa a oscilatorio.
    """
    if x_validacion is None:
        return list(segmentos_candidatos), []

    N = len(x_validacion)
    margen = int(MARGEN_VALIDACION_DESPLAZAMIENTO_MS * Fs / 1000)
    th = umbral_desplazamiento_eje(eje)

    confirmados = []
    rechazados = []

    for ini, fin in segmentos_candidatos:
        ini = int(max(0, ini))
        fin = int(min(N - 1, fin))

        if ini > fin:
            continue

        ini_v = max(0, ini - margen)
        fin_v = min(N - 1, fin + margen)

        if ini_v > fin_v:
            rechazados.append((ini, fin))
            continue

        mask_validada_local = np.abs(x_validacion[ini_v:fin_v + 1]) >= th
        segmentos_validos_locales = mascara_a_segmentos(mask_validada_local)

        if len(segmentos_validos_locales) == 0:
            rechazados.append((ini, fin))
            continue

        confirmados_candidato = []

        for v_ini_local, v_fin_local in segmentos_validos_locales:
            v_ini = ini_v + int(v_ini_local)
            v_fin = ini_v + int(v_fin_local)

            v_ini = max(ini, v_ini)
            v_fin = min(fin, v_fin)

            if v_ini <= v_fin:
                confirmados_candidato.append((v_ini, v_fin))

        if len(confirmados_candidato) == 0:
            rechazados.append((ini, fin))
            continue

        confirmados.extend(confirmados_candidato)

        rechazados.extend(
            restar_segmentos(
                [(ini, fin)],
                confirmados_candidato
            )
        )

    confirmados = fusionar_segmentos_por_hueco(
        sorted(confirmados),
        int(MAX_GAP_NORMAL_MS * Fs / 1000)
    )

    rechazados = fusionar_segmentos_por_hueco(
        sorted(rechazados),
        int(MAX_GAP_OSC_MS * Fs / 1000)
    )

    return confirmados, rechazados


def segmentos_a_mascara(segmentos, N):
    mask = np.zeros(N, dtype=bool)

    for ini, fin in segmentos:
        ini = max(0, int(ini))
        fin = min(N - 1, int(fin))

        if ini <= fin:
            mask[ini:fin + 1] = True

    return mask


def segmentar_eje_xy(x_suave, eje, x_validacion=None):
    """
    Segmentación por eje.

    1) La señal suau se usa para detectar actividad y candidatos a desplazamiento.
    2) La señal agresiva se usa para validar todos los candidatos fuertes.
    3) La señal agresiva de 20 Hz valida los desplazamientos y marca la
       tendencia general.
    4) Las divisiones internas se deciden de forma híbrida: valle en 20 Hz y
       corte fino en filtro suave.
    """
    N = len(x_suave)

    if x_validacion is None:
        x_validacion = x_suave

    x_limites_desplazamiento = x_validacion

    mask_no_reposo = np.abs(x_suave) >= UMBRAL_REPOSO[eje]
    mask_movimiento_fuerte_suave = np.abs(x_suave) >= umbral_desplazamiento_eje(eje)
    mask_actividad_debil = mask_no_reposo & ~mask_movimiento_fuerte_suave

    cruces_ventana, rms_ventana = calcular_cruces_y_rms_ventana(x_suave, eje)
    mask_osc = mask_actividad_debil.copy()

    mask_normal_candidato = mask_movimiento_fuerte_suave.copy()
    mask_normal_candidato_validacion = (
        np.abs(x_validacion) >= umbral_desplazamiento_eje(eje)
    )

    segmentos_osc = mascara_a_segmentos(mask_osc)
    segmentos_normales_candidatos = mascara_a_segmentos(mask_normal_candidato)

    segmentos_osc = fusionar_segmentos_por_hueco(
        segmentos_osc,
        int(MAX_GAP_OSC_MS * Fs / 1000)
    )

    segmentos_normales_candidatos = fusionar_segmentos_por_hueco(
        segmentos_normales_candidatos,
        int(MAX_GAP_NORMAL_MS * Fs / 1000)
    )

    segmentos_osc = eliminar_segmentos_cortos(
        segmentos_osc,
        int(MIN_DUR_OSC_MS * Fs / 1000)
    )

    segmentos_normales_candidatos = eliminar_segmentos_cortos(
        segmentos_normales_candidatos,
        int(MIN_DUR_NORMAL_MS * Fs / 1000)
    )

    segmentos_normales, segmentos_rechazados_validacion = (
        validar_segmentos_desplazamiento_con_filtro_agresivo(
            x_validacion=x_validacion,
            segmentos_candidatos=segmentos_normales_candidatos,
            eje=eje
        )
    )

    if len(segmentos_rechazados_validacion) > 0:
        segmentos_osc = sorted(segmentos_osc + segmentos_rechazados_validacion)

    segmentos_osc = fusionar_segmentos_por_hueco(
        segmentos_osc,
        int(MAX_GAP_OSC_MS * Fs / 1000)
    )

    segmentos_osc = eliminar_segmentos_cortos(
        segmentos_osc,
        int(MIN_DUR_OSC_MS * Fs / 1000)
    )

    segmentos_osc = aplicar_margenes_segmentos(
        segmentos_osc,
        N,
        int(EXPAND_MS * Fs / 1000)
    )

    if AJUSTAR_OSCILATORIOS_A_CRUCE_CERO:
        segmentos_osc = ajustar_limites_segmentos_a_cruce_cero(
            x_suave,
            segmentos_osc,
            eje,
            forzar=True
        )


    segmentos_normales = dividir_normales_hibrido_valle_20hz_corte_suave(
        x_suave=x_suave,
        x_validacion=x_limites_desplazamiento,
        segmentos=segmentos_normales,
        eje=eje
    )

    segmentos_normales = dividir_normales_por_cruce_cero(
        x_limites_desplazamiento,
        segmentos_normales,
        eje
    )

    segmentos_normales = dividir_normales_por_valle_cerca_cero(
        x_limites_desplazamiento,
        segmentos_normales,
        eje
    )

    segmentos_normales = eliminar_segmentos_cortos(
        segmentos_normales,
        int(MIN_DUR_NORMAL_MS * Fs / 1000)
    )

    segmentos_normales = ajustar_limites_segmentos_a_cruce_cero(
        x_limites_desplazamiento,
        segmentos_normales,
        eje
    )

    segmentos_normales = recortar_inicio_segmentos_normales_por_actividad_sostenida(
        x_limites_desplazamiento,
        segmentos_normales,
        eje
    )

    segmentos_normales = recortar_inicio_normales_al_cero_previo_candidato(
        x_limites_desplazamiento,
        segmentos_normales,
        eje,
        mask_normal_candidato_validacion
    )

    segmentos_normales = filtrar_lobulos_normales_dentro_de_oscilacion(
        x_limites_desplazamiento,
        segmentos_normales,
        segmentos_osc,
        eje
    )

    if PRIORIDAD_LOBULO_NORMAL_SOBRE_OSCILACION and RECORTAR_OSCILATORIO_CON_LOBULOS_NORMALES:
        segmentos_osc = restar_segmentos(
            segmentos_osc,
            segmentos_normales
        )

        segmentos_osc = eliminar_segmentos_cortos(
            segmentos_osc,
            int(MIN_DUR_OSC_MS * Fs / 1000)
        )

    segmentos_osc = fusionar_oscilatorios_cercanos_sin_pisar_normales(
        segmentos_osc,
        segmentos_normales
    )

    segmentos_osc = eliminar_segmentos_cortos(
        segmentos_osc,
        int(MIN_DUR_OSC_MS * Fs / 1000)
    )

    segmentos_normales, segmentos_osc, segmentos_normales_convertidos = (
        convertir_desplazamientos_cortos_a_oscilatorios(
            segmentos_normales,
            segmentos_osc
        )
    )

    segmentos_tipo = []

    for ini, fin in segmentos_normales:
        segmentos_tipo.append({
            "ini": ini,
            "fin": fin,
            "tipo": "normal",
        })

    for ini, fin in segmentos_osc:
        segmentos_tipo.append({
            "ini": ini,
            "fin": fin,
            "tipo": "oscilatorio",
        })

    segmentos_tipo = sorted(
        segmentos_tipo,
        key=lambda d: (d["ini"], d["fin"], d["tipo"])
    )

    segmentos_normales = [
        (d["ini"], d["fin"])
        for d in segmentos_tipo
        if d["tipo"] == "normal"
    ]

    segmentos_osc = [
        (d["ini"], d["fin"])
        for d in segmentos_tipo
        if d["tipo"] == "oscilatorio"
    ]

    mask_validacion_desplazamiento = np.abs(x_validacion) >= umbral_desplazamiento_eje(eje)
    mask_normal_confirmada = segmentos_a_mascara(segmentos_normales, N)
    mask_normal_rechazada_validacion = segmentos_a_mascara(segmentos_rechazados_validacion, N)

    info = {
        "mask_osc": mask_osc,
        "cruces_ventana": cruces_ventana,
        "rms_ventana": rms_ventana,
        "mask_movimiento_fuerte": mask_movimiento_fuerte_suave,
        "mask_no_reposo": mask_no_reposo,
        "mask_actividad_debil": mask_actividad_debil,
        "mask_normal": mask_normal_candidato,
        "mask_normal_candidato_validacion": mask_normal_candidato_validacion,
        "mask_validacion_desplazamiento": mask_validacion_desplazamiento,
        "mask_normal_confirmada": mask_normal_confirmada,
        "mask_normal_rechazada_validacion": mask_normal_rechazada_validacion,
        "segmentos_rechazados_validacion": segmentos_rechazados_validacion,
        "segmentos_tipo": segmentos_tipo,
        "segmentos_normales_convertidos_a_osc": segmentos_normales_convertidos,
    }

    return segmentos_normales, segmentos_osc, info


def calcular_segmentos_reposo(N, segmentos_activos):
    reposo = []

    if len(segmentos_activos) == 0:
        return [(0, N - 1)]

    segmentos_activos = sorted(segmentos_activos)
    actual = 0

    for ini, fin in segmentos_activos:
        if ini > actual:
            reposo.append((actual, ini - 1))

        actual = max(actual, fin + 1)

    if actual < N:
        reposo.append((actual, N - 1))

    min_reposo = int(MIN_DURACION_REPOSO_VISUAL_MS * Fs / 1000)

    reposo = [
        (ini, fin)
        for ini, fin in reposo
        if fin - ini + 1 >= min_reposo
    ]

    return reposo


def segmentar_xy_sin_z(
        acc_segmentacion_suave,
        acc_validacion_desplazamiento=None,
        acc_impactos=None
):
    """
    Segmenta solo X e Y.
    El eje Z se ignora completamente para decidir inicio/fin/tipo.

    Esquema actual:
    - acc_segmentacion_suave se usa para la segmentación inicial;
    - acc_validacion_desplazamiento se usa para confirmar/rechazar candidatos
      y para ajustar los límites finales del desplazamiento;
    - acc_impactos se usa únicamente para pintar impactos en naranja.
    """
    if acc_validacion_desplazamiento is None:
        acc_validacion_desplazamiento = acc_segmentacion_suave

    if acc_impactos is None:
        acc_impactos = acc_segmentacion_suave

    N = len(acc_segmentacion_suave)

    info = {
        "segmentos_por_eje": {},
        "debug_por_eje": {},
    }

    segmentos_impacto_global = detectar_impactos_modulo(acc_impactos)

    for eje in EJES_SEGMENTACION:
        idx = IDX_EJE[eje]

        segmentos_normales, segmentos_osc, debug = segmentar_eje_xy(
            x_suave=acc_segmentacion_suave[:, idx],
            eje=eje,
            x_validacion=acc_validacion_desplazamiento[:, idx]
        )

        segmentos_eje = sorted(segmentos_normales + segmentos_osc)
        segmentos_reposo = calcular_segmentos_reposo(N, segmentos_eje)

        segmentos_impacto = segmentos_impacto_global

        items_etiquetas = []

        for seg in segmentos_normales:
            items_etiquetas.append((seg[0], seg[1], "normal", seg))

        for seg in segmentos_osc:
            items_etiquetas.append((seg[0], seg[1], "oscilatorio", seg))

        for seg in segmentos_reposo:
            items_etiquetas.append((seg[0], seg[1], "reposo", seg))

        items_etiquetas.sort(key=lambda item: (item[0], item[1]))

        etiqueta_por_segmento = {}

        for i, (_, _, _, seg) in enumerate(items_etiquetas, start=1):
            etiqueta_por_segmento[seg] = f"{eje.upper()}{i}"

        etiquetas_todos = [
            etiqueta_por_segmento[seg]
            for seg in segmentos_eje
        ]

        etiquetas_reposo = [
            etiqueta_por_segmento[seg]
            for seg in segmentos_reposo
        ]

        info["segmentos_por_eje"][eje] = {
            "normal": segmentos_normales,
            "oscilatorio": segmentos_osc,
            "todos": segmentos_eje,
            "reposo": segmentos_reposo,
            "impacto": segmentos_impacto,
            "etiquetas_todos": etiquetas_todos,
            "etiquetas_reposo": etiquetas_reposo,
        }

        info["debug_por_eje"][eje] = debug

    return info


def obtener_impactos_globales(info_seg):
    """
    Devuelve los impactos globales sin duplicarlos.

    La misma lista de impactos está guardada en X e Y, porque se obtiene
    mediante el módulo de los tres ejes. Por eso se lee únicamente desde X.
    """
    if info_seg is None:
        return []

    impactos = (
        info_seg
        .get("segmentos_por_eje", {})
        .get("x", {})
        .get("impacto", [])
    )

    return [
        (int(ini), int(fin))
        for ini, fin in impactos
    ]


def imprimir_impactos_detectados(info_seg):
    """
    Muestra por consola los impactos detectados y los segmentos X/Y
    con los que se solapa cada impacto.
    """
    impactos = obtener_impactos_globales(info_seg)

    if len(impactos) == 0:
        return

    print("\n" + "!" * 80)
    print(
        f"[AVÍS D'IMPACTE] S'han detectat {len(impactos)} "
        f"impacte(s), amb un llindar de {UMBRAL_IMPACTO_G:.1f} g."
    )

    for numero, impacto in enumerate(impactos, start=1):
        ini_imp, fin_imp = impacto
        etiquetas_afectadas = []

        for eje in ("x", "y"):
            info_eje = (
                info_seg
                .get("segmentos_por_eje", {})
                .get(eje, {})
            )

            segmentos_etiquetados = list(zip(
                info_eje.get("todos", []),
                info_eje.get("etiquetas_todos", [])
            ))

            segmentos_etiquetados.extend(zip(
                info_eje.get("reposo", []),
                info_eje.get("etiquetas_reposo", [])
            ))

            for segmento, etiqueta in segmentos_etiquetados:
                if segmentos_solapan(segmento, impacto):
                    etiquetas_afectadas.append(str(etiqueta))

        etiquetas_afectadas = sorted(set(etiquetas_afectadas))

        if etiquetas_afectadas:
            texto_segmentos = ", ".join(etiquetas_afectadas)
        else:
            texto_segmentos = "cap segment X/Y associat"

        print(
            f"  Impacte {numero}: "
            f"{ini_imp / Fs:.3f}–{fin_imp / Fs:.3f} s | "
            f"segments afectats: {texto_segmentos}"
        )

    print("!" * 80)


##################################################
# CLASIFICACIÓN DE SEGMENTOS POR DTW INDEPENDIENTE X/Y
##################################################

def remuestrear_1d(x, n_salida):
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return np.zeros(n_salida)

    eje_original = np.linspace(0.0, 1.0, len(x))
    eje_nuevo = np.linspace(0.0, 1.0, n_salida)

    return np.interp(eje_nuevo, eje_original, x)


def z_normalizar_1d(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    std = np.std(x)

    if std < 1e-12:
        return np.zeros_like(x)

    return x / std


def obtener_etiquetas_segmentos_tipo_eje(info_seg, eje, tipo):
    """
    Devuelve segmentos y etiquetas manteniendo exactamente las etiquetas
    creadas en la segmentación.
    """
    segs_eje = info_seg["segmentos_por_eje"][eje]

    if tipo == "reposo":
        return list(segs_eje["reposo"]), list(segs_eje["etiquetas_reposo"])

    todos = list(segs_eje["todos"])
    etiquetas_todos = list(segs_eje["etiquetas_todos"])
    objetivo = set(segs_eje[tipo])

    segmentos = []
    etiquetas = []

    for seg, etiqueta_original in zip(todos, etiquetas_todos):
        if seg in objetivo:
            segmentos.append(seg)
            etiquetas.append(etiqueta_original)

    return segmentos, etiquetas


def distancia_dtw_1d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = len(a)
    m = len(b)

    if n == 0 or m == 0:
        return np.inf

    banda = int(DTW_BANDA_PORC * max(n, m))
    banda = max(banda, abs(n - m), 1)

    anterior = np.full(m + 1, np.inf)
    actual = np.full(m + 1, np.inf)

    anterior[0] = 0.0

    for i in range(1, n + 1):
        actual[:] = np.inf

        j_ini = max(1, i - banda)
        j_fin = min(m, i + banda)

        for j in range(j_ini, j_fin + 1):
            coste = abs(a[i - 1] - b[j - 1])

            actual[j] = coste + min(
                anterior[j],
                actual[j - 1],
                anterior[j - 1]
            )

        anterior, actual = actual, anterior

    return anterior[m] / max(n, m)


def diferencia_relativa(valor_a, valor_b):
    """
    Devuelve una diferencia relativa acotada aproximadamente entre 0 y 1
    para magnitudes positivas como RMS, pico o duración.
    """
    valor_a = float(abs(valor_a))
    valor_b = float(abs(valor_b))

    denom = max(valor_a, valor_b, 1e-12)
    return abs(valor_a - valor_b) / denom


def distancia_metricas_segmento(metricas_a, metricas_b):
    if metricas_a is None or metricas_b is None:
        return 0.0

    d_rms = diferencia_relativa(metricas_a["rms"], metricas_b["rms"])
    d_pico = diferencia_relativa(metricas_a["pico_abs"], metricas_b["pico_abs"])
    d_duracion = diferencia_relativa(metricas_a["duracion_s"], metricas_b["duracion_s"])

    d_extra = (
        PESO_RMS_DTW * d_rms
        + PESO_PICO_DTW * d_pico
        + PESO_DURACION_DTW * d_duracion
    )

    if PENALIZAR_SIGNO_PICO_DTW:
        signo_a = metricas_a["signo_pico"]
        signo_b = metricas_b["signo_pico"]

        if signo_a != 0 and signo_b != 0 and signo_a != signo_b:
            d_extra += PENALIZACION_SIGNO_PICO_DTW

    return d_extra


def distancia_formas_1d(a, b, metricas_a=None, metricas_b=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if USAR_DTW:
        d_forma = distancia_dtw_1d(a, b)
    else:
        d_forma = np.sqrt(np.mean((a - b) ** 2))

    d_metricas = distancia_metricas_segmento(metricas_a, metricas_b)

    return PESO_FORMA_DTW * d_forma + d_metricas


##################################################
# GRÁFICAS
##################################################

def limite_simetrico(datos, factor=None, minimo=0.15, usar_maximo_real=None):
    """
    Calcula límites verticales simétricos para las gráficas.

    Si usar_maximo_real=True, usa el máximo absoluto real y no recorta picos.
    Si usar_maximo_real=False, usa percentil 99.7 para evitar que un pico aislado
    comprima demasiado la visualización.
    """
    if factor is None:
        factor = FACTOR_MARGEN_ESCALA_Y

    if usar_maximo_real is None:
        usar_maximo_real = USAR_MAXIMO_REAL_ESCALA_Y

    datos = np.asarray(datos)
    datos_abs = np.abs(datos[np.isfinite(datos)])

    if datos_abs.size == 0:
        return minimo

    if usar_maximo_real:
        ymax = np.max(datos_abs) * factor
    else:
        ymax = np.percentile(datos_abs, 99.7) * factor

    if ymax < minimo:
        ymax = minimo

    return ymax


def pintar_segmentos_eje(
        ax,
        t,
        segmentos_normales,
        segmentos_oscilatorios,
        segmentos_reposo,
        segmentos_impacto=None
):
    if segmentos_impacto is None:
        segmentos_impacto = []

    primera_reposo = True
    primera_normal = True
    primera_osc = True
    primera_impacto = True

    for ini, fin in segmentos_reposo:
        ax.axvspan(
            t[ini],
            t[fin],
            color="#00FFFF",
            alpha=0.10,
            label="Repòs" if primera_reposo else None
        )
        primera_reposo = False

    for ini, fin in segmentos_oscilatorios:
        ax.axvspan(
            t[ini],
            t[fin],
            color="#DA70D6",
            alpha=0.30,
            label="Oscil·latori" if primera_osc else None
        )
        primera_osc = False

    for ini, fin in segmentos_normales:
        ax.axvspan(
            t[ini],
            t[fin],
            color="#FFDAB9",
            alpha=0.22,
            label="Moviment normal" if primera_normal else None
        )
        primera_normal = False

    for ini, fin in segmentos_impacto:
        ax.axvspan(
            t[ini],
            t[fin],
            color="orange",
            alpha=0.45,
            label="Impacte" if primera_impacto else None
        )
        primera_impacto = False


def crear_figura_aceleracion_xy_sin_z(
        t,
        acc_sin_filtrar,
        acc_filtrada_suave,
        acc_filtrada_agresiva,
        info_seg,
        unidad_acc,
        titulo
):
    fig = Figure(figsize=(13, 8), dpi=100)
    axs = fig.subplots(
        3,
        1,
        sharex=True,
        sharey=MISMA_ESCALA_Y_ACELERACION
    )

    fig.suptitle(
        f"{titulo}\nSegmentació: filtre suau {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz, "
        f"validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz. Z només visual.",
        fontsize=11
    )

    if MISMA_ESCALA_Y_ACELERACION:
        ymax_global = limite_simetrico(
            np.vstack([acc_sin_filtrar, acc_filtrada_suave, acc_filtrada_agresiva]),
            minimo=0.15
        )
    else:
        ymax_global = None

    for eje, ax in zip(["x", "y"], axs[:2]):
        idx = IDX_EJE[eje]
        nombre = eje.upper()
        segs_eje = info_seg["segmentos_por_eje"][eje]

        pintar_segmentos_eje(
            ax,
            t,
            segs_eje["normal"],
            segs_eje["oscilatorio"],
            segs_eje["reposo"],
            segs_eje.get("impacto", []),
        )

        ax.plot(
            t,
            acc_sin_filtrar[:, idx],
            linewidth=0.5,
            alpha=0.35,
            linestyle="--",
            label="Sense filtrar" if eje == "x" else None
        )

        ax.plot(
            t,
            acc_filtrada_suave[:, idx],
            linewidth=0.9,
            alpha=0.95,
            label=f"Filtre suau {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz" if eje == "x" else None,
            color="darkblue"
        )

        ax.plot(
            t,
            acc_filtrada_agresiva[:, idx],
            linewidth=0.9,
            alpha=0.80,
            label=f"Filtre de validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz" if eje == "x" else None,
            color="purple"
        )

        ax.axhline(0, linewidth=0.8)
        ax.axhline(umbral_desplazamiento_eje(eje), linestyle=":", linewidth=0.8)
        ax.axhline(-umbral_desplazamiento_eje(eje), linestyle=":", linewidth=0.8)

        if MISMA_ESCALA_Y_ACELERACION:
            ymax = ymax_global
        else:
            ymax = limite_simetrico(acc_filtrada_suave[:, idx], minimo=0.15)

        for etiqueta, (ini, fin) in zip(segs_eje["etiquetas_todos"], segs_eje["todos"]):
            ax.text(
                (t[ini] + t[fin]) / 2,
                ymax * 0.92,
                etiqueta,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold"
            )

        if MOSTRAR_ETIQUETAS_REPOSO:
            for etiqueta, (ini, fin) in zip(segs_eje["etiquetas_reposo"], segs_eje["reposo"]):
                ax.text(
                    (t[ini] + t[fin]) / 2,
                    -ymax * 0.92,
                    etiqueta,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold"
                )

        ax.set_title(f"Eix {nombre}")
        ax.set_ylabel(f"Acc {nombre} [{unidad_acc}]")
        ax.set_ylim(-ymax, ymax)
        ax.grid(True)

    ax_z = axs[2]
    idx_z = IDX_EJE["z"]

    ax_z.plot(
        t,
        acc_sin_filtrar[:, idx_z],
        linewidth=0.5,
        alpha=0.35,
        linestyle="--",
        label="Z sense filtrar"
    )

    ax_z.plot(
        t,
        acc_filtrada_suave[:, idx_z],
        linewidth=0.9,
        alpha=0.95,
        color="darkblue",
        label=f"Z filtre suau {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz"
    )

    ax_z.plot(
        t,
        acc_filtrada_agresiva[:, idx_z],
        linewidth=0.9,
        alpha=0.80,
        color="purple",
        label=f"Z filtre de validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz"
    )

    ax_z.axhline(0, linewidth=0.8)

    if MISMA_ESCALA_Y_ACELERACION:
        ymax_z = ymax_global
    else:
        ymax_z = limite_simetrico(acc_filtrada_suave[:, idx_z], minimo=0.08)

    ax_z.set_title("Eix Z — només visual")
    ax_z.set_ylabel(f"Acc Z [{unidad_acc}]")
    ax_z.set_xlabel("Temps [s]")
    ax_z.set_ylim(-ymax_z, ymax_z)
    ax_z.grid(True)

    legend_elements = [
        Patch(facecolor="#00FFFF", alpha=0.10, label="Repòs X/Y"),
        Patch(facecolor="#FFDAB9", alpha=0.22, label="Desplaçament confirmat X/Y"),
        Patch(facecolor="#DA70D6", alpha=0.30, label="Oscil·latori X/Y"),
        Patch(facecolor="orange", alpha=0.45, label="Impacte X/Y"),
    ]

    axs[0].legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax_z.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


def crear_figura_debug_eje(
        t,
        acc_segmentacion,
        acc_validacion_desplazamiento,
        info_seg,
        unidad_acc,
        titulo,
        eje
):
    """
    Figura de depuración para un único eje.

    Representa:
      1) señal suau y señal agresiva con los segmentos finales pintados;
      2) máscaras internas de candidato/validación;
      3) cruces por cero en ventana y RMS de ventana como señales auxiliares.
    """
    eje = eje.lower()
    idx = IDX_EJE[eje]
    nombre = eje.upper()

    debug = info_seg["debug_por_eje"][eje]
    segs_eje = info_seg["segmentos_por_eje"][eje]

    fig = Figure(figsize=(13, 9), dpi=100)
    axs = fig.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.55, 1.35]}
    )

    fig.suptitle(
        f"{titulo}\nDepuració de l’eix {nombre}: validació "
        f"{FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz + tall amb filtre suau",
        fontsize=11
    )

    ax_sig = axs[0]
    ax_mask = axs[1]
    ax_cruces = axs[2]

    u_rep = UMBRAL_REPOSO[eje]
    u_desp = umbral_desplazamiento_eje(eje)

    pintar_segmentos_eje(
        ax_sig,
        t,
        segs_eje["normal"],
        segs_eje["oscilatorio"],
        segs_eje["reposo"],
        segs_eje.get("impacto", []),
    )

    ax_sig.plot(
        t,
        acc_segmentacion[:, idx],
        linewidth=0.9,
        alpha=0.95,
        color="darkblue",
        label=f"Filtre suau {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz"
    )

    ax_sig.plot(
        t,
        acc_validacion_desplazamiento[:, idx],
        linewidth=0.9,
        alpha=0.80,
        color="purple",
        label=f"Filtre de validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz"
    )

    ax_sig.axhline(0, linewidth=0.8, color="black", alpha=0.65)
    ax_sig.axhline(u_rep, linestyle=":", linewidth=1.0, color="green", label=f"+U repòs = {u_rep:.5f}")
    ax_sig.axhline(-u_rep, linestyle=":", linewidth=1.0, color="green", label="-U repòs")
    ax_sig.axhline(u_desp, linestyle="--", linewidth=1.0, color="red", label=f"+U desplaç. = {u_desp:.5f}")
    ax_sig.axhline(-u_desp, linestyle="--", linewidth=1.0, color="red", label="-U desplaç.")

    ymax = limite_simetrico(
        np.vstack([
            acc_segmentacion[:, idx],
            acc_validacion_desplazamiento[:, idx]
        ]),
        minimo=max(0.15, 1.25 * u_desp)
    )
    ax_sig.set_ylim(-ymax, ymax)

    for etiqueta, (ini, fin) in zip(segs_eje["etiquetas_todos"], segs_eje["todos"]):
        ax_sig.text(
            (t[ini] + t[fin]) / 2,
            ymax * 0.90,
            etiqueta,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold"
        )

    for etiqueta, (ini, fin) in zip(segs_eje["etiquetas_reposo"], segs_eje["reposo"]):
        ax_sig.text(
            (t[ini] + t[fin]) / 2,
            -ymax * 0.90,
            etiqueta,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold"
        )

    texto_info = (
        f"Todo candidato fuerte con {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz "
        f"passa a validació amb {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz; "
        f"els límits rojos s’ajusten amb 20 Hz\n"
        f"MaxRep={MAX_ABS_REPOSO[eje]:.5f} {unidad_acc} | "
        f"Urep={u_rep:.5f} {unidad_acc} | "
        f"Udesp={u_desp:.5f} {unidad_acc} | "
        f"creuaments alts si els creuaments de finestra >= {MIN_CRUCES_OSC}"
    )
    ax_sig.text(
        0.01,
        0.03,
        texto_info,
        transform=ax_sig.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.75")
    )

    ax_sig.set_title(f"Eix {nombre} — senyals i segments finals")
    ax_sig.set_ylabel(f"Acc {nombre} [{unidad_acc}]")
    ax_sig.grid(True)
    ax_sig.legend(fontsize=7, loc="upper right")

    pintar_segmentos_eje(
        ax_mask,
        t,
        segs_eje["normal"],
        segs_eje["oscilatorio"],
        segs_eje["reposo"],
        segs_eje.get("impacto", []),
    )

    mask_no_reposo = debug["mask_no_reposo"].astype(float)
    mask_mov_fuerte = debug["mask_movimiento_fuerte"].astype(float)
    mask_cruces_altos = (debug["cruces_ventana"] >= MIN_CRUCES_OSC).astype(float)
    mask_osc = debug["mask_osc"].astype(float)
    mask_normal = debug["mask_normal"].astype(float)
    mask_validacion = debug.get("mask_validacion_desplazamiento", np.zeros_like(mask_no_reposo)).astype(float)
    mask_confirmada = debug.get("mask_normal_confirmada", np.zeros_like(mask_no_reposo)).astype(float)
    mask_rechazada = debug.get("mask_normal_rechazada_validacion", np.zeros_like(mask_no_reposo)).astype(float)

    ax_mask.step(
        t, 1.0 * mask_no_reposo, where="post", linewidth=0.9,
        label=f"{FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz: |a| >= U repòs"
    )
    ax_mask.step(
        t, 2.0 * mask_mov_fuerte, where="post", linewidth=0.9,
        label=f"{FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz: |a| >= U desplaç."
    )
    ax_mask.step(t, 3.0 * mask_cruces_altos, where="post", linewidth=0.9, label=f"creuaments >= {MIN_CRUCES_OSC}")
    ax_mask.step(t, 4.0 * mask_osc, where="post", linewidth=1.0, label="oscil·lació inicial/validació rebutjada")
    ax_mask.step(t, 5.0 * mask_normal, where="post", linewidth=1.0, label="candidat a desplaç.")
    ax_mask.step(t, 6.0 * mask_validacion, where="post", linewidth=1.0, label="20 Hz: supera U desplaç.")
    ax_mask.step(t, 7.0 * mask_confirmada, where="post", linewidth=1.0, label="desplaç. confirmat final")
    ax_mask.step(t, 8.0 * mask_rechazada, where="post", linewidth=1.0, label="rebutjat -> oscil·lació")

    ax_mask.set_ylim(-0.25, 8.35)
    ax_mask.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax_mask.set_yticklabels([
        "inactiu",
        "no repòs",
        f"desplaç. {FILTRO_SEGMENTACION_SUAVE_HZ:.0f}",
        "creuaments",
        "osc",
        "cand.",
        "desplaç. 20",
        "conf.",
        "rebutj.",
    ])
    ax_mask.set_title(f"Eix {nombre} — condicions internes abans/després de la validació")
    ax_mask.set_ylabel("Màscares")
    ax_mask.grid(True)
    ax_mask.legend(fontsize=6.5, loc="upper right")

    pintar_segmentos_eje(
        ax_cruces,
        t,
        segs_eje["normal"],
        segs_eje["oscilatorio"],
        segs_eje["reposo"],
        segs_eje.get("impacto", []),
    )

    ax_cruces.plot(
        t,
        debug["cruces_ventana"],
        linewidth=0.9,
        alpha=0.85,
        linestyle="-",
        label="nre. de creuaments per zero en finestra"
    )
    ax_cruces.axhline(
        MIN_CRUCES_OSC,
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        label=f"llindar de creuaments = {MIN_CRUCES_OSC}"
    )
    ax_cruces.set_ylabel("Creuaments/finestra")
    ax_cruces.grid(True)

    ax_rms = ax_cruces.twinx()
    ax_rms.plot(
        t,
        debug["rms_ventana"],
        linewidth=0.75,
        alpha=0.50,
        linestyle="--",
        label="RMS de finestra"
    )
    ax_rms.axhline(
        u_rep,
        linestyle=":",
        linewidth=0.9,
        alpha=0.7,
        label="U repòs"
    )
    ax_rms.set_ylabel(f"RMS [{unidad_acc}]")

    h1, l1 = ax_cruces.get_legend_handles_labels()
    h2, l2 = ax_rms.get_legend_handles_labels()
    ax_cruces.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax_cruces.set_title(f"Eix {nombre} — creuaments per zero i RMS de finestra")
    ax_cruces.set_xlabel("Temps [s]")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def segmentos_se_solapan_usuario(seg_a, seg_b):
    """Devuelve True si dos intervalos de mostres se solapan."""
    ini_a, fin_a = int(seg_a[0]), int(seg_a[1])
    ini_b, fin_b = int(seg_b[0]), int(seg_b[1])
    return not (fin_a < ini_b or fin_b < ini_a)


def crear_figura_aceleracion_modo_usuario(
        t,
        acc_sin_filtrar,
        acc_filtrada_suave,
        acc_filtrada_agresiva,
        info_seg,
        info_identificacion_bd,
        unidad_acc,
        titulo
):
    """
    Vista simplificada para el modo usuario.

    Eixs X e Y:
    - verde: segmento correcto, independientemente de su tipo;
    - amarillo: comportamiento anómalo o desconocido;
    - rojo: intervalo en el que se ha detectado un impacto.

    Solo se escriben las etiquetas de los segmentos anómalos o afectados por
    un impacto. El eje Z no se segmenta; únicamente muestra los impactos.
    """
    fig = Figure(figsize=(13, 8), dpi=100)
    axs = fig.subplots(
        3,
        1,
        sharex=True,
        sharey=MISMA_ESCALA_Y_ACELERACION
    )

    fig.suptitle(
        f"{titulo}\nEstat dels segments",
        fontsize=11
    )

    if MISMA_ESCALA_Y_ACELERACION:
        ymax_global = limite_simetrico(
            np.vstack([
                acc_sin_filtrar,
                acc_filtrada_suave,
                acc_filtrada_agresiva,
            ]),
            minimo=0.15
        )
    else:
        ymax_global = None

    info_por_eje = {}

    if info_identificacion_bd is not None:
        info_por_eje = info_identificacion_bd.get("por_eje", {})

    for eje, ax in zip(["x", "y"], axs[:2]):
        idx = IDX_EJE[eje]
        nombre = eje.upper()
        info_eje_ident = info_por_eje.get(eje, {})
        segmentos_identificados = list(info_eje_ident.get("segmentos", []))
        segmentos_impacto = list(
            info_seg.get("segmentos_por_eje", {})
            .get(eje, {})
            .get("impacto", [])
        )

        if len(segmentos_identificados) == 0:
            segs_eje = info_seg.get("segmentos_por_eje", {}).get(eje, {})
            etiquetas_todos = list(segs_eje.get("etiquetas_todos", []))
            etiquetas_reposo = list(segs_eje.get("etiquetas_reposo", []))

            segmentos_identificados = []

            for tipo_seg, tipo_patron in [
                ("normal", "desplazamiento"),
                ("oscilatorio", "oscilatorio"),
            ]:
                objetivo = set(segs_eje.get(tipo_seg, []))

                for seg, etiqueta in zip(
                        segs_eje.get("todos", []),
                        etiquetas_todos
                ):
                    if seg in objetivo:
                        segmentos_identificados.append({
                            "ini": int(seg[0]),
                            "fin": int(seg[1]),
                            "etiqueta": etiqueta,
                            "tipo_patron": tipo_patron,
                            "anomalo": True,
                        })

            for seg, etiqueta in zip(
                    segs_eje.get("reposo", []),
                    etiquetas_reposo
            ):
                segmentos_identificados.append({
                    "ini": int(seg[0]),
                    "fin": int(seg[1]),
                    "etiqueta": etiqueta,
                    "tipo_patron": "reposo",
                    "anomalo": True,
                })

            segmentos_identificados.sort(
                key=lambda item: (item["ini"], item["fin"])
            )

        for segmento in segmentos_identificados:
            ini = max(0, int(segmento.get("ini", 0)))
            fin = min(len(t) - 1, int(segmento.get("fin", 0)))

            if fin < ini:
                continue

            es_anomalo = bool(segmento.get("anomalo", False))
            color_estado = "yellow" if es_anomalo else "green"
            alpha_estado = 0.30 if es_anomalo else 0.16

            ax.axvspan(
                t[ini],
                t[fin],
                color=color_estado,
                alpha=alpha_estado,
                zorder=0
            )

            if es_anomalo:
                ax.axvline(
                    t[ini],
                    color="#D4AE04",
                    alpha=1.0,
                    linewidth=0.8,
                    zorder=1
                )
                ax.axvline(
                    t[fin],
                    color="#D4AE04",
                    alpha=1.0,
                    linewidth=0.8,
                    zorder=1
                )

        for ini_imp, fin_imp in segmentos_impacto:
            ini_imp = max(0, int(ini_imp))
            fin_imp = min(len(t) - 1, int(fin_imp))

            if fin_imp < ini_imp:
                continue

            ax.axvspan(
                t[ini_imp],
                t[fin_imp],
                color="red",
                alpha=0.45,
                zorder=1
            )

        ax.plot(
            t,
            acc_sin_filtrar[:, idx],
            linewidth=0.5,
            alpha=0.35,
            linestyle="--",
            label="Sense filtrar" if eje == "x" else None,
            zorder=2
        )

        ax.plot(
            t,
            acc_filtrada_suave[:, idx],
            linewidth=0.9,
            alpha=0.95,
            color="darkblue",
            label=(
                f"Filtre suau {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz"
                if eje == "x" else None
            ),
            zorder=3
        )

        ax.plot(
            t,
            acc_filtrada_agresiva[:, idx],
            linewidth=0.9,
            alpha=0.80,
            color="purple",
            label=(
                f"Filtre de validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz"
                if eje == "x" else None
            ),
            zorder=3
        )

        ax.axhline(0, linewidth=0.8, zorder=2)

        if MISMA_ESCALA_Y_ACELERACION:
            ymax = ymax_global
        else:
            ymax = limite_simetrico(
                np.column_stack([
                    acc_sin_filtrar[:, idx],
                    acc_filtrada_suave[:, idx],
                    acc_filtrada_agresiva[:, idx],
                ]),
                minimo=0.15
            )

        for segmento in segmentos_identificados:
            if not bool(segmento.get("anomalo", False)):
                continue

            seg_tuple = (
                int(segmento.get("ini", 0)),
                int(segmento.get("fin", 0)),
            )

            if any(
                segmentos_se_solapan_usuario(seg_tuple, impacto)
                for impacto in segmentos_impacto
            ):
                continue

            ini, fin = seg_tuple
            ini = max(0, ini)
            fin = min(len(t) - 1, fin)

            if fin < ini:
                continue

            ax.text(
                (t[ini] + t[fin]) / 2,
                ymax * 0.92,
                str(segmento.get("etiqueta", "")),
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                zorder=5
            )

        etiquetas_impacto_dibujadas = set()

        for ini_imp, fin_imp in segmentos_impacto:
            impacto = (int(ini_imp), int(fin_imp))

            for segmento in segmentos_identificados:
                seg_tuple = (
                    int(segmento.get("ini", 0)),
                    int(segmento.get("fin", 0)),
                )

                if not segmentos_se_solapan_usuario(seg_tuple, impacto):
                    continue

                etiqueta = str(segmento.get("etiqueta", ""))
                clave = (etiqueta, impacto[0], impacto[1])

                if not etiqueta or clave in etiquetas_impacto_dibujadas:
                    continue

                ini_texto = max(0, impacto[0])
                fin_texto = min(len(t) - 1, impacto[1])

                if fin_texto < ini_texto:
                    continue

                ax.text(
                    (t[ini_texto] + t[fin_texto]) / 2,
                    -ymax * 0.92,
                    etiqueta,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    zorder=5
                )
                etiquetas_impacto_dibujadas.add(clave)

        ax.set_title(f"Eix {nombre}")
        ax.set_ylabel(f"Acc {nombre} [{unidad_acc}]")
        ax.set_ylim(-ymax, ymax)
        ax.grid(True)

    ax_z = axs[2]
    idx_z = IDX_EJE["z"]
    segmentos_impacto_z = list(
        info_seg.get("segmentos_por_eje", {})
        .get("x", {})
        .get("impacto", [])
    )

    for ini_imp, fin_imp in segmentos_impacto_z:
        ini_imp = max(0, int(ini_imp))
        fin_imp = min(len(t) - 1, int(fin_imp))

        if fin_imp >= ini_imp:
            ax_z.axvspan(
                t[ini_imp],
                t[fin_imp],
                color="red",
                alpha=0.45,
                zorder=1
            )

    ax_z.plot(
        t,
        acc_sin_filtrar[:, idx_z],
        linewidth=0.5,
        alpha=0.35,
        linestyle="--",
        label="Z sense filtrar",
        zorder=2
    )
    ax_z.plot(
        t,
        acc_filtrada_suave[:, idx_z],
        linewidth=0.9,
        alpha=0.95,
        color="darkblue",
        label=f"Z filtre suau {FILTRO_SEGMENTACION_SUAVE_HZ:.0f} Hz",
        zorder=3
    )
    ax_z.plot(
        t,
        acc_filtrada_agresiva[:, idx_z],
        linewidth=0.9,
        alpha=0.80,
        color="purple",
        label=f"Z filtre de validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.0f} Hz",
        zorder=3
    )
    ax_z.axhline(0, linewidth=0.8, zorder=2)

    if MISMA_ESCALA_Y_ACELERACION:
        ymax_z = ymax_global
    else:
        ymax_z = limite_simetrico(
            np.column_stack([
                acc_sin_filtrar[:, idx_z],
                acc_filtrada_suave[:, idx_z],
                acc_filtrada_agresiva[:, idx_z],
            ]),
            minimo=0.08
        )

    ax_z.set_title("Eix Z — només visual")
    ax_z.set_ylabel(f"Acc Z [{unidad_acc}]")
    ax_z.set_xlabel("Temps [s]")
    ax_z.set_ylim(-ymax_z, ymax_z)
    ax_z.grid(True)

    legend_elements = [
        Patch(facecolor="green", alpha=0.16, label="Segment correcte"),
        Patch(facecolor="yellow", alpha=0.30, label="Comportament anòmal"),
        Patch(facecolor="red", alpha=0.45, label="Impacte"),
    ]
    axs[0].legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax_z.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def crear_figura_modulo_aceleracion(
        t,
        acc_sin_filtrar,
        unidad_acc,
        titulo
):
    """
    Representa el módulo de la aceleración:

        |a| = sqrt(ax² + ay² + az²)

    Se utiliza la señal sin filtrar que ya ha sido centrada mediante
    la resta de la media.
    """
    acc = np.asarray(acc_sin_filtrar, dtype=float)

    if acc.ndim != 2 or acc.shape[1] < 3:
        raise ValueError(
            "L’acceleració ha de tindre almenys tres columnes: X, Y i Z."
        )

    modulo = np.sqrt(
        acc[:, 0] ** 2
        + acc[:, 1] ** 2
        + acc[:, 2] ** 2
    )

    fig = Figure(figsize=(13, 7), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(
        t,
        modulo,
        linewidth=0.8,
        label="Mòdul de l’acceleració"
    )

    if DETECTAR_IMPACTOS:
        ax.axhline(
            umbral_impacto_unidades_actuales(),
            linestyle="--",
            linewidth=1.0,
            label=f"Llindar d’impacte ({UMBRAL_IMPACTO_G:.1f} g)"
        )

    ax.set_title(
        f"{titulo}\nMòdul de l’acceleració"
    )
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel(f"|a| [{unidad_acc}]")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()

    return fig


def crear_figura_frecuencia(f, espectro, unidad_acc, titulo):
    fig = Figure(figsize=(13, 8), dpi=100)
    axs = fig.subplots(3, 1, sharex=True)

    fig.suptitle(f"{titulo}\nEspectre en freqüència", fontsize=11)

    nombres = ["X", "Y", "Z"]

    ymax = np.max(np.abs(espectro))

    if ymax < 1e-12:
        ymax = 1.0

    ymax *= 1.10

    for i, ax in enumerate(axs):
        ax.plot(f, espectro[:, i], linewidth=0.8)
        ax.set_title(f"Eix {nombres[i]}")
        ax.set_ylabel(f"FFT {nombres[i]} [{unidad_acc}]")
        ax.set_ylim(0, ymax)
        ax.grid(True)

    axs[-1].set_xlabel("Freqüència [Hz]")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


##################################################
# IDENTIFICACIÓN DE SEGMENTOS CON BASE DE DATOS
##################################################

def blob_a_array_bd(blob):
    """Recupera un array NumPy guardado como BLOB np.savez_compressed."""
    if blob is None:
        return np.array([], dtype=float)

    buffer = io.BytesIO(blob)

    try:
        with np.load(buffer) as data:
            return data["data"]
    except Exception:
        return np.array([], dtype=float)


def calcular_espectro_segmento_1d_bd(x):
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N < 4:
        return np.array([0.0]), np.array([0.0])

    ventana = np.hanning(N)
    ganancia = np.mean(ventana)

    f = np.fft.rfftfreq(N, d=1 / Fs)
    x0 = x - np.mean(x)
    xw = x0 * ventana

    espectro = np.abs(np.fft.rfft(xw)) / (N * max(ganancia, 1e-12))

    if len(espectro) > 2:
        espectro[1:-1] *= 2

    return f, espectro


def limitar_espectro_identificacion(f, espectro):
    f = np.asarray(f, dtype=float)
    espectro = np.asarray(espectro, dtype=float)

    if len(f) == 0 or len(espectro) == 0:
        return np.array([0.0]), np.array([0.0])

    n = min(len(f), len(espectro))
    f = f[:n]
    espectro = espectro[:n]

    if FREQ_MAX_COMPARACION_IDENTIFICACION_BD is not None:
        mascara = f <= float(FREQ_MAX_COMPARACION_IDENTIFICACION_BD)

        if np.any(mascara):
            f = f[mascara]
            espectro = espectro[mascara]

    return f, espectro


def calcular_espectro_comparacion_identificacion(x):
    """
    Calcula el vector espectral usado para identificación.

    El espectro se recorta a FREQ_MAX_COMPARACION_IDENTIFICACION_BD,
    se remuestrea a una longitud común y se normaliza por su máximo para
    comparar solo la forma frecuencial, no la amplitud absoluta.
    """
    f, espectro = calcular_espectro_segmento_1d_bd(x)
    _, espectro = limitar_espectro_identificacion(f, espectro)

    if len(espectro) < 2:
        espectro_cmp = np.zeros(LONGITUD_ESPECTRO_IDENTIFICACION_BD)
    else:
        espectro_cmp = remuestrear_1d(espectro, LONGITUD_ESPECTRO_IDENTIFICACION_BD)

    max_abs = float(np.max(np.abs(espectro_cmp))) if len(espectro_cmp) > 0 else 0.0

    if max_abs > 1e-12:
        espectro_cmp = espectro_cmp / max_abs

    return np.asarray(espectro_cmp, dtype=float)


def distancia_espectros_identificacion(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) == 0 or len(b) == 0:
        return np.inf

    if len(a) != len(b):
        b = remuestrear_1d(b, len(a))

    return float(np.sqrt(np.mean((a - b) ** 2)))


def distancia_referencia_identificacion(seg, referencia):
    """Distancia entre dos segmentos usando forma, pico y duración."""
    forma_seg = seg.get("forma")
    forma_ref = referencia.get("forma")

    if forma_seg is None or forma_ref is None:
        return 1e6

    forma_seg = np.asarray(forma_seg, dtype=float)
    forma_ref = np.asarray(forma_ref, dtype=float)

    if len(forma_seg) == 0 or len(forma_ref) == 0:
        return 1e6

    if len(forma_seg) != len(forma_ref):
        forma_ref = remuestrear_1d(forma_ref, len(forma_seg))

    d = distancia_formas_1d(
        forma_seg,
        forma_ref,
        seg.get("metrica_dtw"),
        referencia.get("metrica_dtw")
    )

    if not np.isfinite(d):
        return 1e6

    return float(d)


def distancia_forma_identificacion(seg, grupo):
    """
    Distancia usada SOLO para clasificar un desplazamiento.

    aplica exactamente los pesos del entrenamiento y compara el segmento
    actual contra todos los segmentos individuales almacenados en el grupo.
    La distància del grup es la menor distància a uno de sus miembros. Si la
    tabla individual no está disponible, se usa el representante del grupo.
    """
    referencias = grupo.get("miembros_clasificacion", [])

    if not referencias:
        referencias = [grupo]

    distancias = [
        distancia_referencia_identificacion(seg, referencia)
        for referencia in referencias
    ]

    if len(distancias) == 0:
        return 1e6

    return float(min(distancias))


def calcular_features_patron_1d_bd(x):
    """Calcula las características físicas usadas por la identificación."""
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return {
            "rms": 0.0,
            "pico_pico": 0.0,
            "offset_abs": 0.0,
        }

    return {
        "rms": float(np.sqrt(np.mean(x * x))),
        "pico_pico": float(np.max(x) - np.min(x)),
        "offset_abs": float(np.max(np.abs(x))),
    }


def metrica_dtw_desde_senal_segmento(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return {
            "duracion_s": 0.0,
            "rms": 0.0,
            "pico_abs": 0.0,
            "pico_firmado": 0.0,
            "signo_pico": 0.0,
        }

    idx_pico = int(np.argmax(np.abs(x)))
    pico_firmado = float(x[idx_pico])

    return {
        "duracion_s": float(len(x) / Fs),
        "rms": float(np.sqrt(np.mean(x * x))),
        "pico_abs": float(np.max(np.abs(x))),
        "pico_firmado": pico_firmado,
        "signo_pico": float(np.sign(pico_firmado)),
    }


def calcular_forma_normalizada_identificacion(x):
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return np.zeros(LONGITUD_DTW)

    forma = remuestrear_1d(x, LONGITUD_DTW)
    return z_normalizar_1d(forma)


def obtener_grupos_bd_por_tipo_eje(grupos_bd, tipo_patron, eje):
    return [
        g for g in grupos_bd
        if g.get("tipo_patron") == tipo_patron and g.get("eje") == eje
    ]


def obtener_umbral_forma_desplazamiento_bd(eje):
    """
    Umbral usado SOLO para asignar desplazamientos a un grupo por forma.

    No se usa para oscilatorio ni reposo.
    """
    valor = UMBRAL_FORMA_CLASIFICACION_DESPLAZAMIENTO_BD.get(eje)

    if valor is None:
        return DISTANCIA_CLUSTER_DTW[eje]

    return float(valor)


def crear_segmentos_actuales_identificacion(acc_sin_filtrar, acc_filtrada, info_seg, eje):
    segmentos_actuales = []

    tipos = [
        ("normal", "desplazamiento"),
        ("oscilatorio", "oscilatorio"),
        ("reposo", "reposo"),
    ]

    for tipo_seg, tipo_patron in tipos:
        segmentos, etiquetas = obtener_etiquetas_segmentos_tipo_eje(
            info_seg=info_seg,
            eje=eje,
            tipo=tipo_seg
        )

        for segmento, etiqueta in zip(segmentos, etiquetas):
            segmentos_actuales.append(
                crear_segmento_actual_identificacion(
                    acc_sin_filtrar=acc_sin_filtrar,
                    acc_filtrada=acc_filtrada,
                    eje=eje,
                    tipo_patron=tipo_patron,
                    segmento=segmento,
                    etiqueta=etiqueta
                )
            )

    segmentos_actuales.sort(key=lambda s: (s["ini"], s["fin"], s["tipo_patron"]))
    return segmentos_actuales


def grupo_comparable_con_segmento_matriz(seg, grupo, grupos_eje, eje):
    """
    Indica si esa celda de la matriz debe calcularse.

    La lógica actual evita comparar oscilatorio/reposo contra grupos que no sean su
    patrón directo de BD. Así la matriz refleja la misma lógica que la
    identificación.
    """
    tipo_seg = seg.get("tipo_patron")

    if grupo.get("eje") != eje:
        return False

    if grupo.get("tipo_patron") != tipo_seg:
        return False

    if tipo_seg in TIPOS_COMPARACION_DIRECTA_BD:
        grupo_ref = seleccionar_grupo_directo_tipo_bd(grupos_eje, tipo_seg, eje)
        return grupo_ref is not None and int(grupo.get("id")) == int(grupo_ref.get("id"))

    return True


def construir_matriz_forma_frecuencia_segmentos_grupos(segmentos, grupos_eje, eje):
    """
    Construye la matriz de comparación entre segmentos detectados y grupos BD.

    - Desplazamiento: distància de forma usada para asignar el grupo.
    - Oscil·latori/reposo: distància espectral como ayuda visual.
    """
    segmentos_validos = [s for s in segmentos if s.get("eje") == eje]
    grupos_validos = [g for g in grupos_eje if g.get("eje") == eje]

    etiquetas_segmentos = [s["etiqueta"] for s in segmentos_validos]
    etiquetas_grupos = [g["etiqueta"] for g in grupos_validos]

    D = None

    if len(segmentos_validos) > 0 and len(grupos_validos) > 0:
        D = np.full(
            (len(segmentos_validos), len(grupos_validos)),
            np.nan,
            dtype=float
        )

        for i, seg in enumerate(segmentos_validos):
            for j, grupo in enumerate(grupos_validos):
                if not grupo_comparable_con_segmento_matriz(
                        seg,
                        grupo,
                        grupos_eje,
                        eje
                ):
                    continue

                if seg.get("tipo_patron") == "desplazamiento":
                    D[i, j] = distancia_forma_identificacion(seg, grupo)
                else:
                    D[i, j] = distancia_espectros_identificacion(
                        seg.get("espectro_comparacion"),
                        grupo.get("espectro_comparacion")
                    )

    return {
        "D": D,
        "etiquetas_segmentos": etiquetas_segmentos,
        "etiquetas_grupos": etiquetas_grupos,
    }


def seleccionar_grupo_directo_tipo_bd(grupos_eje, tipo_patron, eje):
    """
    Devuelve el patrón fijo de la BD usado para comparación directa.

    Los segmentos oscilatorios y de reposo no buscan el grupo más
    cercano: se comparan con el patrón de su mismo tipo/eje guardado en la BD.

    Si por acumulación de entrenamientos hubiese más de un patrón de ese tipo,
    se toma el de menor ID para que la decisión sea determinista y no dependa
    de cuál quede más cerca.
    """
    grupos_tipo = [
        g for g in grupos_eje
        if g.get("tipo_patron") == tipo_patron and g.get("eje") == eje
    ]

    if len(grupos_tipo) == 0:
        return None

    grupos_tipo.sort(key=lambda g: int(g.get("id", 10**12)))
    return grupos_tipo[0]


##################################################
# AJUSTE: IDENTIFICACIÓN POR LÍMITES
##################################################


# Margen físico aplicado a RMS, pico-pico, offset y duración.
# 0.30 equivale a permitir hasta un 30 % por encima del máximo normal.
MARGEN_LIMITES_IDENTIFICACION_BD = 0.3

# Margen específico para ratios frecuenciales por bandas.
MARGEN_FRECUENCIA_BANDAS_BD = 0.80

# Validaciones activas.
USAR_LIMITES_FISICOS_DESPLAZAMIENTO_BD = True
USAR_LIMITES_FRECUENCIA_BANDAS_BD = True


# Tolerancias absolutas para evitar falsos positivos por diferencias mínimas.
TOL_ABS_ACEL_LIMITES_BD = 0.002
TOL_ABS_DURACION_LIMITES_BD = 0.050

TOL_ABS_RIZADO_PICO_PICO_BD = 0.005


def crear_figura_mensaje_identificacion(texto, titulo=""):
    fig = Figure(figsize=(13, 6), dpi=100)
    ax = fig.add_subplot(111)

    if titulo:
        fig.suptitle(titulo, fontsize=11)

    ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def crear_figura_matriz_bd_identificacion_eje(info_eje, titulo):
    if info_eje is None:
        return crear_figura_mensaje_identificacion("No hi ha informació d'identificació.")

    eje = info_eje["eje"]
    matriz = info_eje.get("matriz", {})
    D = matriz.get("D")
    etiquetas_segmentos = matriz.get("etiquetas_segmentos", [])
    etiquetas_grupos = matriz.get("etiquetas_grupos", [])

    if D is None or len(etiquetas_segmentos) == 0 or len(etiquetas_grupos) == 0:
        return crear_figura_mensaje_identificacion(
            f"No hi ha cap matriu BD per a l’eix {eje.upper()}.\n"
            "Pot ser que no hi haja segments detectats o que la BD no tinga grups per a eixe eix.",
            titulo=titulo
        )

    n_filas, n_cols = D.shape
    ancho = max(8, min(18, 0.45 * n_cols + 5))
    alto = max(6, min(18, 0.30 * n_filas + 4))

    fig = Figure(figsize=(ancho, alto), dpi=100)
    ax = fig.add_subplot(111)

    D_plot = np.asarray(D, dtype=float)
    valores_finitos = D_plot[np.isfinite(D_plot)]

    if len(valores_finitos) == 0:
        return crear_figura_mensaje_identificacion(
            f"No hi ha comparacions vàlides per a la matriu BD de l’eix {eje.upper()}.",
            titulo=titulo
        )

    try:
        cmap = matplotlib.colormaps[CMAP_MATRIZ_DISTANCIAS].copy()
    except AttributeError:
        cmap = matplotlib.cm.get_cmap(CMAP_MATRIZ_DISTANCIAS).copy()

    cmap.set_bad(color="lightgray")
    im = ax.imshow(
        np.ma.masked_invalid(D_plot),
        cmap=cmap,
        interpolation="nearest",
        aspect="auto"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        f"Distància d’identificació {eje.upper()}",
        rotation=270,
        labelpad=15
    )

    ax.set_title(
        f"{titulo}\nMatriu BD — eix {eje.upper()} | vertical = segments, horitzontal = grups BD",
        fontsize=11
    )
    ax.set_xlabel("Grups de la base de dades")
    ax.set_ylabel("Segments detectats")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_filas))
    ax.set_xticklabels(etiquetas_grupos, rotation=90, fontsize=8)
    ax.set_yticklabels(etiquetas_segmentos, fontsize=8)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    if ANOTAR_VALORES_MATRIZ and (n_filas * n_cols) <= MAX_SEGMENTOS_ANOTAR_MATRIZ:
        n_max = max(n_filas, n_cols)
        fontsize_valores = 8 if n_max <= 12 else 7 if n_max <= 25 else 5

        for i in range(n_filas):
            for j in range(n_cols):
                valor = D[i, j]

                if not np.isfinite(valor):
                    ax.text(
                        j,
                        i,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=fontsize_valores,
                        color="black",
                        clip_on=True
                    )
                    continue

                rgba = im.cmap(im.norm(valor))
                r, g, b, _ = rgba
                luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
                color_texto = "white" if luminancia < 0.50 else "black"

                ax.text(
                    j,
                    i,
                    f"{valor:.{DECIMALES_MATRIZ_DISTANCIAS}f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize_valores,
                    color=color_texto,
                    clip_on=True
                )

    fig.tight_layout()
    return fig


def construir_paginas_segmentos_bd_identificacion(info_identificacion):
    paginas = []

    if info_identificacion is None or info_identificacion.get("error_bd") is not None:
        return paginas

    grupos_bd = info_identificacion.get("grupos_bd", [])
    grupo_por_id = {g["id"]: g for g in grupos_bd}
    asignaciones = {}
    anomalias_sin_grupo = []

    for eje, info_eje in info_identificacion.get("por_eje", {}).items():
        for seg in info_eje.get("segmentos", []):
            gid = seg.get("grupo_id")

            if gid is None:
                anomalias_sin_grupo.append(seg)
            else:
                asignaciones.setdefault(gid, []).append(seg)

    for gid in sorted(asignaciones.keys()):
        grupo = grupo_por_id.get(gid)
        segmentos = asignaciones[gid]

        if grupo is None:
            continue

        paginas.append({
            "tipo": "grupo",
            "grupo": grupo,
            "segmentos": segmentos,
        })

    if len(anomalias_sin_grupo) > 0:
        paginas.append({
            "tipo": "sin_grupo",
            "grupo": None,
            "segmentos": anomalias_sin_grupo,
        })

    return paginas


def crear_figura_segmentos_bd_identificacion_pagina(pagina, titulo):
    fig = Figure(figsize=(13, 9), dpi=100)
    axs = fig.subplots(3, 1, sharex=False)
    ax1, ax2, ax3 = axs

    segmentos = pagina.get("segmentos", [])
    grupo = pagina.get("grupo")

    if grupo is None:
        titulo_pagina = "Segments sense grup compatible — possible comportament anòmal"
    else:
        titulo_pagina = (
            f"Grup BD {grupo['id']} | {grupo['tipo_patron']} | eix {grupo['eje'].upper()} | "
            f"{len(segmentos)} segment(s) detectat(s)"
        )

    fig.suptitle(f"{titulo}\n{titulo_pagina}", fontsize=11)

    if grupo is not None:
        rep = grupo.get("segmento_original", np.array([], dtype=float))

        if len(rep) > 1:
            t_rep = np.arange(len(rep)) / Fs
            ax1.plot(t_rep, rep, linewidth=2.2, color="black", label=f"Representant {grupo['etiqueta']}")

        rep_forma = grupo.get("forma", np.array([], dtype=float))

        if len(rep_forma) > 1:
            x_rep = np.linspace(0, 100, len(rep_forma))
            ax2.plot(x_rep, rep_forma, linewidth=2.2, color="black", label=f"Representant {grupo['etiqueta']}")

        if len(rep) > 1:
            f_rep, espectro_rep_original = calcular_espectro_segmento_1d_bd(rep)

            f_rep, espectro_rep_original = limitar_espectro_identificacion(
                f_rep,
                espectro_rep_original
            )

            ax3.plot(
                f_rep,
                espectro_rep_original,
                linewidth=2.2,
                color="black",
                label=f"Representant {grupo['etiqueta']}"
            )

    for seg in segmentos:
        etiqueta = seg["etiqueta"]
        sufijo = " ANÒMAL" if seg.get("anomalo") else ""
        label = f"{etiqueta}{sufijo}"

        y = seg.get("segmento_original", np.array([], dtype=float))

        if len(y) > 1:
            t = np.arange(len(y)) / Fs
            ax1.plot(t, y, linewidth=0.9, alpha=0.65, label=label)

        forma = seg.get("forma")

        if forma is not None and len(forma) > 1:
            x = np.linspace(0, 100, len(forma))
            ax2.plot(x, forma, linewidth=0.9, alpha=0.65, label=label)

        if len(y) > 1:
            f_seg, espectro_seg_original = calcular_espectro_segmento_1d_bd(y)

            f_seg, espectro_seg_original = limitar_espectro_identificacion(
                f_seg,
                espectro_seg_original
            )

            ax3.plot(
                f_seg,
                espectro_seg_original,
                linewidth=0.9,
                alpha=0.65,
                label=label
            )

    ax1.set_title("Segment original detectat vs representant BD")
    ax1.set_xlabel("Temps relatiu [s]")
    ax1.set_ylabel("Acceleració")
    ax1.grid(True)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_title("Forma normalitzada utilitzada per a la comparació")
    ax2.set_xlabel("Temps normalitzat [%]")
    ax2.set_ylabel("Valor normalitzat")
    ax2.grid(True)
    ax2.legend(fontsize=8, loc="upper right")

    ax3.set_title("Freqüències de les senyals reals sense filtrar vs representant BD")
    ax3.set_xlabel("Freqüència [Hz]")
    ax3.set_ylabel("Amplitud")
    ax3.grid(True)
    ax3.legend(fontsize=8, loc="upper right")

    if FREQ_MAX_COMPARACION_IDENTIFICACION_BD is not None:
        ax3.set_xlim(0, float(FREQ_MAX_COMPARACION_IDENTIFICACION_BD))

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig

def insertar_segmentos_bd_identificacion_en_frame(frame_padre, info_identificacion, titulo):
    barra_superior = ttk.Frame(frame_padre)
    barra_superior.pack(fill=tk.X, padx=5, pady=5)

    frame_figura = ttk.Frame(frame_padre)
    frame_figura.pack(fill=tk.BOTH, expand=True)

    paginas = construir_paginas_segmentos_bd_identificacion(info_identificacion)

    estado = {
        "pagina": 0,
        "n_paginas": max(1, len(paginas)),
    }

    btn_anterior = ttk.Button(barra_superior, text="⟨ Anterior")
    btn_siguiente = ttk.Button(barra_superior, text="Següent ⟩")
    lbl_pagina = ttk.Label(barra_superior, text="Pàgina 1/1")

    btn_anterior.pack(side=tk.LEFT, padx=5)
    lbl_pagina.pack(side=tk.LEFT, padx=10)
    btn_siguiente.pack(side=tk.LEFT, padx=5)

    def limpiar():
        for widget in frame_figura.winfo_children():
            widget.destroy()

    def renderizar_pagina():
        limpiar()

        if len(paginas) == 0:
            fig = crear_figura_mensaje_identificacion(
                "No hi ha segments assignats a grups de la BD.",
                titulo=titulo
            )
        else:
            estado["n_paginas"] = len(paginas)
            estado["pagina"] = max(0, min(estado["pagina"], estado["n_paginas"] - 1))
            fig = crear_figura_segmentos_bd_identificacion_pagina(
                paginas[estado["pagina"]],
                titulo=titulo
            )

        canvas = FigureCanvasTkAgg(fig, master=frame_figura)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, frame_figura)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        lbl_pagina.config(text=f"Pàgina {estado['pagina'] + 1} / {estado['n_paginas']}")
        btn_anterior.config(state=tk.NORMAL if estado["pagina"] > 0 else tk.DISABLED)
        btn_siguiente.config(state=tk.NORMAL if estado["pagina"] < estado["n_paginas"] - 1 else tk.DISABLED)

    def pagina_anterior():
        if estado["pagina"] > 0:
            estado["pagina"] -= 1
            renderizar_pagina()

    def pagina_siguiente():
        if estado["pagina"] < estado["n_paginas"] - 1:
            estado["pagina"] += 1
            renderizar_pagina()

    btn_anterior.config(command=pagina_anterior)
    btn_siguiente.config(command=pagina_siguiente)
    renderizar_pagina()


##################################################
# TKINTER
##################################################

def insertar_figura_en_frame(frame, fig):
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()

    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    return canvas


# ============================================================
# VISOR INDEPENDIENTE DE LA BASE DE DATOS
# ============================================================
# Visor de consulta de los patrones y segmentos guardados.
USAR_ESCALA_X_COMUN_DESPLAZAMIENTO_SEGMENTOS_BD = True
MARGEN_ESCALA_X_DESPLAZAMIENTO_SEGMENTOS_BD = 1.05

COLUMNAS_OBSOLETAS_VISOR_BD = {
    "dtw_distancia_min",
    "factor_cresta_original_min",
    "factor_cresta_original_max",
    "factor_cresta_filtrada_min",
    "factor_cresta_filtrada_max",
    "zcr_original_min",
    "zcr_original_max",
    "zcr_filtrada_min",
    "zcr_filtrada_max",
    "jerk_max_original_min",
    "jerk_max_original_max",
    "rms_filtrada_min",
    "rms_filtrada_max",
    "pico_pico_filtrada_min",
    "pico_pico_filtrada_max",
    "jerk_max_filtrada_min",
    "jerk_max_filtrada_max",
    "freq_dom_original_min",
    "freq_dom_original_max",
    "freq_dom_filtrada_min",
    "freq_dom_filtrada_max",
    "offset_abs_filtrada_max",

    "rms_original_min",
    "pico_pico_original_min",
}


def leer_patrones_normales_bd():
    ruta_bd = RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES

    if not ruta_bd.exists():
        return [], [], f"Encara no existeix la base de dades:\n{ruta_bd}"

    conn = sqlite3.connect(ruta_bd)
    cur = conn.cursor()

    try:
        cur.execute("PRAGMA table_info(patrones_normales_dtw)")
        info_columnas = cur.fetchall()

        if len(info_columnas) == 0:
            conn.close()
            return [], [], "La taula patrones_normales_dtw encara no existeix."

        columnas = [
            col[1]
            for col in info_columnas
            if not col[1].endswith("_blob") and col[1] not in COLUMNAS_OBSOLETAS_VISOR_BD
        ]

        consulta = "SELECT " + ", ".join(columnas) + " FROM patrones_normales_dtw"
        cur.execute(consulta)
        filas = cur.fetchall()

        conn.close()
        return columnas, filas, None

    except sqlite3.Error as e:
        conn.close()
        return [], [], f"Error en llegir la base de dades:\n{e}"


def formatear_valor_bd(valor):
    if valor is None:
        return ""

    if isinstance(valor, float):
        return f"{valor:.6g}"

    return str(valor)


def insertar_bd_en_frame(frame_padre):
    barra_superior = ttk.Frame(frame_padre)
    barra_superior.pack(fill=tk.X, padx=5, pady=5)

    frame_tabla = ttk.Frame(frame_padre)
    frame_tabla.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    lbl_info = ttk.Label(barra_superior, text="")
    lbl_info.pack(side=tk.LEFT, padx=5)

    btn_actualizar = ttk.Button(barra_superior, text="Actualitzar")
    btn_actualizar.pack(side=tk.RIGHT, padx=5)

    tree = ttk.Treeview(frame_tabla, show="headings")

    scroll_y = ttk.Scrollbar(frame_tabla, orient=tk.VERTICAL, command=tree.yview)
    scroll_x = ttk.Scrollbar(frame_tabla, orient=tk.HORIZONTAL, command=tree.xview)

    tree.configure(
        yscrollcommand=scroll_y.set,
        xscrollcommand=scroll_x.set
    )

    tree.grid(row=0, column=0, sticky="nsew")
    scroll_y.grid(row=0, column=1, sticky="ns")
    scroll_x.grid(row=1, column=0, sticky="ew")

    frame_tabla.rowconfigure(0, weight=1)
    frame_tabla.columnconfigure(0, weight=1)

    def cargar_tabla():
        for item in tree.get_children():
            tree.delete(item)

        columnas, filas, error = leer_patrones_normales_bd()

        if error is not None:
            tree["columns"] = ["mensaje"]
            tree.heading("mensaje", text="Missatge")
            tree.column("mensaje", width=900, anchor="w")
            tree.insert("", tk.END, values=[error])
            lbl_info.config(text="Base de dades no disponible")
            return

        tree["columns"] = columnas

        for col in columnas:
            tree.heading(col, text=etiqueta_columna_bd(col))

            if col == "id":
                ancho = 60
            elif col == "eje":
                ancho = 60
            elif col.endswith("_min") or col.endswith("_max"):
                ancho = 150
            else:
                ancho = 140

            tree.column(col, width=ancho, anchor="center", stretch=False)

        for fila in filas:
            valores = [formatear_valor_bd(v) for v in fila]
            tree.insert("", tk.END, values=valores)

        lbl_info.config(
            text=f"Patrons normals desats: {len(filas)}"
        )

    btn_actualizar.config(command=cargar_tabla)
    cargar_tabla()


def _blob_a_array_seguro(blob):
    try:
        return blob_a_array_bd(blob)
    except Exception:
        return np.array([], dtype=float)


def leer_segmentos_bd_para_visualizacion():
    """
    Devuelve una lista con un elemento por grupo/patrón guardado en la BD
    (uno por fila de patrones_normales_dtw), incluyendo:
    - el representante del grup (arrays guardados en patrones_normales_dtw)
    - TODOS los segmentos individuales fusionados en ese grupo
      (arrays guardados en segmentos_patrones_dtw)
    Esto permite pintar, en una sola página por grupo, todos los segmentos
    superpuestos junto con el representativo, al estilo de Clasificación X/Y.
    """
    ruta_bd = RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES

    if not ruta_bd.exists():
        return [], f"Encara no existeix la base de dades:\n{ruta_bd}"

    conn = sqlite3.connect(ruta_bd)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                id,
                tipo_patron,
                eje,
                segmento_original_blob,
                segmento_filtrado_blob,
                forma_normalizada_blob,
                frecuencias_original_blob,
                espectro_original_blob,
                frecuencias_filtrado_blob,
                espectro_filtrado_blob
            FROM patrones_normales_dtw
            WHERE tipo_patron != 'senyal_completa'
            ORDER BY tipo_patron, eje, id
        """)
        filas_patrones = cur.fetchall()

        cur.execute("""
            SELECT
                patron_id,
                segmento_original_blob,
                segmento_filtrado_blob,
                forma_normalizada_blob,
                frecuencias_original_blob,
                espectro_original_blob,
                frecuencias_filtrado_blob,
                espectro_filtrado_blob
            FROM segmentos_patrones_dtw
            ORDER BY patron_id, id
        """)
        filas_segmentos = cur.fetchall()

        conn.close()

    except sqlite3.Error as e:
        conn.close()
        return [], f"Error en llegir els segments de la base de dades:\n{e}"

    segmentos_por_patron = {}

    for fila in filas_segmentos:
        (
            patron_id,
            segmento_original_blob,
            segmento_filtrado_blob,
            forma_normalizada_blob,
            frecuencias_original_blob,
            espectro_original_blob,
            frecuencias_filtrado_blob,
            espectro_filtrado_blob
        ) = fila

        segmentos_por_patron.setdefault(patron_id, []).append({
            "segmento_original": _blob_a_array_seguro(segmento_original_blob),
            "segmento_filtrado": _blob_a_array_seguro(segmento_filtrado_blob),
            "forma_normalizada": _blob_a_array_seguro(forma_normalizada_blob),
            "frecuencias_original": _blob_a_array_seguro(frecuencias_original_blob),
            "espectro_original": _blob_a_array_seguro(espectro_original_blob),
            "frecuencias_filtrado": _blob_a_array_seguro(frecuencias_filtrado_blob),
            "espectro_filtrado": _blob_a_array_seguro(espectro_filtrado_blob),
        })

    grupos_bd = []

    for fila in filas_patrones:
        (
            id_patron,
            tipo_patron,
            eje,
            segmento_original_blob,
            segmento_filtrado_blob,
            forma_normalizada_blob,
            frecuencias_original_blob,
            espectro_original_blob,
            frecuencias_filtrado_blob,
            espectro_filtrado_blob
        ) = fila

        representante = {
            "segmento_original": _blob_a_array_seguro(segmento_original_blob),
            "segmento_filtrado": _blob_a_array_seguro(segmento_filtrado_blob),
            "forma_normalizada": _blob_a_array_seguro(forma_normalizada_blob),
            "frecuencias_original": _blob_a_array_seguro(frecuencias_original_blob),
            "espectro_original": _blob_a_array_seguro(espectro_original_blob),
            "frecuencias_filtrado": _blob_a_array_seguro(frecuencias_filtrado_blob),
            "espectro_filtrado": _blob_a_array_seguro(espectro_filtrado_blob),
        }

        grupos_bd.append({
            "patron_id": id_patron,
            "tipo_patron": tipo_patron,
            "eje": eje,
            "representante": representante,
            "segmentos": segmentos_por_patron.get(id_patron, []),
        })

    return grupos_bd, None


def obtener_xmax_comun_desplazamiento_segmentos_bd():
    """
    Devuelve la duració máxima de todos los segmentos de desplazamiento
    guardados en la BD.

    Se usa para que, en la pestaña Segments BD, todos los grupos de
    desplazamiento tengan la misma escala horizontal en la gráfica:
    'Segment original desat (temps real relatiu)'.

    Para reposo y oscilatorio NO se aplica.
    """
    if not USAR_ESCALA_X_COMUN_DESPLAZAMIENTO_SEGMENTOS_BD:
        return None

    ruta_bd = RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES

    if not ruta_bd.exists():
        return None

    conn = sqlite3.connect(ruta_bd)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT MAX(duracion_s_max)
            FROM patrones_normales_dtw
            WHERE tipo_patron = 'desplazamiento'
        """)

        fila = cur.fetchone()
        conn.close()

    except sqlite3.Error:
        conn.close()
        return None

    if fila is None:
        return None

    duracion_max = fila[0]

    if duracion_max is None:
        return None

    duracion_max = float(duracion_max)

    if duracion_max <= 0:
        return None

    return duracion_max * MARGEN_ESCALA_X_DESPLAZAMIENTO_SEGMENTOS_BD


def crear_figura_grupo_segmentos_bd(grupo_bd):
    """
    Crea, para UN grupo/patrón de la BD, una figura con todos sus segmentos
    guardados superpuestos (línea fina y semitransparente) y el representante
    destacado con línea gruesa, igual que en las pestañas Clasificación X/Y.

    En la pestaña "Segments BD" no se muestra la gráfica independiente del
    segmento filtrado en tiempo normalizado. Se mantienen:
    1) segmento original,
    2) forma normalizada usada en DTW,
    3) espectro original y filtrado.
    """
    id_patron = grupo_bd["patron_id"]
    tipo_patron = grupo_bd["tipo_patron"]
    eje = grupo_bd["eje"]
    segmentos = grupo_bd["segmentos"]
    representante = grupo_bd["representante"]

    fig = Figure(figsize=(13, 8), dpi=100)

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    fig.suptitle(
        f"Grup BD ID {id_patron} | {etiqueta_tipus_patro(tipo_patron)} | eix {eje.upper()} | "
        f"{len(segmentos)} segment(s) desat(s)",
        fontsize=11
    )

    for seg in segmentos:
        señal = seg["segmento_original"]
        if len(señal) > 1:
            t_seg = np.arange(len(señal)) / Fs
            ax1.plot(t_seg, señal, linewidth=0.8, alpha=0.30, color="tab:blue")

    señal_rep = representante["segmento_original"]
    if len(señal_rep) > 1:
        t_rep = np.arange(len(señal_rep)) / Fs
        ax1.plot(t_rep, señal_rep, linewidth=2.2, color="black", label="Representatiu")

    ax1.set_title("Segment original desat (temps real relatiu)")
    ax1.set_xlabel("Temps relatiu [s]")
    ax1.set_ylabel("Acceleració")
    ax1.grid(True)

    if tipo_patron == "desplazamiento":
        xmax_comun = obtener_xmax_comun_desplazamiento_segmentos_bd()

        if xmax_comun is not None:
            ax1.set_xlim(0, xmax_comun)

    ax1.legend(fontsize=8, loc="upper right")

    x_forma = np.linspace(0, 100, LONGITUD_DTW)

    for seg in segmentos:
        forma = seg["forma_normalizada"]
        if len(forma) == LONGITUD_DTW:
            ax2.plot(x_forma, forma, linewidth=0.8, alpha=0.30, color="tab:blue")

    forma_rep = representante["forma_normalizada"]
    if len(forma_rep) == LONGITUD_DTW:
        ax2.plot(x_forma, forma_rep, linewidth=2.2, color="black", label="Representatiu")

    ax2.set_title("Forma normalitzada desada (utilitzada en DTW)")
    ax2.set_xlabel("Temps normalitzat [%]")
    ax2.set_ylabel("Amplitud z-normalitzada")
    ax2.grid(True)
    ax2.legend(fontsize=8, loc="upper right")

    for seg in segmentos:
        f_ori = seg["frecuencias_original"]
        e_ori = seg["espectro_original"]

        if len(f_ori) > 0 and len(e_ori) > 0:
            ax3.plot(f_ori, e_ori, linewidth=0.7, alpha=0.25, color="tab:blue")

        f_fil = seg["frecuencias_filtrado"]
        e_fil = seg["espectro_filtrado"]

        if len(f_fil) > 0 and len(e_fil) > 0:
            ax3.plot(f_fil, e_fil, linewidth=0.7, alpha=0.25, color="tab:orange")

    f_ori_rep = representante["frecuencias_original"]
    e_ori_rep = representante["espectro_original"]

    if len(f_ori_rep) > 0 and len(e_ori_rep) > 0:
        ax3.plot(f_ori_rep, e_ori_rep, linewidth=2.0, color="tab:blue", label="Original (representatiu)")

    f_fil_rep = representante["frecuencias_filtrado"]
    e_fil_rep = representante["espectro_filtrado"]

    if len(f_fil_rep) > 0 and len(e_fil_rep) > 0:
        ax3.plot(f_fil_rep, e_fil_rep, linewidth=2.0, color="tab:orange", label="Filtrada (representatiu)")

    ax3.set_title("Espectre desat")
    ax3.set_xlabel("Freqüència [Hz]")
    ax3.set_ylabel("Amplitud")
    ax3.grid(True)
    ax3.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


def insertar_segmentos_bd_paginados_en_frame(frame_padre):
    barra_superior = ttk.Frame(frame_padre)
    barra_superior.pack(fill=tk.X, padx=5, pady=5)

    frame_figura = ttk.Frame(frame_padre)
    frame_figura.pack(fill=tk.BOTH, expand=True)

    grupos_bd, error = leer_segmentos_bd_para_visualizacion()

    estado = {
        "pagina": 0,
        "n_paginas": max(1, len(grupos_bd))
    }

    btn_anterior = ttk.Button(barra_superior, text="⟨ Anterior")
    btn_siguiente = ttk.Button(barra_superior, text="Següent ⟩")
    lbl_pagina = ttk.Label(barra_superior, text="Pàgina 1/1")
    btn_actualizar = ttk.Button(barra_superior, text="Actualitzar")

    btn_anterior.pack(side=tk.LEFT, padx=5)
    lbl_pagina.pack(side=tk.LEFT, padx=10)
    btn_siguiente.pack(side=tk.LEFT, padx=5)
    btn_actualizar.pack(side=tk.RIGHT, padx=5)

    def limpiar_frame_figura():
        for widget in frame_figura.winfo_children():
            widget.destroy()

    def renderizar_pagina():
        limpiar_frame_figura()

        if error is not None:
            lbl = ttk.Label(
                frame_figura,
                text=error,
                anchor="center",
                justify="center"
            )
            lbl.pack(fill=tk.BOTH, expand=True)
            lbl_pagina.config(text="Base de dades no disponible")
            btn_anterior.config(state=tk.DISABLED)
            btn_siguiente.config(state=tk.DISABLED)
            return

        if len(grupos_bd) == 0:
            lbl = ttk.Label(
                frame_figura,
                text="No hi ha patrons desats en la base de dades.",
                anchor="center",
                justify="center"
            )
            lbl.pack(fill=tk.BOTH, expand=True)
            lbl_pagina.config(text="0 / 0")
            btn_anterior.config(state=tk.DISABLED)
            btn_siguiente.config(state=tk.DISABLED)
            return

        estado["n_paginas"] = len(grupos_bd)

        if estado["pagina"] < 0:
            estado["pagina"] = 0

        if estado["pagina"] >= estado["n_paginas"]:
            estado["pagina"] = estado["n_paginas"] - 1

        grupo_bd = grupos_bd[estado["pagina"]]

        fig = crear_figura_grupo_segmentos_bd(grupo_bd)

        canvas = FigureCanvasTkAgg(fig, master=frame_figura)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, frame_figura)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        lbl_pagina.config(
            text=f"Grup {estado['pagina'] + 1} / {estado['n_paginas']} "
                 f"| ID {grupo_bd['patron_id']} "
                 f"| {etiqueta_tipus_patro(grupo_bd['tipo_patron'])} | eix {grupo_bd['eje']} "
                 f"| {len(grupo_bd['segmentos'])} segment(s)"
        )

        btn_anterior.config(
            state=tk.NORMAL if estado["pagina"] > 0 else tk.DISABLED
        )

        btn_siguiente.config(
            state=tk.NORMAL if estado["pagina"] < estado["n_paginas"] - 1 else tk.DISABLED
        )

    def pagina_anterior():
        if estado["pagina"] > 0:
            estado["pagina"] -= 1
            renderizar_pagina()

    def pagina_siguiente():
        if estado["pagina"] < estado["n_paginas"] - 1:
            estado["pagina"] += 1
            renderizar_pagina()

    def actualizar():
        nuevos_grupos, nuevo_error = leer_segmentos_bd_para_visualizacion()

        grupos_bd.clear()
        grupos_bd.extend(nuevos_grupos)

        nonlocal error
        error = nuevo_error

        estado["pagina"] = 0
        estado["n_paginas"] = max(1, len(grupos_bd))

        renderizar_pagina()

    btn_anterior.config(command=pagina_anterior)
    btn_siguiente.config(command=pagina_siguiente)
    btn_actualizar.config(command=actualizar)

    renderizar_pagina()


def leer_bandas_senyal_completa_bd():
    """
    Lee de la BD las energías máximas por banda para el tipo senyal_completa.
    Devuelve un diccionario por eje: {'x': {...}, 'y': {...}, 'z': {...}}.
    """
    ruta_bd = RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES

    if not ruta_bd.exists():
        return {}, f"Encara no existeix la base de dades:\n{ruta_bd}"

    conn = sqlite3.connect(ruta_bd)
    cur = conn.cursor()

    try:
        cur.execute("PRAGMA table_info(patrones_normales_dtw)")
        columnas_existentes = [fila[1] for fila in cur.fetchall()]

        columnas_necesarias = ["eje"] + COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD
        columnas_disponibles = [col for col in columnas_necesarias if col in columnas_existentes]

        if "eje" not in columnas_disponibles:
            conn.close()
            return {}, "La taula no conté la columna 'eje'."

        columnas_sql = ", ".join(columnas_disponibles)

        cur.execute(
            f"""
            SELECT {columnas_sql}
            FROM patrones_normales_dtw
            WHERE tipo_patron = 'senyal_completa'
            ORDER BY eje ASC
            """
        )

        filas = cur.fetchall()
        conn.close()

    except sqlite3.Error as e:
        conn.close()
        return {}, f"Error en llegir les bandes de freqüència:\n{e}"

    datos = {"x": {}, "y": {}, "z": {}}

    for fila in filas:
        valores = dict(zip(columnas_disponibles, fila))
        eje = valores.get("eje")

        if eje not in datos:
            continue

        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            datos[eje][col] = valores.get(col, 0.0)

    return datos, None


def crear_figura_bandas_senyal_completa_bd(datos_bandas):
    """
    Crea una figura con tres gráficos de barras, uno por eje.
    Cada barra representa la energía máxima guardada en una banda de frecuencia.
    """
    fig = Figure(figsize=(13, 8), dpi=100)
    ejes = ["x", "y", "z"]

    fig.suptitle(
        "Bandes de freqüència generals del senyal complet",
        fontsize=12
    )

    valores_globales = []

    for eje in ejes:
        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            valor = datos_bandas.get(eje, {}).get(col, 0.0)
            if valor is not None:
                valores_globales.append(float(valor))

    ymax = max(valores_globales) * 1.10 if len(valores_globales) > 0 and max(valores_globales) > 0 else None

    for i, eje in enumerate(ejes, start=1):
        ax = fig.add_subplot(3, 1, i)

        valores = []

        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            valor = datos_bandas.get(eje, {}).get(col, 0.0)
            valores.append(0.0 if valor is None else float(valor))

        etiquetas_bandas = [
            ETIQUETAS_BANDAS_FRECUENCIA_BD.get(col, col)
            for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD
        ]
        x = np.arange(len(etiquetas_bandas))
        ax.bar(x, valores)

        ax.set_title(f"Eix {eje.upper()}")
        ax.set_ylabel("Energia")
        ax.grid(True, axis="y")

        ax.set_xticks(x)
        ax.set_xticklabels(
            etiquetas_bandas,
            rotation=30,
            ha="right",
            fontsize=8
        )

        if ymax is not None:
            ax.set_ylim(0, ymax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def crear_figura_energia_senyal_completa_vs_bd(info_identificacion, titulo=""):
    """
    Crea una figura con tres gráficos de barras, uno por eje X/Y/Z.

    En cada banda se comparan:
    - la energía de la señal completa del experimento actual;
    - la energía máxima guardada en la BD para el patrón senyal_completa.

    La comparación utiliza las columnas
    energia_banda_*_original_max de la base de datos actual.
    """
    fig = Figure(figsize=(13, 8), dpi=100)
    axs = fig.subplots(3, 1)

    subtitulo = (
        "Energia per bandes de la senyal completa actual "
        "enfront del màxim desat en la base de dades"
    )

    if titulo:
        fig.suptitle(f"{titulo}\n{subtitulo}", fontsize=11)
    else:
        fig.suptitle(subtitulo, fontsize=11)

    info_identificacion = info_identificacion or {}
    info_senyal = info_identificacion.get("senyal_completa") or {}
    info_por_eje = info_senyal.get("por_eje", {})

    grupos_senyal_completa = {
        grupo.get("eje"): grupo
        for grupo in info_identificacion.get("grupos_bd", [])
        if grupo.get("tipo_patron") == "senyal_completa"
    }

    etiquetas_bandas = [
        ETIQUETAS_BANDAS_FRECUENCIA_BD.get(col, col)
        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD
    ]

    x = np.arange(len(etiquetas_bandas), dtype=float)
    ancho_barra = 0.38

    for ax, eje in zip(axs, ["x", "y", "z"]):
        info_eje = info_por_eje.get(eje, {})
        energias_actuales = info_eje.get("features_frecuencia_bandas", {})

        grupo_bd = grupos_senyal_completa.get(eje)
        limites_bd = (
            grupo_bd.get("limites_frecuencia_bandas", {})
            if grupo_bd is not None
            else {}
        )

        valores_actuales = []
        valores_max_bd = []

        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            valor_actual = energias_actuales.get(col, 0.0)

            try:
                valor_actual = float(valor_actual)
                if not np.isfinite(valor_actual):
                    valor_actual = 0.0
            except Exception:
                valor_actual = 0.0

            limite = limites_bd.get(col, (None, None))

            if isinstance(limite, (tuple, list)) and len(limite) >= 2:
                valor_bd = limite[1]
            else:
                valor_bd = None

            try:
                valor_bd = float(valor_bd)
                if not np.isfinite(valor_bd):
                    valor_bd = 0.0
            except Exception:
                valor_bd = 0.0

            valores_actuales.append(valor_actual)
            valores_max_bd.append(valor_bd)

        ax.bar(
            x - ancho_barra / 2,
            valores_actuales,
            width=ancho_barra,
            label="Senyal completa actual"
        )
        ax.bar(
            x + ancho_barra / 2,
            valores_max_bd,
            width=ancho_barra,
            label="Màxim BD"
        )

        if grupo_bd is None:
            titulo_eje = f"Eix {eje.upper()} — sense patró senyal_completa en la BD"
        else:
            titulo_eje = f"Eix {eje.upper()}"

        ax.set_title(titulo_eje)
        ax.set_ylabel("Energia")
        ax.set_xticks(x)
        ax.set_xticklabels(
            etiquetas_bandas,
            rotation=30,
            ha="right",
            fontsize=8
        )
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(fontsize=8, loc="upper right")

    axs[-1].set_xlabel("Banda de freqüència")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def insertar_bandas_frecuencia_bd_en_frame(frame_padre):
    barra_superior = ttk.Frame(frame_padre)
    barra_superior.pack(fill=tk.X, padx=5, pady=5)

    frame_figura = ttk.Frame(frame_padre)
    frame_figura.pack(fill=tk.BOTH, expand=True)

    lbl_info = ttk.Label(barra_superior, text="")
    lbl_info.pack(side=tk.LEFT, padx=5)

    btn_actualizar = ttk.Button(barra_superior, text="Actualitzar")
    btn_actualizar.pack(side=tk.RIGHT, padx=5)

    def limpiar_frame():
        for widget in frame_figura.winfo_children():
            widget.destroy()

    def cargar_figura():
        limpiar_frame()

        datos_bandas, error = leer_bandas_senyal_completa_bd()

        if error is not None:
            lbl = ttk.Label(
                frame_figura,
                text=error,
                anchor="center",
                justify="center"
            )
            lbl.pack(fill=tk.BOTH, expand=True)
            lbl_info.config(text="Bandes no disponibles")
            return

        hay_datos = any(
            any(v is not None and float(v) != 0.0 for v in datos_bandas.get(eje, {}).values())
            for eje in ["x", "y", "z"]
        )

        if not hay_datos:
            lbl = ttk.Label(
                frame_figura,
                text="Encara no hi ha dades de senyal_completa desades.",
                anchor="center",
                justify="center"
            )
            lbl.pack(fill=tk.BOTH, expand=True)
            lbl_info.config(text="Sense dades de bandes")
            return

        fig = crear_figura_bandas_senyal_completa_bd(datos_bandas)

        canvas = FigureCanvasTkAgg(fig, master=frame_figura)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, frame_figura)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        lbl_info.config(text="Bandes de freqüència de senyal_completa")

    btn_actualizar.config(command=cargar_figura)
    cargar_figura()


def mostrar_ventana_bd_final(parent=None):
    """Abre una ventana independiente para consultar la BD sin analizar datos."""
    if parent is None:
        root = tk.Tk()
        usar_mainloop = True
    else:
        root = tk.Toplevel(parent)
        usar_mainloop = False

    root.title("Base de dades de patrons normals")
    root.geometry("1300x850")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    frame_bd = ttk.Frame(notebook)
    frame_segmentos_bd = ttk.Frame(notebook)
    frame_bandas_bd = ttk.Frame(notebook)

    notebook.add(frame_bd, text="Taula de dades")
    notebook.add(frame_segmentos_bd, text="Segments BD")
    notebook.add(frame_bandas_bd, text="Bandes de freqüència")

    insertar_bd_en_frame(frame_bd)
    insertar_segmentos_bd_paginados_en_frame(frame_segmentos_bd)
    insertar_bandas_frecuencia_bd_en_frame(frame_bandas_bd)

    if usar_mainloop:
        root.mainloop()


# ============================================================
# RESUMEN DE ANOMALÍAS — MODO USUARIO
# ============================================================

def formatear_valor_resumen(valor):
    """Formatea valores numéricos de la tabla de resumen."""
    if valor is None:
        return "—"

    try:
        valor = float(valor)
    except Exception:
        return "—"

    if not np.isfinite(valor):
        return "—"

    valor_abs = abs(valor)

    if valor_abs != 0.0 and (valor_abs < 1e-3 or valor_abs >= 1e4):
        return f"{valor:.3e}"

    return f"{valor:.5g}"


def texto_superacion_resumen(valor, limite, es_inferior=False):
    """Muestra el valor y cuánto ha rebasado el límite permitido."""
    valor = float(valor)
    limite = float(limite)

    porcentaje = (
        abs(valor - limite)
        / max(abs(limite), 1e-12)
        * 100.0
    )

    signo = "-" if es_inferior else "+"

    return (
        f"{formatear_valor_resumen(valor)} "
        f"({signo}{porcentaje:.0f} %)"
    )

def buscar_grupo_segmento_resumen(segmento, grupos_bd):
    """
    Busca el grupo realmente asignado al segmento.

    No se usa el grupo candidato de un desplazamiento desconocido, porque sus
    límites físicos y sus bandas no participaron en la decisión final.
    """
    grupo_id = segmento.get("grupo_id")

    if grupo_id is None:
        return None

    for grupo in grupos_bd:
        if grupo.get("id") == grupo_id:
            return grupo

    return None


def evaluar_feature_resumen(segmento, grupo_bd, nombre_feature):
    """
    Solo muestra la característica cuando está fuera de los límites.
    Si es correcta, la celda queda vacía.
    """
    if grupo_bd is None:
        return "—", False

    valores = segmento.get("valores", {})
    limites = grupo_bd.get("limites", {})

    if nombre_feature not in valores or nombre_feature not in limites:
        return "—", False

    valor = valores.get(nombre_feature)
    minimo, maximo = limites.get(nombre_feature, (None, None))

    try:
        valor = float(valor)
    except Exception:
        return "—", False

    if not np.isfinite(valor):
        return "—", False

    margen = float(MARGEN_LIMITES_IDENTIFICACION_BD)
    tolerancia = float(
        tolerancia_absoluta_limite_bd(nombre_feature)
    )

    if minimo is not None:
        try:
            minimo = float(minimo)
        except Exception:
            minimo = None

        if minimo is not None and np.isfinite(minimo):
            limite_inferior = (
                minimo
                - abs(minimo) * margen
                - tolerancia
            )

            if valor < limite_inferior:
                return (
                    texto_superacion_resumen(
                        valor,
                        limite_inferior,
                        es_inferior=True
                    ),
                    True
                )

    if maximo is not None:
        try:
            maximo = float(maximo)
        except Exception:
            maximo = None

        if maximo is not None and np.isfinite(maximo):
            limite_superior = (
                maximo
                + abs(maximo) * margen
                + tolerancia
            )

            if valor > limite_superior:
                return (
                    texto_superacion_resumen(
                        valor,
                        limite_superior
                    ),
                    True
                )

    return formatear_valor_resumen(valor), False


def evaluar_energia_resumen(segmento, grupo_bd, columna_banda):
    """
    Solo muestra la energía cuando supera el límite permitido.
    Las bandas correctas quedan vacías.
    """
    if grupo_bd is None:
        return "—", False

    valores = segmento.get(
        "features_frecuencia_bandas",
        {}
    )
    limites = grupo_bd.get(
        "limites_frecuencia_bandas",
        {}
    )

    if columna_banda not in valores or columna_banda not in limites:
        return "—", False

    valor = valores.get(columna_banda)
    _minimo, maximo = limites.get(
        columna_banda,
        (None, None)
    )

    if valor is None or maximo is None:
        return "—", False

    try:
        valor = float(valor)
        maximo = float(maximo)
    except Exception:
        return "—", False

    if not np.isfinite(valor) or not np.isfinite(maximo):
        return "—", False

    limite = (
        maximo
        + abs(maximo) * MARGEN_FRECUENCIA_BANDAS_BD
        + TOL_ABS_ENERGIA_BANDA_BD
    )

    if valor > limite:
        return texto_superacion_resumen(valor, limite), True

    return formatear_valor_resumen(valor), False


def construir_filas_resumen_anomalias_usuario(
        info_identificacion,
        solo_anomalos=True
):
    """
    Construye las filas del resumen.

    Mode usuari:
    - segmentos anómalos;
    - segmentos afectados por impacto;
    - ejes de la señal completa fuera de límites.

    Mode depuració:
    - todos los segmentos;
    - impactos sin segmento asociado;
    - los tres ejes de la señal completa.
    """
    info_identificacion = info_identificacion or {}

    grupos_bd = info_identificacion.get("grupos_bd", [])
    info_por_eje = info_identificacion.get("por_eje", {})
    impactos_globales = [
        (int(ini), int(fin))
        for ini, fin in info_identificacion.get("impactos", [])
    ]

    filas = []
    impactos_asociados = set()

    for eje in ("x", "y"):
        info_eje = info_por_eje.get(eje, {})

        for segmento in info_eje.get("segmentos", []):
            es_anomalo = bool(segmento.get("anomalo", False))
            tipo = str(segmento.get("tipo_patron", ""))
            ini = int(segmento.get("ini", 0))
            fin = int(segmento.get("fin", ini))

            impactos_segmento = [
                impacto
                for impacto in impactos_globales
                if segmentos_solapan((ini, fin), impacto)
            ]
            tiene_impacto = len(impactos_segmento) > 0
            impactos_asociados.update(impactos_segmento)

            if solo_anomalos and not es_anomalo and not tiene_impacto:
                continue

            grupo = buscar_grupo_segmento_resumen(segmento, grupos_bd)

            if es_anomalo and tiene_impacto:
                estado = "Anòmal + impacte"
            elif tiene_impacto:
                estado = "Impacte"
            elif es_anomalo:
                estado = "Anòmal"
            else:
                estado = "Correcte"

            if tiene_impacto:
                intervalos_impacto = ", ".join(
                    f"{ini_imp / Fs:.3f}–{fin_imp / Fs:.3f} s"
                    for ini_imp, fin_imp in impactos_segmento
                )
                texto_impacto = f"Sí: {intervalos_impacto}"
            else:
                texto_impacto = "No"

            fila = {
                "_orden": (0 if eje == "x" else 1, ini),
                "segmento": (str(segmento.get("etiqueta", "")), False),
                "eje": (eje.upper(), False),
                "tipo": (etiqueta_tipus_patro(tipo), False),
                "intervalo": (f"{ini / Fs:.3f}–{fin / Fs:.3f} s", False),
                "grupo": (
                    str(segmento.get("grupo_etiqueta", "Sense grup")),
                    False
                ),
                "impacto": (texto_impacto, False),
                "estado": (estado, False),
                "_es_anomalo": es_anomalo,
                "_tiene_impacto": tiene_impacto,
            }

            if grupo is not None:
                fila["rizo"] = evaluar_feature_resumen(
                    segmento,
                    grupo,
                    "rizado_pico_pico_original"
                )

                if tipo in ("oscilatorio", "reposo"):
                    fila["rms"] = evaluar_feature_resumen(
                        segmento,
                        grupo,
                        "rms_original"
                    )
                    fila["pico_pico"] = evaluar_feature_resumen(
                        segmento,
                        grupo,
                        "pico_pico_original"
                    )
                else:
                    fila["rms"] = ("—", False)
                    fila["pico_pico"] = ("—", False)

                if tipo == "reposo":
                    fila["offset"] = evaluar_feature_resumen(
                        segmento,
                        grupo,
                        "offset_abs_original"
                    )
                else:
                    fila["offset"] = ("—", False)

                for columna_banda in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
                    fila[columna_banda] = evaluar_energia_resumen(
                        segmento,
                        grupo,
                        columna_banda
                    )
            else:
                fila["rms"] = ("—", False)
                fila["pico_pico"] = ("—", False)
                fila["rizo"] = ("—", False)
                fila["offset"] = ("—", False)

                for columna_banda in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
                    fila[columna_banda] = ("—", False)

            filas.append(fila)

    for numero_impacto, impacto in enumerate(impactos_globales, start=1):
        if impacto in impactos_asociados:
            continue

        ini_imp, fin_imp = impacto
        texto_intervalo = f"{ini_imp / Fs:.3f}–{fin_imp / Fs:.3f} s"

        fila_impacto = {
            "_orden": (2, ini_imp),
            "segmento": (f"IMP{numero_impacto}", False),
            "eje": ("Global", False),
            "tipo": ("Impacte", False),
            "intervalo": (texto_intervalo, False),
            "grupo": ("—", False),
            "impacto": (f"Sí: {texto_intervalo}", False),
            "estado": ("Impacte sense segment associat", False),
            "rms": ("—", False),
            "pico_pico": ("—", False),
            "rizo": ("—", False),
            "offset": ("—", False),
            "_es_anomalo": False,
            "_tiene_impacto": True,
            "_impacto_sin_segmento": True,
        }

        for columna_banda in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            fila_impacto[columna_banda] = ("—", False)

        filas.append(fila_impacto)

    info_senyal_completa = info_identificacion.get("senyal_completa") or {}
    info_senyal_por_eje = info_senyal_completa.get("por_eje", {})
    orden_ejes_senyal = {"x": 0, "y": 1, "z": 2}

    for eje in ("x", "y", "z"):
        info_eje_senyal = info_senyal_por_eje.get(eje)

        if info_eje_senyal is None:
            continue

        es_anomalo_senyal = bool(info_eje_senyal.get("anomalo", False))

        if solo_anomalos and not es_anomalo_senyal:
            continue

        grupo_senyal = buscar_grupo_segmento_resumen(
            info_eje_senyal,
            grupos_bd
        )
        estado_senyal = (
            "Fora dels límits"
            if es_anomalo_senyal
            else "Correcta"
        )

        fila_senyal = {
            "_orden": (3, orden_ejes_senyal[eje]),
            "segmento": (f"SC-{eje.upper()}", False),
            "eje": (eje.upper(), False),
            "tipo": ("Senyal completa", False),
            "intervalo": ("Tota la senyal", False),
            "grupo": (
                str(info_eje_senyal.get("grupo_etiqueta", "Sense grup")),
                False
            ),
            "impacto": ("—", False),
            "estado": (estado_senyal, False),
            "rms": ("—", False),
            "pico_pico": ("—", False),
            "rizo": ("—", False),
            "offset": ("—", False),
            "_es_anomalo": es_anomalo_senyal,
            "_tiene_impacto": False,
            "_es_senyal_completa": True,
        }

        if grupo_senyal is not None:
            for columna_banda in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
                fila_senyal[columna_banda] = evaluar_energia_resumen(
                    info_eje_senyal,
                    grupo_senyal,
                    columna_banda
                )
        else:
            for columna_banda in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
                fila_senyal[columna_banda] = ("—", False)

        filas.append(fila_senyal)

    filas.sort(key=lambda fila: fila.get("_orden", (99, 0)))
    return filas


def insertar_resumen_anomalias_usuario_en_frame(
        frame_padre,
        info_identificacion,
        solo_anomalos=True,
        mostrar_estado=False
):
    """
    Muestra una tabla tipo Excel con los segmentos identificados.

    En modo usuario se muestran únicamente los anómalos. En modo depuración
    se muestran todos los segmentos, sus valores y el estado final.

    Permite:
    - redimensionar columnas arrastrando la cabecera;
    - ajustar una columna con doble clic;
    - seleccionar y copiar celdas;
    - desplazarse horizontal y verticalmente;
    - resaltar individualmente las celdas fuera de límite.
    """
    filas = construir_filas_resumen_anomalias_usuario(
        info_identificacion,
        solo_anomalos=solo_anomalos
    )

    impactos_globales = list(
        (info_identificacion or {}).get("impactos", [])
    )

    n_impactos_globales = len(impactos_globales)

    n_segmentos_tabla = sum(
        1
        for fila in filas
        if (
            not bool(fila.get("_es_senyal_completa", False))
            and not bool(fila.get("_impacto_sin_segmento", False))
        )
    )

    n_segmentos_anomalos = sum(
        1
        for fila in filas
        if (
                bool(fila.get("_es_anomalo", False))
                and not bool(
            fila.get("_es_senyal_completa", False)
        )
        )
    )

    n_segmentos_impactados = sum(
        1
        for fila in filas
        if (
            bool(fila.get("_tiene_impacto", False))
            and not bool(fila.get("_es_senyal_completa", False))
            and not bool(fila.get("_impacto_sin_segmento", False))
        )
    )

    n_senyal_completa_mostrados = sum(
        1
        for fila in filas
        if bool(
            fila.get("_es_senyal_completa", False)
        )
    )

    n_senyal_completa_anomalos = sum(
        1
        for fila in filas
        if (
                bool(fila.get("_es_senyal_completa", False))
                and bool(fila.get("_es_anomalo", False))
        )
    )

    if info_identificacion is None:
        ttk.Label(
            frame_padre,
            text="No hi ha informació d'identificació disponible.",
            anchor="center"
        ).pack(fill=tk.BOTH, expand=True)
        return

    if info_identificacion.get("error_bd") is not None:
        ttk.Label(
            frame_padre,
            text=info_identificacion.get("error_bd"),
            anchor="center",
            justify="center"
        ).pack(fill=tk.BOTH, expand=True)
        return

    if len(filas) == 0:
        if solo_anomalos:
            mensaje_vacio = (
                "No s'han detectat segments anòmals, impactes "
                "ni eixos de la senyal completa fora dels límits."
            )
        else:
            mensaje_vacio = (
                "No hi ha segments identificats "
                "en els eixos X i Y."
            )

        ttk.Label(
            frame_padre,
            text=mensaje_vacio,
            anchor="center",
            font=("TkDefaultFont", 12, "bold")
        ).pack(fill=tk.BOTH, expand=True)

        return

    if solo_anomalos:
        texto_cabecera = (
            f"Segments anòmals: {n_segmentos_anomalos}. "
            f"Segments afectats per impacte: "
            f"{n_segmentos_impactados}. "
            f"Impactes detectats: {n_impactos_globales}. "
            f"Eixos de la senyal completa fora dels límits: "
            f"{n_senyal_completa_anomalos}. "
            "Les cel·les grogues indiquen límits superats "
            "i les roges indiquen impactes."
        )
    else:
        texto_cabecera = (
            f"Segments identificats en X i Y: "
            f"{n_segmentos_tabla}. "
            f"Segments anòmals: {n_segmentos_anomalos}. "
            f"Segments afectats per impacte: "
            f"{n_segmentos_impactados}. "
            f"Impactes detectats: {n_impactos_globales}. "
            f"Senyal completa: {n_senyal_completa_mostrados} eixos, "
            f"{n_senyal_completa_anomalos} fora dels límits. "
            "Les cel·les grogues indiquen límits superats "
            "i les roges indiquen impactes."
        )

    ttk.Label(
        frame_padre,
        text=texto_cabecera,
        anchor="w"
    ).pack(
        fill=tk.X,
        padx=8,
        pady=(8, 4)
    )

    columnas = [
        ("segmento", "Segment", 85),
        ("eje", "Eix", 50),
        ("tipo", "Tipus", 115),
        ("intervalo", "Interval", 120),
        ("grupo", "Grup BD", 90),
        ("impacto", "Impacte", 190),
    ]

    if mostrar_estado:
        columnas.append(("estado", "Estat", 90))

    columnas.extend([
        ("rms", "RMS", 120),
        ("pico_pico", "Pic-pic", 120),
        ("rizo", "Ondulació pic pic", 135),
        ("offset", "Òfset abs.", 115),
    ])

    for columna_banda in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
        columnas.append((
            columna_banda,
            ETIQUETAS_BANDAS_FRECUENCIA_BD.get(
                columna_banda,
                columna_banda
            ),
            125
        ))

    cabeceras = [
        titulo
        for _clave, titulo, _ancho in columnas
    ]

    datos = []
    celdas_superadas = []

    for indice_fila, fila in enumerate(filas):
        valores_fila = []

        for indice_columna, (clave, _titulo, _ancho) in enumerate(columnas):
            texto, supera = fila.get(
                clave,
                ("—", False)
            )

            valores_fila.append(str(texto))

            if supera:
                celdas_superadas.append((
                    indice_fila,
                    indice_columna
                ))

        datos.append(valores_fila)

    sheet = Sheet(
        frame_padre,
        data=datos,
        headers=cabeceras,
        show_row_index=False,
        show_header=True,
        show_x_scrollbar=True,
        show_y_scrollbar=True,
        default_column_width=120,
        default_row_height=28,
        default_header_height=32,
        min_column_width=45,
        alternate_color="#F7F7F7",
        align="center",
        header_align="center",
        table_wrap="",
        header_wrap="",
        allow_cell_overflow=False,
        tooltips=True
    )

    sheet.pack(
        fill=tk.BOTH,
        expand=True,
        padx=6,
        pady=(0, 6)
    )

    sheet.enable_bindings(
        "single_select",
        "drag_select",
        "column_select",
        "row_select",
        "select_all",
        "copy",
        "arrowkeys",
        "column_width_resize",
        "double_click_column_resize",
        "right_click_popup_menu",
        "rc_select"
    )

    sheet.set_options(
        header_bg="#F0F0F0",
        header_fg="black",
        header_grid_fg="#C7C7C7",
        header_border_fg="#C7C7C7",
        table_bg="#FFFFFF",
        table_fg="black",
        table_grid_fg="#D6D6D6",
        table_selected_cells_border_fg="#4A7AA5",
        table_selected_box_cells_fg="#4A7AA5",
        display_selected_fg_over_highlights=False
    )

    for indice_columna, (_clave, _titulo, ancho) in enumerate(columnas):
        sheet.column_width(
            column=indice_columna,
            width=ancho,
            redraw=False
        )

    for indice_fila, indice_columna in celdas_superadas:
        sheet.highlight_cells(
            row=indice_fila,
            column=indice_columna,
            bg="#FFF2A8",
            fg="black"
        )

    if mostrar_estado:
        indice_estado = next(
            (
                i for i, (clave, _titulo, _ancho) in enumerate(columnas)
                if clave == "estado"
            ),
            None
        )

        if indice_estado is not None:
            for indice_fila, fila in enumerate(filas):
                tiene_impacto = bool(
                    fila.get("_tiene_impacto", False)
                )

                es_anomalo = bool(
                    fila.get("_es_anomalo", False)
                )

                if tiene_impacto:
                    sheet.highlight_cells(
                        row=indice_fila,
                        column=indice_estado,
                        bg="#F4CCCC",
                        fg="black"
                    )

                elif es_anomalo:
                    sheet.highlight_cells(
                        row=indice_fila,
                        column=indice_estado,
                        bg="#FFF2A8",
                        fg="black"
                    )
    indice_impacto = next(
        (
            indice
            for indice, (clave, _titulo, _ancho)
            in enumerate(columnas)
            if clave == "impacto"
        ),
        None
    )

    if indice_impacto is not None:
        for indice_fila, fila in enumerate(filas):
            if bool(fila.get("_tiene_impacto", False)):
                sheet.highlight_cells(
                    row=indice_fila,
                    column=indice_impacto,
                    bg="#F4CCCC",
                    fg="black"
                )

    sheet.refresh()


def mostrar_ventana(
        fig_ac,
        fig_debug_x,
        fig_debug_y,
        fig_frec,
        info_identificacion_bd=None,
        figuras_matriz_bd=None,
        titulo_clasificacion="",
        parent=None,
        modo_visualizacion="depuracion",
        fig_ac_usuario=None,
        fig_modulo=None
):
    """
    Muestra la ventana de resultados.

    Si parent es None, crea una ventana principal con tk.Tk().
    Si parent existe, crea una ventana secundaria con tk.Toplevel(parent),
    de forma que la ventana de selección permanece abierta y permite lanzar
    nuevos análisis.
    """
    if parent is None:
        root = tk.Tk()
        usar_mainloop = True
    else:
        root = tk.Toplevel(parent)
        usar_mainloop = False

    root.title("Identificació de segments amb BD")
    root.geometry("1300x850")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    modo_visualizacion = str(modo_visualizacion).strip().lower()

    if modo_visualizacion == "usuario":
        root.title("Identificació de segments amb BD — mode usuari")

        frame_ac_usuario = ttk.Frame(notebook)
        frame_frec_usuario = ttk.Frame(notebook)
        frame_energia_usuario = ttk.Frame(notebook)
        frame_resumen_usuario = ttk.Frame(notebook)

        notebook.add(frame_ac_usuario, text="Acceleració")
        notebook.add(frame_frec_usuario, text="Freqüències")
        notebook.add(frame_energia_usuario, text="Energia per bandes")
        notebook.add(frame_resumen_usuario, text="Resum")

        if fig_ac_usuario is not None:
            insertar_figura_en_frame(frame_ac_usuario, fig_ac_usuario)
        else:
            ttk.Label(
                frame_ac_usuario,
                text="No s'ha pogut generar la vista d'usuari.",
                anchor="center"
            ).pack(fill=tk.BOTH, expand=True)

        if fig_frec is not None:
            insertar_figura_en_frame(frame_frec_usuario, fig_frec)
        else:
            ttk.Label(
                frame_frec_usuario,
                text="No s'ha pogut generar l'espectre en freqüència.",
                anchor="center"
            ).pack(fill=tk.BOTH, expand=True)

        if info_identificacion_bd is not None:
            fig_energia_usuario = crear_figura_energia_senyal_completa_vs_bd(
                info_identificacion=info_identificacion_bd,
                titulo=titulo_clasificacion
            )
            insertar_figura_en_frame(frame_energia_usuario, fig_energia_usuario)
        else:
            ttk.Label(
                frame_energia_usuario,
                text="No hi ha informació de la base de dades per a mostrar l’energia per bandes.",
                anchor="center",
                justify="center"
            ).pack(fill=tk.BOTH, expand=True)

        insertar_resumen_anomalias_usuario_en_frame(
            frame_padre=frame_resumen_usuario,
            info_identificacion=info_identificacion_bd
        )

        if usar_mainloop:
            root.mainloop()

        return


    root.title("Identificació de segments amb BD — mode depuració")


    frame_ac_tiempo = ttk.Frame(notebook)
    frame_modulo = ttk.Frame(notebook)

    frame_segmentacion_ac = ttk.Frame(notebook)
    frame_debug_x = ttk.Frame(notebook)
    frame_debug_y = ttk.Frame(notebook)

    frame_frec = ttk.Frame(notebook)
    frame_resumen_depuracion = ttk.Frame(notebook)


    notebook.add(
        frame_ac_tiempo,
        text="Acceleració en el temps"
    )

    if fig_ac_usuario is not None:
        insertar_figura_en_frame(
            frame_ac_tiempo,
            fig_ac_usuario
        )
    else:
        ttk.Label(
            frame_ac_tiempo,
            text="No s'ha pogut generar la gràfica d'acceleració.",
            anchor="center"
        ).pack(
            fill=tk.BOTH,
            expand=True
        )


    notebook.add(
        frame_modulo,
        text="Mòdul de l’acceleració"
    )

    if fig_modulo is not None:
        insertar_figura_en_frame(
            frame_modulo,
            fig_modulo
        )
    else:
        ttk.Label(
            frame_modulo,
            text="No s'ha pogut generar el mòdul de l'acceleració.",
            anchor="center"
        ).pack(
            fill=tk.BOTH,
            expand=True
        )


    notebook.add(
        frame_segmentacion_ac,
        text="Segmentació de l’acceleració"
    )

    insertar_figura_en_frame(
        frame_segmentacion_ac,
        fig_ac
    )


    notebook.add(
        frame_debug_x,
        text="Segmentació de depuració X"
    )

    insertar_figura_en_frame(
        frame_debug_x,
        fig_debug_x
    )


    notebook.add(
        frame_debug_y,
        text="Segmentació de depuració Y"
    )

    insertar_figura_en_frame(
        frame_debug_y,
        fig_debug_y
    )


    notebook.add(
        frame_frec,
        text="Freqüència"
    )

    insertar_figura_en_frame(
        frame_frec,
        fig_frec
    )


    if info_identificacion_bd is not None:


        frame_energia_bandas = ttk.Frame(notebook)

        notebook.add(
            frame_energia_bandas,
            text="Energia de la senyal completa"
        )

        fig_energia_bandas = crear_figura_energia_senyal_completa_vs_bd(
            info_identificacion=info_identificacion_bd,
            titulo=titulo_clasificacion
        )

        insertar_figura_en_frame(
            frame_energia_bandas,
            fig_energia_bandas
        )


        for eje in ("x", "y"):

            if figuras_matriz_bd is None:
                continue

            fig_matriz = figuras_matriz_bd.get(eje)

            if fig_matriz is None:
                continue

            frame_matriz = ttk.Frame(notebook)

            notebook.add(
                frame_matriz,
                text=f"Matriu BD {eje.upper()}"
            )

            insertar_figura_en_frame(
                frame_matriz,
                fig_matriz
            )


        frame_segmentos_bd = ttk.Frame(notebook)

        notebook.add(
            frame_segmentos_bd,
            text="Segments DB"
        )

        insertar_segmentos_bd_identificacion_en_frame(
            frame_padre=frame_segmentos_bd,
            info_identificacion=info_identificacion_bd,
            titulo=titulo_clasificacion
        )


    notebook.add(
        frame_resumen_depuracion,
        text="Resum de segments"
    )

    insertar_resumen_anomalias_usuario_en_frame(
        frame_padre=frame_resumen_depuracion,
        info_identificacion=info_identificacion_bd,
        solo_anomalos=False,
        mostrar_estado=True
    )

    if usar_mainloop:
        root.mainloop()


def pedir_configuracion_inicial():
    """
    Ventana inicial para elegir experimento, carpeta interna y bloque.

    Sustituye la selección por consola. Devuelve un diccionario con:
    - experimento
    - carpeta
    - cargar_todos_los_bloques
    - bloque
    """
    seleccion = {}

    root = tk.Tk()
    root.title("Identificació de possibles comportaments anòmals")
    root.geometry("620x415")
    root.resizable(False, False)

    exp_var = tk.StringVar(value=str(experimento))
    bloque_var = tk.StringVar(value=str(bloque))
    todos_var = tk.BooleanVar(value=CARGAR_TODOS_LOS_BLOQUES)
    carpeta_var = tk.StringVar()
    estado_var = tk.StringVar(value="")
    modo_var = tk.StringVar(value="Mode depuració")

    carpetas_actuales = []
    ruta_base_actual = None

    main_frame = ttk.Frame(root, padding=18)
    main_frame.pack(fill=tk.BOTH, expand=True)

    titulo = ttk.Label(
        main_frame,
        text="Selecció de l'experiència a analitzar",
        font=("Segoe UI", 13, "bold")
    )
    titulo.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

    ttk.Label(main_frame, text="Experiència:").grid(row=1, column=0, sticky="w", pady=6)
    entry_exp = ttk.Entry(main_frame, textvariable=exp_var, width=14)
    entry_exp.grid(row=1, column=1, sticky="w", pady=6)

    btn_actualizar = ttk.Button(main_frame, text="Actualitzar carpetes")
    btn_actualizar.grid(row=1, column=2, sticky="w", padx=(8, 0), pady=6)

    ttk.Label(main_frame, text="Carpeta/programa:").grid(row=2, column=0, sticky="w", pady=6)
    combo_carpeta = ttk.Combobox(
        main_frame,
        textvariable=carpeta_var,
        state="readonly",
        width=48
    )
    combo_carpeta.grid(row=2, column=1, columnspan=2, sticky="w", pady=6)

    ttk.Label(main_frame, text="Mode de visualització:").grid(
        row=3,
        column=0,
        sticky="w",
        pady=(12, 6)
    )
    combo_modo = ttk.Combobox(
        main_frame,
        textvariable=modo_var,
        state="readonly",
        width=24,
        values=("Mode depuració", "Mode usuari")
    )
    combo_modo.grid(row=3, column=1, columnspan=2, sticky="w", pady=(12, 6))
    combo_modo.current(1)

    check_todos = ttk.Checkbutton(
        main_frame,
        text="Carregar tots els blocs",
        variable=todos_var
    )
    check_todos.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 6))

    ttk.Label(main_frame, text="Bloc:").grid(row=5, column=0, sticky="w", pady=6)
    entry_bloque = ttk.Entry(main_frame, textvariable=bloque_var, width=14)
    entry_bloque.grid(row=5, column=1, sticky="w", pady=6)

    label_estado = ttk.Label(main_frame, textvariable=estado_var, foreground="gray")
    label_estado.grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 6))

    botones = ttk.Frame(main_frame)
    botones.grid(row=7, column=0, columnspan=3, sticky="e", pady=(18, 0))

    btn_cancelar = ttk.Button(botones, text="Cancel·lar")
    btn_cancelar.pack(side=tk.RIGHT, padx=(8, 0))

    btn_aceptar = ttk.Button(botones, text="Acceptar")
    btn_aceptar.pack(side=tk.RIGHT, padx=(8, 0))

    btn_ver_bd = ttk.Button(botones, text="Veure la base de dades")
    btn_ver_bd.pack(side=tk.RIGHT)

    def actualizar_estado_bloque(*_):
        if todos_var.get():
            entry_bloque.configure(state="disabled")
        else:
            entry_bloque.configure(state="normal")

    def cargar_carpetas():
        nonlocal carpetas_actuales, ruta_base_actual

        try:
            num_exp = int(exp_var.get().strip())
        except ValueError:
            messagebox.showerror("Experiència no vàlida", "Introdueix un número d’experiència vàlida.")
            return

        try:
            ruta_base_actual = obtener_ruta_base(num_exp)

            if not ruta_base_actual.exists():
                raise FileNotFoundError(f"No existeix la ruta base:\n{ruta_base_actual}")

            carpetas_actuales = obtener_carpetas_con_timeblocks(ruta_base_actual)

            if len(carpetas_actuales) == 0:
                raise FileNotFoundError(
                    f"No s'han trobat fitxers timeblock*.txt en:\n{ruta_base_actual}"
                )

            nombres = []
            for i, carpeta in enumerate(carpetas_actuales):
                try:
                    nombre = str(carpeta.relative_to(ruta_base_actual))
                except ValueError:
                    nombre = carpeta.name
                nombres.append(f"{i}: {nombre}")

            combo_carpeta["values"] = nombres
            combo_carpeta.current(0)

            estado_var.set(
                f"Experiència {num_exp}: {len(carpetas_actuales)} carpeta/es trobada/es."
            )

        except Exception as e:
            carpetas_actuales = []
            combo_carpeta["values"] = []
            carpeta_var.set("")
            estado_var.set("Error al carregar carpetes.")
            messagebox.showerror("Error", str(e))

    def ver_base_datos():
        """Abre el visor de la BD sin ejecutar ni guardar ningún análisis."""
        mostrar_ventana_bd_final(parent=root)

    def aceptar():
        if len(carpetas_actuales) == 0:
            messagebox.showerror(
                "Sense carpeta seleccionada",
                "Primer carrega les carpetes de l’experiment."
            )
            return

        try:
            num_exp = int(exp_var.get().strip())
        except ValueError:
            messagebox.showerror("Experiència no vàlida", "Introdueix un número d’experiència vàlida.")
            return

        idx = combo_carpeta.current()
        if idx < 0 or idx >= len(carpetas_actuales):
            messagebox.showerror("Carpeta no vàlida", "Selecciona una carpeta/programa.")
            return

        cargar_todos = bool(todos_var.get())

        try:
            num_bloque = int(bloque_var.get().strip())
        except ValueError:
            messagebox.showerror("Bloc no vàlid", "Introdueix un número de bloc vàlid.")
            return

        if num_bloque < 0:
            messagebox.showerror("Bloc no vàlid", "El bloc no pot ser negatiu.")
            return

        modo_seleccionado = (
            "usuario"
            if modo_var.get().strip().lower() == "mode usuari"
            else "depuracion"
        )

        seleccion_actual = {
            "experimento": num_exp,
            "carpeta": carpetas_actuales[idx],
            "cargar_todos_los_bloques": cargar_todos,
            "bloque": num_bloque,
            "modo_visualizacion": modo_seleccionado,
        }

        estado_var.set("Processant l'experiment...")
        btn_aceptar.configure(state="disabled")
        btn_actualizar.configure(state="disabled")
        root.update_idletasks()

        try:
            ejecutar_analisis_desde_seleccion(
                seleccion=seleccion_actual,
                parent=root
            )

            estado_var.set(
                "Anàlisi oberta. Pots seleccionar un altre experiment, carpeta o bloc."
            )

        except Exception as e:
            estado_var.set("Error durant l'anàlisi.")
            messagebox.showerror("Error durant l'anàlisi", str(e), parent=root)

        finally:
            btn_aceptar.configure(state="normal")
            btn_actualizar.configure(state="normal")

    def cancelar():
        seleccion.clear()
        root.destroy()

    btn_actualizar.configure(command=cargar_carpetas)
    btn_ver_bd.configure(command=ver_base_datos)
    btn_aceptar.configure(command=aceptar)
    btn_cancelar.configure(command=cancelar)
    check_todos.configure(command=actualizar_estado_bloque)
    root.protocol("WM_DELETE_WINDOW", cancelar)

    actualizar_estado_bloque()
    cargar_carpetas()

    entry_exp.focus_set()
    root.mainloop()


##################################################
# MAIN
##################################################


def calcular_rizo_pico_pico_1d_identificacion(seg_original, seg_filtrada):
    """
    Calcula el rizo pico-pico con el mismo criterio usado al crear la BD:

        rizo = segmento_original - segmento_filtrado

    La señal filtrada solo define la componente lenta del movimiento.
    La señal real se usa para medir el rizo/vibración superpuesta.
    """
    seg_original = np.asarray(seg_original, dtype=float)
    seg_filtrada = np.asarray(seg_filtrada, dtype=float)

    n = min(len(seg_original), len(seg_filtrada))

    if n == 0:
        return 0.0

    rizo = seg_original[:n] - seg_filtrada[:n]
    return float(np.max(rizo) - np.min(rizo))


def crear_segmento_actual_identificacion(
        acc_sin_filtrar,
        acc_filtrada,
        eje,
        tipo_patron,
        segmento,
        etiqueta
):
    """
    Crea la información necesaria para identificar un segmento.

    La señal filtrada se usa para calcular la forma y las métricas de
    clasificación. La señal real sin filtrar se usa para validar límites,
    calcular el rizo y analizar la energia per bandes.
    """
    ini, fin = segmento
    idx = IDX_EJE[eje]

    seg_original = np.asarray(
        acc_sin_filtrar[ini:fin + 1, idx],
        dtype=float
    )
    seg_filtrado = np.asarray(
        acc_filtrada[ini:fin + 1, idx],
        dtype=float
    )

    features_original = calcular_features_patron_1d_bd(seg_original)

    valores = {
        "duracion_s": float(len(seg_original) / Fs),
        "rms_original": features_original["rms"],
        "pico_pico_original": features_original["pico_pico"],
        "rizado_pico_pico_original": calcular_rizo_pico_pico_1d_identificacion(
            seg_original,
            seg_filtrado
        ),
        "offset_abs_original": features_original["offset_abs"],
    }

    forma = calcular_forma_normalizada_identificacion(seg_filtrado)
    metrica_dtw = metrica_dtw_desde_senal_segmento(seg_filtrado)

    espectro_comparacion = calcular_espectro_comparacion_identificacion(
        seg_original
    )
    features_frecuencia_bandas = calcular_features_frecuencia_bandas_bd(
        seg_original
    )

    return {
        "etiqueta": etiqueta,
        "eje": eje,
        "tipo_patron": tipo_patron,
        "ini": int(ini),
        "fin": int(fin),
        "segmento_original": seg_original,
        "forma": forma,
        "metrica_dtw": metrica_dtw,
        "espectro_comparacion": espectro_comparacion,
        "features_frecuencia_bandas": features_frecuencia_bandas,
        "valores": valores,
        "grupo_id": None,
        "grupo_etiqueta": "Sense grup",
        "distancia_dtw": None,
        "anomalo": False,
        "motivos_anomalia": [],
    }


# ============================================================
# LECTURA DE LA BASE DE DATOS Y COMPARACIONES ACTUALES
# ============================================================
# La base de datos actual no contiene métricas filtradas ni frecuencia dominante.
# Por tanto, la identificación NO debe leer columnas como rms_filtrada_min,
# pico_pico_filtrada_min, freq_dom_filtrada_min, etc.
#
# Características utilizadas:
# - desplazamiento: forma/DTW y duració solo para clasificar el grupo;
#   rizo pico-pico y energia per bandes para validar anomalías.
# - oscilatorio: RMS original, pico-pico original, rizo pico-pico y energia per bandes.
# - reposo: RMS original, pico-pico original, rizo pico-pico, offset absoluto original
#   y energia per bandes.
# - senyal_completa: solo energia per bandes.

BANDAS_FRECUENCIA_BD = [
    (0, 30),
    (30, 80),
    (80, 120),
    (120, 180),
    (180, 220),
    (220, 320),
    (320, 500),
]

COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD = [
    f"energia_banda_{f_ini}_{f_fin}_original_max"
    for f_ini, f_fin in BANDAS_FRECUENCIA_BD
]

ETIQUETAS_BANDAS_FRECUENCIA_BD = {
    f"energia_banda_{f_ini}_{f_fin}_original_max": f"{f_ini}-{f_fin} Hz"
    for f_ini, f_fin in BANDAS_FRECUENCIA_BD
}

# Para desplazamientos, la duració y la forma/DTW se usan
# exclusivamente para decidir si el segmento pertenece a un grupo conocido.
# No se usan para decidir si el comportamiento es anómalo.
# Una vez asignado el grupo, la anomalía se valida con el rizo pico-pico
# de la señal real y con la energia per bandes.
FEATURES_DESPLAZAMIENTO_IDENTIFICACION_BD = [
    "rizado_pico_pico_original",
]

FEATURES_OSCILATORIO_IDENTIFICACION_BD = [
    "rms_original",
    "pico_pico_original",
    "rizado_pico_pico_original",
]

FEATURES_REPOSO_IDENTIFICACION_BD = [
    "rms_original",
    "pico_pico_original",
    "rizado_pico_pico_original",
    "offset_abs_original",
]


# Tolerancia absoluta pequeña para la energia per bandes. El criterio principal
# sigue siendo el margen relativo MARGEN_FRECUENCIA_BANDAS_BD.
TOL_ABS_ENERGIA_BANDA_BD = 1e-12
TOL_ABS_DTW_DISTANCIA_BD = 1e-6


def valor_fila_bd(fila_dict, nombre, defecto=None):
    return fila_dict.get(nombre, defecto)


def es_numero_valido(x):
    if x is None:
        return False

    try:
        x = float(x)
    except Exception:
        return False

    return np.isfinite(x)


def calcular_energia_bandas_desde_espectro_identificacion(f, espectro):
    """
    Calcula la energia per bandes con el mismo criterio usado al crear la BD:
    suma de espectro**2 dentro de cada banda.
    """
    f = np.asarray(f, dtype=float)
    espectro = np.asarray(espectro, dtype=float)

    energias = {}

    for f_ini, f_fin in BANDAS_FRECUENCIA_BD:
        col = f"energia_banda_{f_ini}_{f_fin}_original_max"

        if len(f) == 0 or len(espectro) == 0:
            energias[col] = 0.0
            continue

        mascara = (f >= f_ini) & (f < f_fin)

        if f_fin == BANDAS_FRECUENCIA_BD[-1][1]:
            mascara = (f >= f_ini) & (f <= f_fin)

        if not np.any(mascara):
            energias[col] = 0.0
        else:
            energias[col] = float(np.sum(espectro[mascara] ** 2))

    return energias


def calcular_features_frecuencia_bandas_bd(senal):
    """
    Solo se calculan las energías absolutas por banda guardadas en la BD.
    """
    f, espectro = calcular_espectro_segmento_1d_bd(senal)
    return calcular_energia_bandas_desde_espectro_identificacion(f, espectro)


def tolerancia_absoluta_limite_bd(nombre):
    if nombre == "duracion_s":
        return TOL_ABS_DURACION_LIMITES_BD

    if nombre == "rizado_pico_pico_original":
        return TOL_ABS_RIZADO_PICO_PICO_BD

    if nombre == "dtw_distancia":
        return TOL_ABS_DTW_DISTANCIA_BD

    return TOL_ABS_ACEL_LIMITES_BD


def valor_fuera_rango_bd(valor, minimo, maximo, margen, tolerancia_abs=0.0):
    """
    Compara contra min/max si existen.
    - Si solo existe máximo, compara únicamente por exceso.
    - Si existen mínimo y máximo, comprueba que el valor cae dentro del rango
      con margen relativo y tolerancia absoluta.
    """
    if valor is None:
        return False, None

    try:
        valor = float(valor)
    except Exception:
        return False, None

    if not np.isfinite(valor):
        return False, None

    tolerancia_abs = float(tolerancia_abs)

    if minimo is not None:
        try:
            minimo = float(minimo)
        except Exception:
            minimo = None

        if minimo is not None and np.isfinite(minimo):
            limite_inf = minimo - abs(minimo) * margen - tolerancia_abs

            if valor < limite_inf:
                detalle = (
                    f"{valor:.5g} < mín {minimo:.5g} "
                    f"(-{margen * 100:.1f} %, tol {tolerancia_abs:.3g})"
                )
                return True, detalle

    if maximo is not None:
        try:
            maximo = float(maximo)
        except Exception:
            maximo = None

        if maximo is not None and np.isfinite(maximo):
            limite_sup = maximo + abs(maximo) * margen + tolerancia_abs

            if valor > limite_sup:
                detalle = (
                    f"{valor:.5g} > máx {maximo:.5g} "
                    f"(+{margen * 100:.1f} %, tol {tolerancia_abs:.3g})"
                )
                return True, detalle

    return False, None


def distancia_clasificacion_grupo_desplazamiento(
        segmento,
        grupo_bd,
        distancia_forma
):
    """
    Calcula la distància usada SOLO para clasificar un desplazamiento.

    - La forma/DTW se calcula previamente con la señal filtrada.
    - La duració se compara con el intervalo normal del grup guardado en BD.
    - Ninguna de las dos magnitudes se usa después como característica de
      comportamiento anómalo.

    Si la BD no contiene límites de duració para el grupo, la duració no
    impide la asignación y su contribución a la distància es cero.
    """
    valores = segmento.get("valores", {})
    limites = grupo_bd.get("limites", {})

    duracion = valores.get("duracion_s")
    dur_min, dur_max = limites.get("duracion_s", (None, None))

    duracion_valida = es_numero_valido(duracion)
    min_valido = es_numero_valido(dur_min)
    max_valido = es_numero_valido(dur_max)

    if not duracion_valida or (not min_valido and not max_valido):
        info = {
            "compatible": True,
            "distancia": 0.0,
            "motivo": None,
            "limite_inferior": None,
            "limite_superior": None,
        }
        return float(distancia_forma), info

    duracion = float(duracion)
    margen = float(MARGEN_LIMITES_IDENTIFICACION_BD)
    tolerancia = float(TOL_ABS_DURACION_LIMITES_BD)

    limite_inferior = None
    limite_superior = None

    if min_valido:
        dur_min = float(dur_min)
        limite_inferior = dur_min - abs(dur_min) * margen - tolerancia

    if max_valido:
        dur_max = float(dur_max)
        limite_superior = dur_max + abs(dur_max) * margen + tolerancia

    compatible = True
    exceso_relativo = 0.0
    motivo = None

    if limite_inferior is not None and duracion < limite_inferior:
        compatible = False
        exceso_relativo = (
            (limite_inferior - duracion)
            / max(abs(limite_inferior), abs(duracion), 1e-12)
        )
        motivo = (
            f"duració {duracion:.4f} s inferior a l’interval compatible "
            f"del grup {grupo_bd.get('etiqueta', grupo_bd.get('id', '?'))}: "
            f"mínim permés {limite_inferior:.4f} s"
        )

    elif limite_superior is not None and duracion > limite_superior:
        compatible = False
        exceso_relativo = (
            (duracion - limite_superior)
            / max(abs(limite_superior), abs(duracion), 1e-12)
        )
        motivo = (
            f"duració {duracion:.4f} s superior a l’interval compatible "
            f"del grup {grupo_bd.get('etiqueta', grupo_bd.get('id', '?'))}: "
            f"màxim permés {limite_superior:.4f} s"
        )

    distancia_total = float(distancia_forma)

    info = {
        "compatible": compatible,
        "distancia": float(exceso_relativo),
        "motivo": motivo,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
    }

    return distancia_total, info


def comprobar_features_limites_superiores_con_grupo(segmento, grupo_bd, features, margen):
    """
    Comprueba los límites disponibles en la base de datos.
    Aunque conserva el nombre antiguo, ahora usa min/max cuando existen.
    """
    motivos = []
    valores = segmento.get("valores", {})
    limites = grupo_bd.get("limites", {})

    for nombre in features:
        if nombre not in valores or nombre not in limites:
            continue

        minimo, maximo = limites[nombre]
        tolerancia = tolerancia_absoluta_limite_bd(nombre)

        fuera, detalle = valor_fuera_rango_bd(
            valor=valores[nombre],
            minimo=minimo,
            maximo=maximo,
            margen=margen,
            tolerancia_abs=tolerancia
        )

        if fuera:
            motivos.append(f"{etiqueta_caracteristica_bd(nombre)}: {detalle}")

    return motivos


def comparar_energias_bandas_con_limites(features_actuales, limites_bandas, margen, prefijo="energia per bandes"):
    detalles = []

    for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
        if col not in features_actuales or col not in limites_bandas:
            continue

        valor = features_actuales.get(col)
        _minimo, maximo = limites_bandas.get(col, (None, None))

        if valor is None or maximo is None:
            continue

        try:
            valor = float(valor)
            maximo = float(maximo)
        except Exception:
            continue

        if not np.isfinite(valor) or not np.isfinite(maximo):
            continue

        limite = maximo + abs(maximo) * margen + TOL_ABS_ENERGIA_BANDA_BD

        if valor > limite:
            etiqueta = ETIQUETAS_BANDAS_FRECUENCIA_BD.get(col, col)
            detalles.append(
                f"{etiqueta}: {valor:.5g} > {limite:.5g} "
                f"(màx. BD {maximo:.5g}, +{margen * 100:.0f} %)"
            )

    if len(detalles) == 0:
        return []

    texto = f"{prefijo} fora dels límits: " + " | ".join(detalles[:7])

    if len(detalles) > 7:
        texto += f" | ... y {len(detalles) - 7} banda(es) més"

    return [texto]


def comprobar_frecuencia_bandas_con_grupo(segmento, grupo_bd):
    """
    Compara energías absolutas por bandas contra los máximos guardados
    en la BD, usando las columnas energia_banda_..._original_max.
    """
    if not USAR_LIMITES_FRECUENCIA_BANDAS_BD:
        return []

    features_seg = segmento.get("features_frecuencia_bandas", {})
    limites = grupo_bd.get("limites_frecuencia_bandas", {})

    return comparar_energias_bandas_con_limites(
        features_actuales=features_seg,
        limites_bandas=limites,
        margen=MARGEN_FRECUENCIA_BANDAS_BD,
        prefijo="energia per bandes"
    )


def leer_grupos_bd_identificacion():
    """
    Lee la base de datos V7.

    Además del representante de cada patrón, carga todos los segmentos
    individuales de ``segmentos_patrones_dtw``. Estos miembros se usan solo
    para asignar los desplazamientos al grupo más próximo mediante la misma
    distància combinada del entrenamiento.
    """
    ruta_bd = RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES
    print(f"Base de dades utilitzada: {ruta_bd.resolve()}")

    if not ruta_bd.exists():
        return [], f"No existeix la base de dades:\n{ruta_bd}"

    conn = sqlite3.connect(ruta_bd)
    cur = conn.cursor()

    try:
        cur.execute("PRAGMA table_info(patrones_normales_dtw)")
        columnas = {fila[1] for fila in cur.fetchall()}

        columnas_base = [
            "id",
            "tipo_patron",
            "eje",
            "duracion_s_min",
            "duracion_s_max",
            "rms_original_max",
            "pico_pico_original_max",
            "rizo_pico_pico_original_max",
            "offset_abs_original_max",
            "forma_normalizada_blob",
            "segmento_original_blob",
            "segmento_filtrado_blob",
        ] + COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD

        columnas_select = [c for c in columnas_base if c in columnas]

        for obligatoria in ["id", "tipo_patron", "eje"]:
            if obligatoria not in columnas_select:
                conn.close()
                return [], (
                    "La taula patrones_normales_dtw no conté la columna "
                    f"obligatòria '{obligatoria}'."
                )

        cur.execute(
            f"""
            SELECT {', '.join(columnas_select)}
            FROM patrones_normales_dtw
            ORDER BY tipo_patron, eje, id
            """
        )
        filas = cur.fetchall()

        miembros_por_patron = {}
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'segmentos_patrones_dtw'
            """
        )
        existe_tabla_segmentos = cur.fetchone() is not None

        if existe_tabla_segmentos:
            cur.execute(
                """
                SELECT
                    patron_id,
                    segmento_original_blob,
                    segmento_filtrado_blob,
                    forma_normalizada_blob
                FROM segmentos_patrones_dtw
                ORDER BY patron_id, id
                """
            )

            for (
                patron_id,
                segmento_original_blob,
                segmento_filtrado_blob,
                forma_normalizada_blob,
            ) in cur.fetchall():
                segmento_original = blob_a_array_bd(segmento_original_blob)
                segmento_filtrado = blob_a_array_bd(segmento_filtrado_blob)
                forma = blob_a_array_bd(forma_normalizada_blob)

                senal_metrica = (
                    segmento_filtrado
                    if len(segmento_filtrado) > 1
                    else segmento_original
                )

                if len(forma) == 0 and len(senal_metrica) > 1:
                    forma = calcular_forma_normalizada_identificacion(senal_metrica)

                if len(forma) == 0 or len(senal_metrica) < 2:
                    continue

                miembros_por_patron.setdefault(int(patron_id), []).append({
                    "forma": forma,
                    "metrica_dtw": metrica_dtw_desde_senal_segmento(senal_metrica),
                })

        conn.close()

    except sqlite3.Error as e:
        conn.close()
        return [], f"Error en llegir la base de dades:\n{e}"

    grupos = []

    for fila in filas:
        fila_dict = dict(zip(columnas_select, fila))

        patron_id = int(valor_fila_bd(fila_dict, "id"))
        tipo_patron = str(valor_fila_bd(fila_dict, "tipo_patron", ""))
        eje = str(valor_fila_bd(fila_dict, "eje", ""))

        forma = blob_a_array_bd(valor_fila_bd(fila_dict, "forma_normalizada_blob"))
        segmento_original = blob_a_array_bd(
            valor_fila_bd(fila_dict, "segmento_original_blob")
        )
        segmento_filtrado = blob_a_array_bd(
            valor_fila_bd(fila_dict, "segmento_filtrado_blob")
        )

        if len(forma) == 0 and len(segmento_filtrado) > 1:
            forma = calcular_forma_normalizada_identificacion(segmento_filtrado)

        if len(forma) == 0 and len(segmento_original) > 1:
            forma = calcular_forma_normalizada_identificacion(segmento_original)

        metrica_senal = (
            segmento_filtrado
            if len(segmento_filtrado) > 1
            else segmento_original
        )

        if len(segmento_original) > 1:
            espectro_comparacion = calcular_espectro_comparacion_identificacion(
                segmento_original
            )
            features_frecuencia_bandas = calcular_features_frecuencia_bandas_bd(
                segmento_original
            )
        else:
            espectro_comparacion = np.zeros(
                LONGITUD_ESPECTRO_IDENTIFICACION_BD
            )
            features_frecuencia_bandas = {
                col: 0.0
                for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD
            }

        limites_frecuencia_bandas = {}

        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            valor = valor_fila_bd(fila_dict, col)

            if valor is None:
                continue

            try:
                valor = float(valor)
            except Exception:
                continue

            if np.isfinite(valor):
                limites_frecuencia_bandas[col] = (None, valor)

        if len(limites_frecuencia_bandas) == 0:
            limites_frecuencia_bandas = {
                col: (None, valor)
                for col, valor in features_frecuencia_bandas.items()
            }

        limites = {
            "duracion_s": (
                valor_fila_bd(fila_dict, "duracion_s_min"),
                valor_fila_bd(fila_dict, "duracion_s_max"),
            ),
            "rms_original": (
                None,
                valor_fila_bd(fila_dict, "rms_original_max"),
            ),
            "pico_pico_original": (
                None,
                valor_fila_bd(fila_dict, "pico_pico_original_max"),
            ),
            "rizado_pico_pico_original": (
                None,
                valor_fila_bd(fila_dict, "rizo_pico_pico_original_max"),
            ),
            "offset_abs_original": (
                None,
                valor_fila_bd(fila_dict, "offset_abs_original_max"),
            ),
        }

        miembros_clasificacion = miembros_por_patron.get(patron_id, [])

        if not miembros_clasificacion and len(forma) > 0 and len(metrica_senal) > 1:
            miembros_clasificacion = [{
                "forma": forma,
                "metrica_dtw": metrica_dtw_desde_senal_segmento(metrica_senal),
            }]

        grupos.append({
            "id": patron_id,
            "etiqueta": f"G{patron_id}",
            "tipo_patron": tipo_patron,
            "eje": eje,
            "forma": forma,
            "segmento_original": segmento_original,
            "metrica_dtw": metrica_dtw_desde_senal_segmento(metrica_senal),
            "miembros_clasificacion": miembros_clasificacion,
            "espectro_comparacion": espectro_comparacion,
            "limites_frecuencia_bandas": limites_frecuencia_bandas,
            "limites": limites,
        })

    return grupos, None


def obtener_motivos_fuera_limites_segmento_grupo(segmento, grupo_bd):
    tipo = segmento.get("tipo_patron")
    motivos = []

    if tipo == "desplazamiento":
        if USAR_LIMITES_FISICOS_DESPLAZAMIENTO_BD:
            motivos.extend(
                comprobar_features_limites_superiores_con_grupo(
                    segmento=segmento,
                    grupo_bd=grupo_bd,
                    features=FEATURES_DESPLAZAMIENTO_IDENTIFICACION_BD,
                    margen=MARGEN_LIMITES_IDENTIFICACION_BD
                )
            )

        motivos.extend(comprobar_frecuencia_bandas_con_grupo(segmento, grupo_bd))

    elif tipo == "oscilatorio":
        motivos.extend(
            comprobar_features_limites_superiores_con_grupo(
                segmento=segmento,
                grupo_bd=grupo_bd,
                features=FEATURES_OSCILATORIO_IDENTIFICACION_BD,
                margen=MARGEN_LIMITES_IDENTIFICACION_BD
            )
        )

        motivos.extend(comprobar_frecuencia_bandas_con_grupo(segmento, grupo_bd))

    elif tipo == "reposo":
        motivos.extend(
            comprobar_features_limites_superiores_con_grupo(
                segmento=segmento,
                grupo_bd=grupo_bd,
                features=FEATURES_REPOSO_IDENTIFICACION_BD,
                margen=MARGEN_LIMITES_IDENTIFICACION_BD
            )
        )

        motivos.extend(comprobar_frecuencia_bandas_con_grupo(segmento, grupo_bd))

    return motivos


def asignar_segmentos_forma_frecuencia_con_bd(segmentos, grupos_eje, eje):
    """
    Criterio actual:
    - desplazamiento:
        * primero se intenta clasificar por forma filtrada/DTW + duración;
        * si no encaja con ningún grupo, se marca como comportamiento desconocido;
        * si encaja con un grupo, se valida con rizado y energia per bandes;
        * la distància DTW NO se usa como anomalía.
    - oscilatorio/reposo:
        * se comparan directamente con su patrón fijo.
    """
    umbral_forma = obtener_umbral_forma_desplazamiento_bd(eje)

    for seg in segmentos:
        tipo_patron = seg.get("tipo_patron")

        grupos_tipo = [
            g for g in grupos_eje
            if g.get("tipo_patron") == tipo_patron
            and g.get("eje") == eje
        ]

        if len(grupos_tipo) == 0:
            seg["grupo_id"] = None
            seg["grupo_etiqueta"] = "Sense patró en la BD"
            seg["desconocido"] = True
            seg["anomalo"] = True
            seg["motivos_anomalia"].append(
                f"No hi ha cap patró {etiqueta_tipus_patro(tipo_patron).lower()} en la base de dades per a l’eix {eje.upper()}"
            )
            continue

        if tipo_patron == "desplazamiento":
            candidatos = []

            for grupo in grupos_tipo:
                distancia_forma = distancia_forma_identificacion(seg, grupo)

                distancia_clasificacion, info_duracion = distancia_clasificacion_grupo_desplazamiento(
                    segmento=seg,
                    grupo_bd=grupo,
                    distancia_forma=distancia_forma
                )

                forma_compatible = bool(distancia_forma <= umbral_forma)
                duracion_compatible = bool(info_duracion["compatible"])
                grupo_compatible = forma_compatible and duracion_compatible

                candidatos.append({
                    "grupo": grupo,
                    "distancia_forma": float(distancia_forma),
                    "distancia_clasificacion": float(distancia_clasificacion),
                    "forma_compatible": forma_compatible,
                    "duracion_compatible": duracion_compatible,
                    "grupo_compatible": grupo_compatible,
                    "distancia_duracion": float(info_duracion["distancia"]),
                    "motivo_duracion": info_duracion["motivo"],
                })

            candidatos.sort(
                key=lambda item: (
                    not item["grupo_compatible"],
                    item["distancia_clasificacion"],
                    item["distancia_forma"],
                )
            )

            mejor = candidatos[0]
            mejor_grupo = mejor["grupo"]

            seg["grupo_candidato_id"] = mejor_grupo.get("id")
            seg["grupo_candidato_etiqueta"] = mejor_grupo.get("etiqueta")
            seg["distancia_forma_candidato"] = mejor["distancia_forma"]
            seg["distancia_clasificacion_candidato"] = mejor["distancia_clasificacion"]
            seg["distancia_duracion_candidato"] = mejor["distancia_duracion"]
            seg["forma_compatible_candidato"] = mejor["forma_compatible"]
            seg["duracion_compatible_candidato"] = mejor["duracion_compatible"]

            if not mejor["grupo_compatible"]:
                seg["grupo_id"] = None
                seg["grupo_etiqueta"] = "Comportament desconegut"
                seg["distancia_dtw"] = None
                seg["distancia_forma"] = mejor["distancia_forma"]
                seg["distancia_clasificacion"] = mejor["distancia_clasificacion"]
                seg["distancia_duracion"] = mejor["distancia_duracion"]
                seg["duracion_compatible_grupo"] = mejor["duracion_compatible"]
                seg["comparacion_directa_bd"] = False
                seg["desconocido"] = True
                seg["anomalo"] = True

                motivos_desconocido = []

                if not mejor["forma_compatible"]:
                    motivos_desconocido.append(
                        f"distància combinada no compatible amb cap grup: "
                        f"distància {mejor['distancia_forma']:.3f} > llindar {umbral_forma:.3f}"
                    )

                if not mejor["duracion_compatible"]:
                    if mejor["motivo_duracion"]:
                        motivos_desconocido.append(mejor["motivo_duracion"])
                    else:
                        motivos_desconocido.append(
                            "duració incompatible amb tots els grups de desplaçament"
                        )

                if len(motivos_desconocido) == 0:
                    motivos_desconocido.append(
                        "no pertany a cap grup de desplaçament conegut"
                    )

                seg["motivos_anomalia"].extend(motivos_desconocido)

                seg["motivos_diagnostico_candidato"] = obtener_motivos_fuera_limites_segmento_grupo(
                    seg,
                    mejor_grupo
                )

                continue

            seg["grupo_id"] = mejor_grupo["id"]
            seg["grupo_etiqueta"] = mejor_grupo["etiqueta"]
            seg["distancia_dtw"] = mejor["distancia_forma"]
            seg["distancia_forma"] = mejor["distancia_forma"]
            seg["distancia_clasificacion"] = mejor["distancia_clasificacion"]
            seg["distancia_duracion"] = mejor["distancia_duracion"]
            seg["duracion_compatible_grupo"] = mejor["duracion_compatible"]
            seg["comparacion_directa_bd"] = False
            seg["desconocido"] = False

            motivos = obtener_motivos_fuera_limites_segmento_grupo(seg, mejor_grupo)

            if len(motivos) > 0:
                seg["anomalo"] = True
                seg["motivos_anomalia"].extend(motivos)

            continue

        if tipo_patron in ("oscilatorio", "reposo"):
            grupo_ref = seleccionar_grupo_directo_tipo_bd(
                grupos_eje=grupos_eje,
                tipo_patron=tipo_patron,
                eje=eje
            )

            if grupo_ref is None:
                seg["grupo_id"] = None
                seg["grupo_etiqueta"] = "Sense patró en la BD"
                seg["desconocido"] = True
                seg["anomalo"] = True
                seg["motivos_anomalia"].append(
                    f"No hi ha cap patró {etiqueta_tipus_patro(tipo_patron).lower()} en la base de dades per a l’eix {eje.upper()}"
                )
                continue

            seg["grupo_id"] = grupo_ref["id"]
            seg["grupo_etiqueta"] = grupo_ref["etiqueta"]
            seg["distancia_dtw"] = None
            seg["distancia_forma"] = None
            seg["comparacion_directa_bd"] = True
            seg["desconocido"] = False

            motivos = obtener_motivos_fuera_limites_segmento_grupo(seg, grupo_ref)

            if len(motivos) > 0:
                seg["anomalo"] = True
                seg["motivos_anomalia"].extend(motivos)

            continue


def identificar_segmentos_bd_eje(acc_sin_filtrar, acc_filtrada, info_seg, grupos_bd, eje):
    segmentos = crear_segmentos_actuales_identificacion(
        acc_sin_filtrar=acc_sin_filtrar,
        acc_filtrada=acc_filtrada,
        info_seg=info_seg,
        eje=eje
    )

    grupos_segmentados = [
        g for g in grupos_bd
        if g.get("tipo_patron") != "senyal_completa"
    ]

    grupos_eje = [g for g in grupos_segmentados if g.get("eje") == eje]

    asignar_segmentos_forma_frecuencia_con_bd(segmentos, grupos_eje, eje)
    info_matriz = construir_matriz_forma_frecuencia_segmentos_grupos(segmentos, grupos_eje, eje)

    return {
        "eje": eje,
        "segmentos": segmentos,
        "grupos_bd": grupos_bd,
        "grupos_eje": grupos_eje,
        "grupos_mov": obtener_grupos_bd_por_tipo_eje(grupos_segmentados, "desplazamiento", eje),
        "matriz": info_matriz,
        "anomalias": [s for s in segmentos if s.get("anomalo")],
    }


def comparar_senyal_completa_con_bd(acc_sin_filtrar, grupos_bd):
    """
    Compara la energia per bandes de la señal completa actual contra el patrón
    senyal_completa de la BD.
    """
    resultado = {
        "por_eje": {},
        "anomalias": [],
    }

    grupos_senyal = {
        g.get("eje"): g
        for g in grupos_bd
        if g.get("tipo_patron") == "senyal_completa"
    }

    for eje in ["x", "y", "z"]:
        idx = IDX_EJE[eje]
        senyal = np.asarray(acc_sin_filtrar[:, idx], dtype=float)
        features = calcular_features_frecuencia_bandas_bd(senyal)
        grupo = grupos_senyal.get(eje)

        info_eje = {
            "eje": eje,
            "features_frecuencia_bandas": features,
            "grupo_id": None,
            "grupo_etiqueta": "Sense grup senyal_completa",
            "anomalo": False,
            "motivos_anomalia": [],
        }

        if grupo is None:
            info_eje["anomalo"] = True
            info_eje["motivos_anomalia"].append(
                f"No hi ha cap patró senyal_completa en la base de dades per a l’eix {eje.upper()}"
            )
        else:
            info_eje["grupo_id"] = grupo.get("id")
            info_eje["grupo_etiqueta"] = grupo.get("etiqueta")
            motivos = comparar_energias_bandas_con_limites(
                features_actuales=features,
                limites_bandas=grupo.get("limites_frecuencia_bandas", {}),
                margen=MARGEN_FRECUENCIA_BANDAS_BD,
                prefijo="senyal_completa: energia per bandes"
            )

            if len(motivos) > 0:
                info_eje["anomalo"] = True
                info_eje["motivos_anomalia"].extend(motivos)

        resultado["por_eje"][eje] = info_eje

        if info_eje["anomalo"]:
            resultado["anomalias"].append(info_eje)

    return resultado


def identificar_segmentos_bd_xy(acc_sin_filtrar, acc_filtrada, info_seg):
    grupos_bd, error = leer_grupos_bd_identificacion()

    info = {
        "grupos_bd": grupos_bd,
        "error_bd": error,
        "por_eje": {},
        "senyal_completa": None,
        "impactos": obtener_impactos_globales(info_seg),
    }

    if error is not None:
        print("\n[ERROR BD]", error)
        return info

    for eje in EJES_CLASIFICACION_DTW:
        info["por_eje"][eje] = identificar_segmentos_bd_eje(
            acc_sin_filtrar=acc_sin_filtrar,
            acc_filtrada=acc_filtrada,
            info_seg=info_seg,
            grupos_bd=grupos_bd,
            eje=eje
        )

    info["senyal_completa"] = comparar_senyal_completa_con_bd(
        acc_sin_filtrar=acc_sin_filtrar,
        grupos_bd=grupos_bd
    )

    if IMPRIMIR_RESUMEN_ANOMALIAS_BD:
        imprimir_resumen_identificacion_bd(info)

    return info


def imprimir_resumen_identificacion_bd(info_identificacion):
    print("\nResum d'identificació amb la base de dades")
    print("-" * 80)

    if info_identificacion.get("error_bd") is not None:
        print(info_identificacion["error_bd"])
        return

    total_anomalias = 0

    for eje, info_eje in info_identificacion.get("por_eje", {}).items():
        anomalias = info_eje.get("anomalias", [])
        total_anomalias += len(anomalias)
        print(f"Eix {eje.upper()}: {len(anomalias)} possible(s) anomalia(es) en segments")

        for seg in anomalias:
            motivos = "; ".join(seg.get("motivos_anomalia", []))
            print(
                f"  {seg['etiqueta']} ({etiqueta_tipus_patro(seg['tipo_patron'])}) "
                f"-> {seg['grupo_etiqueta']}: {motivos}"
            )

    info_senyal = info_identificacion.get("senyal_completa")

    if info_senyal is not None:
        anomalias_senyal = info_senyal.get("anomalias", [])
        total_anomalias += len(anomalias_senyal)
        print(
            f"Senyal completa: {len(anomalias_senyal)} eix(os) amb energia per bandes fora dels límits"
        )

        for info_eje in anomalias_senyal:
            motivos = "; ".join(info_eje.get("motivos_anomalia", []))
            print(
                f"  Eix {info_eje['eje'].upper()} -> "
                f"{info_eje['grupo_etiqueta']}: {motivos}"
            )

    if total_anomalias == 0:
        print("No s'han detectat possibles anomalies amb els criteris actuals.")

    print("-" * 80)


def ejecutar_analisis_desde_seleccion(seleccion, parent=None):
    """
    Eixcuta un análisis completo a partir de la selección inicial.
    Puede llamarse varias veces sin cerrar la ventana principal.
    """
    global experimento, bloque, CARGAR_TODOS_LOS_BLOQUES

    experimento = seleccion["experimento"]
    bloque = seleccion["bloque"]
    CARGAR_TODOS_LOS_BLOQUES = seleccion["cargar_todos_los_bloques"]
    carpeta = seleccion["carpeta"]
    modo_visualizacion = seleccion.get("modo_visualizacion", "depuracion")

    actualizar_umbral_reposo_desde_experimento_referencia()

    (
        t,
        acc_sin_filtrar,
        acc_filtrada_suave,
        acc_filtrada_agresiva,
        unidad_acc
    ) = cargar_datos_carpeta(carpeta)

    acc_segmentacion = acc_filtrada_suave
    acc_validacion_desplazamiento = acc_filtrada_agresiva
    acc_clasificacion_dtw = acc_validacion_desplazamiento

    if USAR_SENAL_SIN_FILTRAR_PARA_IMPACTOS:
        acc_impactos = acc_sin_filtrar
    else:
        acc_impactos = acc_segmentacion

    info_seg = segmentar_xy_sin_z(
        acc_segmentacion_suave=acc_segmentacion,
        acc_validacion_desplazamiento=acc_validacion_desplazamiento,
        acc_impactos=acc_impactos
    )

    imprimir_impactos_detectados(info_seg)

    titulo = (
        f"Experiència {experimento} | {carpeta.name} | "
        "identificació"
    )

    if CARGAR_TODOS_LOS_BLOQUES:
        titulo += " | tots els blocs"
    else:
        titulo += f" | bloc {bloque}"

    f, espectro = calcular_espectro_fft(acc_filtrada_suave)

    fig_ac = crear_figura_aceleracion_xy_sin_z(
        t=t,
        acc_sin_filtrar=acc_sin_filtrar,
        acc_filtrada_suave=acc_filtrada_suave,
        acc_filtrada_agresiva=acc_filtrada_agresiva,
        info_seg=info_seg,
        unidad_acc=unidad_acc,
        titulo=titulo,
    )

    fig_debug_x = crear_figura_debug_eje(
        t=t,
        acc_segmentacion=acc_segmentacion,
        acc_validacion_desplazamiento=acc_validacion_desplazamiento,
        info_seg=info_seg,
        unidad_acc=unidad_acc,
        titulo=titulo,
        eje="x",
    )

    fig_debug_y = crear_figura_debug_eje(
        t=t,
        acc_segmentacion=acc_segmentacion,
        acc_validacion_desplazamiento=acc_validacion_desplazamiento,
        info_seg=info_seg,
        unidad_acc=unidad_acc,
        titulo=titulo,
        eje="y",
    )

    fig_frec = crear_figura_frecuencia(
        f=f,
        espectro=espectro,
        unidad_acc=unidad_acc,
        titulo=titulo,
    )

    info_identificacion_bd = None
    figuras_matriz_bd = None

    if USAR_CLASIFICACION_DTW:
        info_identificacion_bd = identificar_segmentos_bd_xy(
            acc_sin_filtrar=acc_sin_filtrar,
            acc_filtrada=acc_clasificacion_dtw,
            info_seg=info_seg
        )

        figuras_matriz_bd = {}

        for eje in EJES_CLASIFICACION_DTW:
            info_eje_bd = info_identificacion_bd.get("por_eje", {}).get(eje)

            if MOSTRAR_MATRIZ_DISTANCIAS_DTW and info_eje_bd is not None:
                figuras_matriz_bd[eje] = crear_figura_matriz_bd_identificacion_eje(
                    info_eje=info_eje_bd,
                    titulo=titulo
                )
            else:
                figuras_matriz_bd[eje] = None

    fig_ac_usuario = crear_figura_aceleracion_modo_usuario(
        t=t,
        acc_sin_filtrar=acc_sin_filtrar,
        acc_filtrada_suave=acc_filtrada_suave,
        acc_filtrada_agresiva=acc_filtrada_agresiva,
        info_seg=info_seg,
        info_identificacion_bd=info_identificacion_bd,
        unidad_acc=unidad_acc,
        titulo=titulo,
    )

    fig_modulo = crear_figura_modulo_aceleracion(
        t=t,
        acc_sin_filtrar=acc_sin_filtrar,
        unidad_acc=unidad_acc,
        titulo=titulo,
    )

    mostrar_ventana(
        fig_ac=fig_ac,
        fig_debug_x=fig_debug_x,
        fig_debug_y=fig_debug_y,
        fig_frec=fig_frec,
        info_identificacion_bd=info_identificacion_bd,
        figuras_matriz_bd=figuras_matriz_bd,
        titulo_clasificacion=titulo,
        parent=parent,
        modo_visualizacion=modo_visualizacion,
        fig_ac_usuario=fig_ac_usuario,
        fig_modulo=fig_modulo,
    )


def main():
    pedir_configuracion_inicial()


if __name__ == "__main__":
    main()
