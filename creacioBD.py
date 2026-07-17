
"""
creacioBDpy

Crea des de zero la base de dades de patrons normals.

"""


import matplotlib
matplotlib.use("TkAgg")

import re
import sqlite3
import io
import hashlib
import time
import gc
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


##################################################
# DATOS BÁSICOS
##################################################

Fs = 1000

experimento = "123, 124, 132, 133, 135, 136, 138, 139, 140, 141, 142, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160"

bloque = 0

# True  -> carga todos los timeblock*.txt de la carpeta seleccionada
# False -> carga solo timeblock{bloque}.txt
CARGAR_TODOS_LOS_BLOQUES = True

# Pausa breve entre experimentos/carpetas durante guardado masivo.
# Reduce un poco la carga continua de CPU y mantiene la interfaz más respirable.
PAUSA_CPU_BATCH_S = 0.02

CONVERTIR_A_MS2 = False
G0 = 9.80665

APLICAR_RESTA_MEDIA = True

RUTA_SCRIPT = Path(__file__).resolve().parent
RUTA_DATOS = RUTA_SCRIPT.parent


##################################################
# FILTRADO
##################################################

APLICAR_FILTRO_PASO_BAJO = True


# - filtro suave: primera segmentación y detección de candidatos;
# - filtro agresivo: validación y ajuste de límites de desplazamiento.
FILTRO_SEGMENTACION_SUAVE_HZ = 150.0
FILTRO_VALIDACION_DESPLAZAMIENTO_HZ = 20.0

FILTRO_LOWPASS_HZ = FILTRO_SEGMENTACION_SUAVE_HZ

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

# Los candidatos a desplazamiento detectados con el filtro suave se revisan
# con la señal filtrada agresivamente. Si no superan UMBRAL_DESPLAZAMIENTO
# en la señal de validación, se reclasifican como oscilatorios.
MARGEN_VALIDACION_DESPLAZAMIENTO_MS = 0

# Si dentro de la ventana hay suficientes cruces por cero, ese tramo se
# considera oscilatorio aunque su amplitud supere el umbral de desplazamiento.
# Esto evita que una vibración de alta amplitud se pinte como desplazamiento.
USAR_CRUCES_CERO_PARA_RECLASIFICAR_NORMAL = True


# Duraciones mínimas.
MIN_DUR_NORMAL_MS = 50
MIN_DUR_OSC_MS = 40

# Corrección final: un desplazamiento demasiado corto se considera oscilatorio.
# Se aplica al final del postprocesado de los segmentos rojos, para medir la
# duración real del segmento ya ajustado/recortado.
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

# división híbrida de desplazamientos.
# La señal de 20 Hz valida que existe desplazamiento y detecta valles internos;
# la señal de 150 Hz da el punto fino de corte, porque conserva mejor los pasos
# por cero que pueden desaparecer con el filtrado agresivo.
DIVIDIR_DESPLAZAMIENTO_HIBRIDO_20_150 = True
VENTANA_LOBULO_DIVISION_HIBRIDA_MS = 120
VENTANA_BUSQUEDA_CORTE_150HZ_MS = 35
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


# Reposo visual.
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

# Porcentaje mínimo de muestras dentro de la ventana que deben superar el umbral de desplazamiento.
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
# PREPARACIÓN DE SEGMENTOS PARA DTW/BD INDEPENDIENTE X/Y
##################################################

USAR_CLASIFICACION_DTW = True

# Cada eje se clasifica por separado: segmentos X con señal X, segmentos Y con señal Y.
EJES_CLASIFICACION_DTW = ["x", "y"]

# Tipo de segmento a clasificar por eje:
# "todos"       -> normales + oscilatorios
# "normal"      -> solo lóbulos normales rojos
# "oscilatorio" -> solo oscilatorios amarillos
TIPO_SEGMENTOS_DTW = "normal"

# Además de clasificar los movimientos normales por DTW, se añaden grupos fijos
# para oscilatorios y reposo. Estos grupos NO se normalizan ni se comparan por DTW.
INCLUIR_GRUPO_OSCILATORIO = True
INCLUIR_GRUPO_REPOSO = True

LONGITUD_DTW = 120

NUM_CLUSTERS_DTW = {"x": None, "y": None}

# Umbral de distancia por eje.
DISTANCIA_CLUSTER_DTW = {"x": 0.20, "y": 0.20}

MIN_MUESTRAS_SEGMENTO_DTW = 20

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

# "complete" es estricto y evita grupos por efecto cadena.
LINKAGE_DTW = "complete"

MOSTRAR_MATRIZ_DISTANCIAS_DTW = True
MOSTRAR_DENDROGRAMA_DTW = True

CMAP_MATRIZ_DISTANCIAS = "viridis"
ANOTAR_VALORES_MATRIZ = True
MAX_SEGMENTOS_ANOTAR_MATRIZ = 9999
DECIMALES_MATRIZ_DISTANCIAS = 2


##################################################
# BASE DE DATOS DE PATRONES NORMALES DTW
##################################################

GUARDAR_BD_PATRONES_NORMALES = True
MOSTRAR_BD_EN_VENTANA = True

NOMBRE_BD_PATRONES_NORMALES = "patrones_normales_dtw.sqlite"

# Si se pone a True, borra todos los patrones guardados antes de insertar
BORRAR_BD_PATRONES_ANTES_DE_GUARDAR = True

# Si un patrón nuevo se parece a uno ya guardado por debajo de este umbral,
# no se inserta una fila nueva: se actualizan los mínimos y máximos del patrón existente.
# Si se deja en None, se usa DISTANCIA_CLUSTER_DTW[eje].
UMBRAL_FUSION_PATRONES_BD = {"x": 0.20, "y": 0.20}


# Escala horizontal común en Segmentos BD.
# Solo se aplica a los grupos de desplazamiento.
USAR_ESCALA_X_COMUN_DESPLAZAMIENTO_SEGMENTOS_BD = True
MARGEN_ESCALA_X_DESPLAZAMIENTO_SEGMENTOS_BD = 1.05

# Bandas espectrales guardadas en la BD.
# Se guarda únicamente el máximo de energía de la señal original.
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

ETIQUETAS_BANDAS_FRECUENCIA_BD = [
    f"{f_ini}-{f_fin} Hz"
    for f_ini, f_fin in BANDAS_FRECUENCIA_BD
]


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
            f"Experiència {num_experimento} fora del rang contemplat. "
            "Rangs vàlids: 65-113 per a pòrtic, 114-119 i >122 per a màquina real."
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
            raise FileNotFoundError(f"No hi ha timeblock*.txt en:\n{carpeta}")

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
        raise ValueError("La freqüència de tall ha d'estar entre 0 i Fs/2.")

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

    print(f"\nCarregant carpeta: {carpeta.name}")

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
            "  Filtres pas baix: "
            f"suau {FILTRO_SEGMENTACION_SUAVE_HZ:.1f} Hz | "
            f"validació {FILTRO_VALIDACION_DESPLAZAMIENTO_HZ:.1f} Hz | "
            f"orde {ORDEN_FILTRO}"
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
            f"No existeix la ruta de l'experiència de referència {num_experimento}:\n{ruta_ref}"
        )

    carpetas_ref = obtener_carpetas_con_timeblocks(ruta_ref)

    if len(carpetas_ref) == 0:
        raise FileNotFoundError(
            f"No s'han trobat timeblock*.txt en l'experiència de referència {num_experimento}:\n{ruta_ref}"
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
            f"No s'ha pogut carregar cap timeblock*.txt de l'experiència {num_experimento}."
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

    print("\nLlindar de repòs calculat des de l'experiència de referència")
    print("-" * 80)
    print(f"Experiència de referència: {EXPERIMENTO_REFERENCIA_REPOSO}")
    print(f"Ruta de referència:     {info['ruta_ref']}")
    print(f"Carpetes utilitzades:   {info['n_carpetas']}")
    print(f"Fitxers utilitzats:     {info['n_archivos']}")
    print(f"Mitjana X aplicada:     {info['media_x']:.8f} {unidad_acc}")
    print(f"Mitjana Y aplicada:     {info['media_y']:.8f} {unidad_acc}")
    print(f"Màx. abs X ref. sense filtrar: {info['max_abs_x']:.8f} {unidad_acc}")
    print(f"Màx. abs Y ref. sense filtrar: {info['max_abs_y']:.8f} {unidad_acc}")
    print(f"Factor de repòs:        {FACTOR_UMBRAL_REPOSO:.3f}")
    print(f"Factor de desplaçament: {FACTOR_UMBRAL_DESPLAZAMIENTO:.3f}")
    print(f"Llindar de repòs X:     {UMBRAL_REPOSO['x']:.8f} {unidad_acc}")
    print(f"Llindar de repòs Y:     {UMBRAL_REPOSO['y']:.8f} {unidad_acc}")
    print(f"Llindar de desplaç. X:  {info['umbral_desplazamiento']['x']:.8f} {unidad_acc}")
    print(f"Llindar de desplaç. Y:  {info['umbral_desplazamiento']['y']:.8f} {unidad_acc}")
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

    Se aplica después del postprocesado de los rojos, porque así la duración se
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
    Devuelve True si entre dos muestras consecutivas hay cruce real por 0
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


def obtener_candidatos_corte_150hz(x_suave, ini, fin, eje):
    """
    Devuelve posibles puntos de corte dentro de un desplazamiento usando la
    señal filtrada a 150 Hz.

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


def dividir_normales_hibrido_valle_20hz_corte_150hz(
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
    - 150 Hz: fija el instante exacto de corte, porque conserva mejor los pasos
      por cero que el filtrado a 20 Hz puede suavizar.

    Esto evita dos errores opuestos:
    - no cortar por cualquier rizado de 150 Hz;
    - no unir dos lóbulos reales porque 20 Hz haya suavizado el cruce por cero.
    """
    if not DIVIDIR_DESPLAZAMIENTO_HIBRIDO_20_150:
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
    radio_corte = max(1, int(VENTANA_BUSQUEDA_CORTE_150HZ_MS * Fs / 1000))
    min_len = max(1, int(MIN_DUR_NORMAL_MS * Fs / 1000))
    min_sep = max(1, int(MIN_SEPARACION_CORTES_HIBRIDA_MS * Fs / 1000))

    segmentos_refinados = []

    for ini, fin in sorted((int(a), int(b)) for a, b in segmentos):
        ini = max(0, ini)
        fin = min(N - 1, fin)

        if fin - ini + 1 < 2 * min_len:
            segmentos_refinados.append((ini, fin))
            continue

        candidatos = obtener_candidatos_corte_150hz(
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
      duración mínima, pico suficiente y pocos cruces internos.
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
    1) cruce real de signo entre dos muestras consecutivas;
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
    Segmentación V5 por eje.

    1) La señal suave se usa para detectar actividad y candidatos a desplazamiento.
    2) La señal agresiva se usa para validar todos los candidatos fuertes.
    3) La señal agresiva de 20 Hz valida los desplazamientos y marca la
       tendencia general.
    4) Las divisiones internas se deciden de forma híbrida: valle en 20 Hz y
       corte fino en 150 Hz.
    """
    N = len(x_suave)

    if x_validacion is None:
        x_validacion = x_suave

    x_limites_desplazamiento = x_validacion

    mask_no_reposo = np.abs(x_suave) >= UMBRAL_REPOSO[eje]
    mask_movimiento_fuerte_suave = np.abs(x_suave) >= umbral_desplazamiento_eje(eje)
    mask_actividad_debil = mask_no_reposo & ~mask_movimiento_fuerte_suave

    cruces_ventana, rms_ventana = calcular_cruces_y_rms_ventana(x_suave, eje)
    mask_cruces_altos = cruces_ventana >= MIN_CRUCES_OSC

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


    segmentos_normales = dividir_normales_hibrido_valle_20hz_corte_150hz(
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


def obtener_segmentos_eje_para_dtw(info_seg, eje):
    """
    Devuelve los segmentos que entran en la clasificación DTW y sus etiquetas.
    """
    segs_eje = info_seg["segmentos_por_eje"][eje]

    todos = list(segs_eje["todos"])
    etiquetas_todos = list(segs_eje["etiquetas_todos"])
    normales = set(segs_eje["normal"])
    oscilatorios = set(segs_eje["oscilatorio"])

    segmentos = []
    etiquetas = []

    for seg, etiqueta_original in zip(todos, etiquetas_todos):
        if TIPO_SEGMENTOS_DTW == "todos":
            incluir = True

        elif TIPO_SEGMENTOS_DTW == "normal":
            incluir = seg in normales

        elif TIPO_SEGMENTOS_DTW == "oscilatorio":
            incluir = seg in oscilatorios

        else:
            raise ValueError("TIPO_SEGMENTOS_DTW debe ser 'todos', 'normal' u 'oscilatorio'.")

        if incluir:
            segmentos.append(seg)
            etiquetas.append(etiqueta_original)

    return segmentos, etiquetas


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


def extraer_senal_cruda_segmento_eje(acc, eje, ini, fin):
    idx = IDX_EJE[eje]
    return np.asarray(acc[ini:fin + 1, idx], dtype=float)


def calcular_metricas_segmento_crudo(acc, eje, ini, fin):
    x = extraer_senal_cruda_segmento_eje(acc, eje, ini, fin)

    if len(x) == 0:
        return {
            "duracion_s": 0.0,
            "rms": 0.0,
            "pico_abs": 0.0,
            "pico_firmado": 0.0,
            "signo_pico": 0.0,
            "ini": int(ini),
            "fin": int(fin),
        }

    idx_pico = int(np.argmax(np.abs(x)))
    pico_firmado = float(x[idx_pico])

    return {
        "duracion_s": float((fin - ini + 1) / Fs),
        "rms": float(np.sqrt(np.mean(x * x))),
        "pico_abs": float(np.max(np.abs(x))),
        "pico_firmado": pico_firmado,
        "signo_pico": float(np.sign(pico_firmado)),
        "ini": int(ini),
        "fin": int(fin),
    }


def crear_grupos_fijos_no_dtw(acc, info_seg, eje):
    """
    Crea grupos fijos que no pasan por DTW:
    - un grupo con todos los segmentos oscilatorios;
    - un grupo con todos los segmentos de reposo.

    Las señales se guardan crudas, sin remuestrear ni normalizar, para poder
    extraer características físicas directamente.
    """
    grupos_fijos = []

    tipos = []

    if INCLUIR_GRUPO_OSCILATORIO:
        tipos.append(("oscilatorio", "Oscil·latori"))

    if INCLUIR_GRUPO_REPOSO:
        tipos.append(("reposo", "Repòs"))

    for tipo, nombre in tipos:
        segmentos, etiquetas = obtener_etiquetas_segmentos_tipo_eje(
            info_seg=info_seg,
            eje=eje,
            tipo=tipo
        )

        if len(segmentos) == 0:
            continue

        senales = []
        metricas = []

        for ini, fin in segmentos:
            senales.append(
                extraer_senal_cruda_segmento_eje(
                    acc=acc,
                    eje=eje,
                    ini=ini,
                    fin=fin
                )
            )

            metricas.append(
                calcular_metricas_segmento_crudo(
                    acc=acc,
                    eje=eje,
                    ini=ini,
                    fin=fin
                )
            )

        grupos_fijos.append({
            "tipo": tipo,
            "nombre": nombre,
            "segmentos": segmentos,
            "etiquetas": etiquetas,
            "senales": senales,
            "metricas": metricas,
        })

    return grupos_fijos


def extraer_forma_segmento_eje(acc, eje, ini, fin):
    """
    Extrae la forma usada para comparar por DTW y las métricas físicas
    del segmento.

    La forma se z-normaliza para comparar la morfología temporal, pero
    las métricas conservan la amplitud y la duración reales para evitar
    agrupaciones falsas entre segmentos con escala física distinta.
    """
    if fin <= ini:
        return None, None

    if fin - ini + 1 < MIN_MUESTRAS_SEGMENTO_DTW:
        return None, None

    idx = IDX_EJE[eje]

    x_original = np.asarray(acc[ini:fin + 1, idx], dtype=float)

    if len(x_original) < 2:
        return None, None

    pico_abs = float(np.max(np.abs(x_original)))
    rms = float(np.sqrt(np.mean(x_original * x_original)))
    duracion_s = float((fin - ini + 1) / Fs)

    idx_pico = int(np.argmax(np.abs(x_original)))
    pico_firmado = float(x_original[idx_pico])
    signo_pico = float(np.sign(pico_firmado))

    forma = remuestrear_1d(x_original, LONGITUD_DTW)
    forma = z_normalizar_1d(forma)

    metricas = {
        "duracion_s": duracion_s,
        "rms": rms,
        "pico_abs": pico_abs,
        "pico_firmado": pico_firmado,
        "signo_pico": signo_pico,
        "ini": int(ini),
        "fin": int(fin),
    }

    return forma, metricas


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


def calcular_matriz_distancias_dtw_1d(formas, metricas=None):
    n = len(formas)
    D = np.zeros((n, n))

    if metricas is None:
        metricas = [None] * n

    for i in range(n):
        for j in range(i + 1, n):
            d = distancia_formas_1d(
                formas[i],
                formas[j],
                metricas[i],
                metricas[j]
            )
            D[i, j] = d
            D[j, i] = d

    return D


def obtener_medoide_cluster(indices, D):
    if len(indices) == 1:
        return indices[0]

    subD = D[np.ix_(indices, indices)]
    dist_media = np.mean(subD, axis=1)

    return indices[int(np.argmin(dist_media))]


def clasificar_segmentos_dtw_eje(acc, info_seg, eje):
    """
    Prepara los segmentos normales del experimento para compararlos contra la BD.
    """
    grupos_fijos = crear_grupos_fijos_no_dtw(
        acc=acc,
        info_seg=info_seg,
        eje=eje
    )

    segmentos, etiquetas = obtener_segmentos_eje_para_dtw(info_seg, eje)

    formas = []
    metricas = []
    segmentos_validos = []
    etiquetas_validas = []

    for seg, etiqueta in zip(segmentos, etiquetas):
        ini, fin = seg

        forma, metrica = extraer_forma_segmento_eje(
            acc=acc,
            eje=eje,
            ini=ini,
            fin=fin
        )

        if forma is None:
            continue

        formas.append(forma)
        metricas.append(metrica)
        segmentos_validos.append(seg)
        etiquetas_validas.append(etiqueta)

    if len(formas) == 0:
        print(f"\nNo hi ha segments de desplaçament vàlids en {eje.upper()} per a comparar amb la BD.")
    else:
        print(
            f"\nEix {eje.upper()}: {len(formas)} segment(s) de desplaçament "
            "preparat(s) per a comparar amb la BD. "
            "No es realitza agrupació interna entre segments de l'experiment."
        )

    grupos = {i + 1: [i] for i in range(len(formas))}
    medoides = {i + 1: i for i in range(len(formas))}
    labels = np.arange(1, len(formas) + 1, dtype=int)

    return {
        "eje": eje,
        "formas": formas,
        "metricas": metricas,
        "segmentos_validos": segmentos_validos,
        "etiquetas": etiquetas_validas,
        "D": None,
        "Z": None,
        "labels": labels,
        "grupos": grupos,
        "medoides": medoides,
        "distancia_cluster": DISTANCIA_CLUSTER_DTW[eje],
        "grupos_fijos": grupos_fijos,
    }

def clasificar_segmentos_dtw_xy(acc, info_seg):
    info_clasificacion = {}

    for eje in EJES_CLASIFICACION_DTW:
        info_clasificacion[eje] = clasificar_segmentos_dtw_eje(
            acc=acc,
            info_seg=info_seg,
            eje=eje
        )

    return info_clasificacion


##################################################
# GRÁFICAS
##################################################


def calcular_matriz_items_dtw(items):
    """
    Calcula la matriz de distancias DTW combinadas para una lista de items.
    Cada item debe contener 'forma' y 'metrica'.
    """
    formas = [item["forma"] for item in items]
    metricas = [item.get("metrica") for item in items]
    return calcular_matriz_distancias_dtw_1d(formas, metricas)


##################################################
# BASE DE DATOS DE PATRONES NORMALES DTW
##################################################

def array_a_blob(array):
    """
    Convierte un array NumPy a BLOB comprimido para guardarlo en SQLite.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.asarray(array))
    return buffer.getvalue()


def blob_a_array(blob):
    """
    Recupera un array NumPy guardado como BLOB.
    """
    if blob is None:
        return np.array([], dtype=float)

    buffer = io.BytesIO(blob)

    with np.load(buffer) as data:
        return data["data"]


def crear_conexion_bd_patrones():
    ruta_bd = RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES
    print(f"Base de dades: {ruta_bd.resolve()}")
    return sqlite3.connect(ruta_bd)


def asegurar_columna_bd(conn, tabla, columna, definicion_sql):
    """
    Añade una columna si la base de datos ya existía de una versión anterior.
    SQLite no modifica una tabla creada con CREATE TABLE IF NOT EXISTS,
    por eso se hace esta comprobación explícita.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({tabla})")
    columnas_existentes = [fila[1] for fila in cur.fetchall()]

    if columna not in columnas_existentes:
        cur.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {definicion_sql}")


COLUMNAS_ESCALARES_BD = [
    "tipo_patron",
    "eje",

    # Solo desplazamiento.
    "duracion_s_min",
    "duracion_s_max",

    # Patrones segmentados: desplazamiento, oscilatorio y reposo.
    "rms_original_max",
    "pico_pico_original_max",
    "rizo_pico_pico_original_max",

    # Solo desplazamiento.
    "dtw_distancia_max",

    # Solo reposo.
    "offset_abs_original_max",
]

COLUMNAS_ESCALARES_BD += COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD


COLUMNAS_BLOB_BD = [
    "forma_normalizada_blob",
    "segmento_original_blob",
    "segmento_filtrado_blob",
    "frecuencias_original_blob",
    "espectro_original_blob",
    "frecuencias_filtrado_blob",
    "espectro_filtrado_blob",
]

PARES_MIN_MAX_BD = [
    ("duracion_s_min", "duracion_s_max"),
]

COLUMNAS_MAX_BD = [
    "rms_original_max",
    "pico_pico_original_max",
    "rizo_pico_pico_original_max",
    "dtw_distancia_max",
    "offset_abs_original_max",
]

COLUMNAS_MAX_BD += COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD


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

    # Columnas antiguas retiradas. Se mantienen aquí solo para ocultarlas
    # si se abre una base de datos creada con versiones anteriores.
    "rms_original_min",
    "pico_pico_original_min",
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
    "energia_banda_0_30_original_max",
}


def inicializar_bd_patrones_normales(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patrones_normales_dtw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            tipo_patron TEXT,
            eje TEXT,

            duracion_s_min REAL,
            duracion_s_max REAL,

            rms_original_max REAL,

            pico_pico_original_max REAL,

            rizo_pico_pico_original_max REAL,

            dtw_distancia_max REAL,

            offset_abs_original_max REAL,

            forma_normalizada_blob BLOB,
            segmento_original_blob BLOB,
            segmento_filtrado_blob BLOB,

            frecuencias_original_blob BLOB,
            espectro_original_blob BLOB,

            frecuencias_filtrado_blob BLOB,
            espectro_filtrado_blob BLOB
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS segmentos_patrones_dtw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patron_id INTEGER,
            tipo_patron TEXT,
            eje TEXT,
            huella_segmento TEXT,

            segmento_original_blob BLOB,
            segmento_filtrado_blob BLOB,
            forma_normalizada_blob BLOB,

            frecuencias_original_blob BLOB,
            espectro_original_blob BLOB,

            frecuencias_filtrado_blob BLOB,
            espectro_filtrado_blob BLOB
        )
    """)

    for columna in COLUMNAS_ESCALARES_BD:
        if columna in ("tipo_patron", "eje"):
            definicion = "TEXT"
        else:
            definicion = "REAL"

        asegurar_columna_bd(conn, "patrones_normales_dtw", columna, definicion)

    for columna in COLUMNAS_BLOB_BD:
        asegurar_columna_bd(conn, "patrones_normales_dtw", columna, "BLOB")

    asegurar_columna_bd(
        conn,
        "segmentos_patrones_dtw",
        "huella_segmento",
        "TEXT"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_segmentos_patrones_huella "
        "ON segmentos_patrones_dtw(tipo_patron, eje, huella_segmento)"
    )

    conn.commit()


def calcular_espectro_segmento_1d(x):
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N < 4:
        f = np.array([0.0])
        espectro = np.array([0.0])
        return f, espectro

    ventana = np.hanning(N)
    ganancia = np.mean(ventana)

    f = np.fft.rfftfreq(N, d=1 / Fs)

    x0 = x - np.mean(x)
    xw = x0 * ventana

    espectro = np.abs(np.fft.rfft(xw)) / (N * ganancia)

    if len(espectro) > 2:
        espectro[1:-1] *= 2

    return f, espectro


def calcular_energia_bandas_desde_espectro(f, espectro):
    """
    Calcula la energía por bandas a partir del espectro de amplitud.
    Se guarda únicamente el valor máximo acumulado en la BD.
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


def calcular_energia_bandas_1d(x):
    f, espectro = calcular_espectro_segmento_1d(x)
    return calcular_energia_bandas_desde_espectro(f, espectro), f, espectro


def calcular_rizo_pico_pico_1d(seg_original, seg_filtrada):
    seg_original = np.asarray(seg_original, dtype=float)
    seg_filtrada = np.asarray(seg_filtrada, dtype=float)

    n = min(len(seg_original), len(seg_filtrada))

    if n == 0:
        return 0.0

    rizo = seg_original[:n] - seg_filtrada[:n]
    return float(np.max(rizo) - np.min(rizo))


def combinar_maximos_dict(lista_dicts, columnas):
    resultado = {}

    for col in columnas:
        valores = []

        for d in lista_dicts:
            if d is None:
                continue
            valor = d.get(col)
            if valor is not None:
                valores.append(valor)

        resultado[col] = max_bd(valores)

    return resultado


def calcular_features_patron_1d(x):
    """
    Calcula únicamente las características temporales que se guardan en la BD:
    RMS original y pico-pico original.

    """
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return {
            "rms": 0.0,
            "pico_pico": 0.0,
        }, np.array([0.0]), np.array([0.0])

    f, espectro = calcular_espectro_segmento_1d(x)

    features = {
        "rms": float(np.sqrt(np.mean(x * x))),
        "pico_pico": float(np.max(x) - np.min(x)),
    }

    return features, f, espectro


def min_max_bd(valores):
    valores = np.asarray(valores, dtype=float)

    if len(valores) == 0:
        return None, None

    return float(np.min(valores)), float(np.max(valores))


def max_bd(valores):
    valores = np.asarray(valores, dtype=float)

    if len(valores) == 0:
        return None

    return float(np.max(valores))


def calcular_forma_normalizada_bd(x):
    """
    Genera una forma normalizada de longitud LONGITUD_DTW para guardar como patrón.
    """
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return np.zeros(LONGITUD_DTW)

    return z_normalizar_1d(remuestrear_1d(x, LONGITUD_DTW))


def indice_segmento_mas_largo(segmentos):
    if len(segmentos) == 0:
        return None

    longitudes = [fin - ini + 1 for ini, fin in segmentos]
    return int(np.argmax(longitudes))


def preparar_estadisticas_segmentos_bd(acc_sin_filtrar, acc_filtrada, segmentos, eje):
    """
    Calcula características para reposo u oscilación.
    """
    idx_eje = IDX_EJE[eje]

    rms_original = []
    pico_pico_original = []
    rizo_pico_pico_original = []
    energias_bandas_original = []
    offset_abs_original = []

    for ini, fin in segmentos:
        seg_original = acc_sin_filtrar[ini:fin + 1, idx_eje]
        seg_filtrada = acc_filtrada[ini:fin + 1, idx_eje]

        features_original, _, _ = calcular_features_patron_1d(seg_original)
        energias_original, _, _ = calcular_energia_bandas_1d(seg_original)

        rms_original.append(features_original["rms"])
        pico_pico_original.append(features_original["pico_pico"])
        rizo_pico_pico_original.append(
            calcular_rizo_pico_pico_1d(seg_original, seg_filtrada)
        )
        energias_bandas_original.append(energias_original)

        if len(seg_original) > 0:
            offset_abs_original.append(float(np.max(np.abs(seg_original))))

    idx_rep_lista = indice_segmento_mas_largo(segmentos)

    if idx_rep_lista is None:
        segmento_original_rep = np.array([], dtype=float)
        segmento_filtrado_rep = np.array([], dtype=float)
    else:
        ini_rep, fin_rep = segmentos[idx_rep_lista]
        segmento_original_rep = acc_sin_filtrar[ini_rep:fin_rep + 1, idx_eje]
        segmento_filtrado_rep = acc_filtrada[ini_rep:fin_rep + 1, idx_eje]

    f_original_rep, espectro_original_rep = calcular_espectro_segmento_1d(
        segmento_original_rep
    )

    f_filtrado_rep, espectro_filtrado_rep = calcular_espectro_segmento_1d(
        segmento_filtrado_rep
    )

    return {
        "rms_original_max": max_bd(rms_original),
        "pico_pico_original_max": max_bd(pico_pico_original),
        "rizo_pico_pico_original_max": max_bd(rizo_pico_pico_original),
        "energias_bandas_original": combinar_maximos_dict(
            energias_bandas_original,
            COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD
        ),
        "offset_abs_original_max": max_bd(offset_abs_original),

        "segmento_original_rep": segmento_original_rep,
        "segmento_filtrado_rep": segmento_filtrado_rep,
        "forma_representante": calcular_forma_normalizada_bd(segmento_filtrado_rep),
        "f_original_rep": f_original_rep,
        "espectro_original_rep": espectro_original_rep,
        "f_filtrado_rep": f_filtrado_rep,
        "espectro_filtrado_rep": espectro_filtrado_rep,
    }


def valor_min_existente_nuevo(valor_existente, valor_nuevo):
    if valor_existente is None:
        return valor_nuevo
    if valor_nuevo is None:
        return valor_existente
    return min(float(valor_existente), float(valor_nuevo))


def valor_max_existente_nuevo(valor_existente, valor_nuevo):
    if valor_existente is None:
        return valor_nuevo
    if valor_nuevo is None:
        return valor_existente
    return max(float(valor_existente), float(valor_nuevo))


def obtener_umbral_fusion_bd(eje):
    if UMBRAL_FUSION_PATRONES_BD is None:
        return DISTANCIA_CLUSTER_DTW[eje]

    if isinstance(UMBRAL_FUSION_PATRONES_BD, dict):
        return UMBRAL_FUSION_PATRONES_BD.get(eje, DISTANCIA_CLUSTER_DTW[eje])

    return float(UMBRAL_FUSION_PATRONES_BD)


def calcular_metricas_array_1d(x):
    """
    Calcula las mismas métricas físicas que se usan junto con la forma DTW
    en la clasificación interna del experimento.

    Se usa también para comparar contra los patrones guardados en la BD,
    de forma que la fusión con la BD no dependa solo de la forma normalizada.
    """
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return {
            "duracion_s": 0.0,
            "rms": 0.0,
            "pico_abs": 0.0,
            "pico_firmado": 0.0,
            "signo_pico": 0.0,
            "ini": 0,
            "fin": 0,
        }

    idx_pico = int(np.argmax(np.abs(x)))
    pico_firmado = float(x[idx_pico])

    return {
        "duracion_s": float(len(x) / Fs),
        "rms": float(np.sqrt(np.mean(x * x))),
        "pico_abs": float(np.max(np.abs(x))),
        "pico_firmado": pico_firmado,
        "signo_pico": float(np.sign(pico_firmado)),
        "ini": 0,
        "fin": int(len(x) - 1),
    }


def leer_segmentos_individuales_patron_bd(cur, patron_id):
    """
    Lee todos los segmentos individuales asociados a un patrón de la BD.
    Estos segmentos permiten comparar un grupo nuevo contra el grupo completo
    guardado, no solo contra su representante.
    """
    cur.execute(
        """
        SELECT
            segmento_original_blob,
            segmento_filtrado_blob,
            forma_normalizada_blob,
            frecuencias_original_blob,
            espectro_original_blob,
            frecuencias_filtrado_blob,
            espectro_filtrado_blob
        FROM segmentos_patrones_dtw
        WHERE patron_id = ?
        ORDER BY id ASC
        """,
        (patron_id,)
    )

    segmentos = []

    for fila in cur.fetchall():
        (
            segmento_original_blob,
            segmento_filtrado_blob,
            forma_normalizada_blob,
            frecuencias_original_blob,
            espectro_original_blob,
            frecuencias_filtrado_blob,
            espectro_filtrado_blob
        ) = fila

        segmentos.append({
            "segmento_original": _blob_a_array_seguro(segmento_original_blob),
            "segmento_filtrado": _blob_a_array_seguro(segmento_filtrado_blob),
            "forma_normalizada": _blob_a_array_seguro(forma_normalizada_blob),
            "frecuencias_original": _blob_a_array_seguro(frecuencias_original_blob),
            "espectro_original": _blob_a_array_seguro(espectro_original_blob),
            "frecuencias_filtrado": _blob_a_array_seguro(frecuencias_filtrado_blob),
            "espectro_filtrado": _blob_a_array_seguro(espectro_filtrado_blob),
        })

    return segmentos


def buscar_patron_parecido_bd(
        cur,
        tipo_patron,
        eje,
        forma_representante,
        metrica_representante=None,
        formas_nuevas=None,
        metricas_nuevas=None
):
    """
    Busca el patrón de la BD más parecido al segmento nuevo.
    """
    cur.execute(
        """
        SELECT id, forma_normalizada_blob, segmento_filtrado_blob
        FROM patrones_normales_dtw
        WHERE tipo_patron = ? AND eje = ?
        ORDER BY id ASC
        """,
        (tipo_patron, eje)
    )

    filas = cur.fetchall()

    if len(filas) == 0:
        return None, None

    if tipo_patron in ("reposo", "oscilatorio"):
        return int(filas[0][0]), 0.0

    forma_representante = np.asarray(forma_representante, dtype=float)

    if len(forma_representante) == 0:
        return None, None

    if metrica_representante is None:
        metrica_representante = {
            "duracion_s": 0.0,
            "rms": 0.0,
            "pico_abs": 0.0,
            "pico_firmado": 0.0,
            "signo_pico": 0.0,
            "ini": 0,
            "fin": 0,
        }

    umbral = obtener_umbral_fusion_bd(eje)

    mejor_id = None
    mejor_distancia = None

    for id_patron, blob_forma, blob_segmento_filtrado in filas:
        id_patron = int(id_patron)

        forma_bd = _blob_a_array_seguro(blob_forma)
        segmento_filtrado_bd = _blob_a_array_seguro(blob_segmento_filtrado)

        if len(forma_bd) == 0:
            continue

        metrica_bd = calcular_metricas_array_1d(segmento_filtrado_bd)

        d = distancia_formas_1d(
            forma_representante,
            forma_bd,
            metrica_representante,
            metrica_bd
        )

        if mejor_distancia is None or d < mejor_distancia:
            mejor_distancia = float(d)
            mejor_id = id_patron

    if mejor_id is None:
        return None, None

    if mejor_distancia <= umbral:
        return mejor_id, mejor_distancia

    return None, mejor_distancia


def insertar_patron_normal_bd(cur, datos_patron):
    columnas = COLUMNAS_ESCALARES_BD + COLUMNAS_BLOB_BD

    valores = []

    for col in COLUMNAS_ESCALARES_BD:
        valores.append(datos_patron.get(col))

    for col in COLUMNAS_BLOB_BD:
        valores.append(array_a_blob(datos_patron.get(col, np.array([], dtype=float))))

    placeholders = ", ".join(["?"] * len(columnas))
    columnas_sql = ", ".join(columnas)

    cur.execute(
        f"INSERT INTO patrones_normales_dtw ({columnas_sql}) VALUES ({placeholders})",
        valores
    )

    return cur.lastrowid


def calcular_huella_segmento_bd(
        segmento_original,
        segmento_filtrado,
        forma_normalizada
):
    """Devuelve una huella reproducible del contenido numérico del segmento."""
    h = hashlib.sha256()

    for array in (segmento_original, segmento_filtrado, forma_normalizada):
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes(order="C"))

    return h.hexdigest()


def insertar_segmento_patron_bd(
        cur,
        patron_id,
        tipo_patron,
        eje,
        segmento_original,
        segmento_filtrado,
        forma_normalizada,
        f_original,
        espectro_original,
        f_filtrado,
        espectro_filtrado
):
    """
    Guarda un segmento individual asociado a un patrón.
    """
    huella = calcular_huella_segmento_bd(
        segmento_original=segmento_original,
        segmento_filtrado=segmento_filtrado,
        forma_normalizada=forma_normalizada
    )

    cur.execute(
        """
        SELECT id
        FROM segmentos_patrones_dtw
        WHERE tipo_patron = ?
          AND eje = ?
          AND huella_segmento = ?
        LIMIT 1
        """,
        (tipo_patron, eje, huella)
    )

    if cur.fetchone() is not None:
        return False

    cur.execute(
        """
        INSERT INTO segmentos_patrones_dtw (
            patron_id, tipo_patron, eje, huella_segmento,
            segmento_original_blob, segmento_filtrado_blob, forma_normalizada_blob,
            frecuencias_original_blob, espectro_original_blob,
            frecuencias_filtrado_blob, espectro_filtrado_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patron_id,
            tipo_patron,
            eje,
            huella,
            array_a_blob(segmento_original),
            array_a_blob(segmento_filtrado),
            array_a_blob(forma_normalizada),
            array_a_blob(f_original),
            array_a_blob(espectro_original),
            array_a_blob(f_filtrado),
            array_a_blob(espectro_filtrado),
        )
    )

    return True


def recalcular_representante_patron_bd(cur, id_patron):
    """
    Recalcula el representante real de un patrón usando todos los segmentos
    individuales guardados en segmentos_patrones_dtw.

    Para desplazamiento se escoge el medoide según la misma distancia combinada
    que usa la clasificación interna. Para reposo/oscilatorio se mantiene el
    criterio simple de escoger el segmento más largo, ya que son grupos fijos
    que no pasan por DTW.
    """
    cur.execute(
        """
        SELECT tipo_patron, eje
        FROM patrones_normales_dtw
        WHERE id = ?
        """,
        (id_patron,)
    )

    fila_patron = cur.fetchone()

    if fila_patron is None:
        return

    tipo_patron, eje = fila_patron
    segmentos = leer_segmentos_individuales_patron_bd(cur, id_patron)

    if len(segmentos) == 0:
        return

    idx_representante = None
    dtw_distancia_max_recalculada = None

    if tipo_patron == "desplazamiento":
        indices_validos = []
        formas_validas = []
        metricas_validas = []

        for i, seg in enumerate(segmentos):
            forma = np.asarray(seg.get("forma_normalizada", []), dtype=float)
            segmento_filtrado = np.asarray(seg.get("segmento_filtrado", []), dtype=float)

            if len(forma) == 0:
                continue

            indices_validos.append(i)
            formas_validas.append(forma)
            metricas_validas.append(calcular_metricas_array_1d(segmento_filtrado))

        if len(indices_validos) == 0:
            idx_representante = int(np.argmax([
                len(seg.get("segmento_filtrado", []))
                for seg in segmentos
            ]))
        elif len(indices_validos) == 1:
            idx_representante = indices_validos[0]
            dtw_distancia_max_recalculada = 0.0
        else:
            D = calcular_matriz_distancias_dtw_1d(
                formas_validas,
                metricas_validas
            )
            idx_local_representante = obtener_medoide_cluster(
                np.arange(len(formas_validas)),
                D
            )
            idx_representante = indices_validos[int(idx_local_representante)]

            distancias_a_representante = D[:, int(idx_local_representante)]
            dtw_distancia_max_recalculada = float(np.max(distancias_a_representante))

    else:
        idx_representante = int(np.argmax([
            len(seg.get("segmento_filtrado", []))
            for seg in segmentos
        ]))
        dtw_distancia_max_recalculada = None

    if idx_representante is None:
        return

    representante = segmentos[idx_representante]

    cur.execute(
        """
        UPDATE patrones_normales_dtw
        SET
            forma_normalizada_blob = ?,
            segmento_original_blob = ?,
            segmento_filtrado_blob = ?,
            frecuencias_original_blob = ?,
            espectro_original_blob = ?,
            frecuencias_filtrado_blob = ?,
            espectro_filtrado_blob = ?,
            dtw_distancia_max = ?
        WHERE id = ?
        """,
        (
            array_a_blob(representante.get("forma_normalizada", np.array([], dtype=float))),
            array_a_blob(representante.get("segmento_original", np.array([], dtype=float))),
            array_a_blob(representante.get("segmento_filtrado", np.array([], dtype=float))),
            array_a_blob(representante.get("frecuencias_original", np.array([], dtype=float))),
            array_a_blob(representante.get("espectro_original", np.array([], dtype=float))),
            array_a_blob(representante.get("frecuencias_filtrado", np.array([], dtype=float))),
            array_a_blob(representante.get("espectro_filtrado", np.array([], dtype=float))),
            dtw_distancia_max_recalculada,
            id_patron,
        )
    )


def actualizar_patron_normal_bd(cur, id_patron, datos_patron, distancia_representantes=None):
    """
    Fusiona un patrón nuevo con uno existente actualizando sus rangos normales.
    """
    columnas_actualizables = []

    for col_min, col_max in PARES_MIN_MAX_BD:
        columnas_actualizables.append(col_min)
        columnas_actualizables.append(col_max)

    columnas_actualizables += COLUMNAS_MAX_BD

    columnas_actualizables = list(dict.fromkeys(columnas_actualizables))

    columnas_select = ", ".join(columnas_actualizables)

    cur.execute(
        f"SELECT {columnas_select} FROM patrones_normales_dtw WHERE id = ?",
        (id_patron,)
    )

    fila = cur.fetchone()

    if fila is None:
        return

    existentes = dict(zip(columnas_actualizables, fila))
    nuevos_valores = {}

    for col_min, col_max in PARES_MIN_MAX_BD:
        nuevos_valores[col_min] = valor_min_existente_nuevo(
            existentes.get(col_min),
            datos_patron.get(col_min)
        )
        nuevos_valores[col_max] = valor_max_existente_nuevo(
            existentes.get(col_max),
            datos_patron.get(col_max)
        )

    for col in COLUMNAS_MAX_BD:
        nuevos_valores[col] = valor_max_existente_nuevo(
            existentes.get(col),
            datos_patron.get(col)
        )

    if datos_patron.get("tipo_patron") == "desplazamiento" and distancia_representantes is not None:
        nuevos_valores["dtw_distancia_max"] = valor_max_existente_nuevo(
            nuevos_valores.get("dtw_distancia_max"),
            distancia_representantes
        )

    set_sql = ", ".join([f"{col} = ?" for col in nuevos_valores.keys()])
    valores = list(nuevos_valores.values()) + [id_patron]

    cur.execute(
        f"UPDATE patrones_normales_dtw SET {set_sql} WHERE id = ?",
        valores
    )


def guardar_o_actualizar_patron_normal_bd(cur, datos_patron):
    """
    Inserta el patrón si no existe uno parecido.
    Si existe uno parecido, actualiza sus mínimos/máximos.

    En desplazamientos, la búsqueda de parecido compara el segmento nuevo con
    el representante de cada grupo BD usando distancia_formas_1d(), igual que
    cuando se comparan dos segmentos dentro del experimento.
    """
    tipo_patron = datos_patron.get("tipo_patron")
    eje = datos_patron.get("eje")
    forma_representante = datos_patron.get("forma_normalizada_blob", np.array([], dtype=float))
    metrica_representante = datos_patron.get("metrica_representante")
    formas_segmentos = datos_patron.get("formas_segmentos")
    metricas_segmentos = datos_patron.get("metricas_segmentos")

    id_parecido, distancia_representantes = buscar_patron_parecido_bd(
        cur=cur,
        tipo_patron=tipo_patron,
        eje=eje,
        forma_representante=forma_representante,
        metrica_representante=metrica_representante
    )

    if id_parecido is None:
        nuevo_id = insertar_patron_normal_bd(cur, datos_patron)
        return "insertado", nuevo_id, distancia_representantes

    actualizar_patron_normal_bd(
        cur=cur,
        id_patron=id_parecido,
        datos_patron=datos_patron,
        distancia_representantes=distancia_representantes
    )

    return "actualizado", id_parecido, distancia_representantes


def crear_datos_patron_bd(
        tipo_patron,
        eje,
        duracion_min=None,
        duracion_max=None,
        rms_ori_max=None,
        pp_ori_max=None,
        dtw_max=None,
        offset_abs_original_max=None,
        forma_representante=None,
        segmento_original_rep=None,
        segmento_filtrado_rep=None,
        f_original_rep=None,
        espectro_original_rep=None,
        f_filtrado_rep=None,
        espectro_filtrado_rep=None,
        rizo_pico_pico_original_max=None,
        energias_bandas_original=None
):
    """
    Crea el diccionario de inserción/actualización de la BD.
    """
    datos = {
        "tipo_patron": tipo_patron,
        "eje": eje,

        "duracion_s_min": duracion_min,
        "duracion_s_max": duracion_max,

        "rms_original_max": rms_ori_max,

        "pico_pico_original_max": pp_ori_max,

        "rizo_pico_pico_original_max": rizo_pico_pico_original_max,
        "dtw_distancia_max": dtw_max,
        "offset_abs_original_max": offset_abs_original_max,

        "forma_normalizada_blob": forma_representante if forma_representante is not None else np.array([], dtype=float),
        "segmento_original_blob": segmento_original_rep if segmento_original_rep is not None else np.array([], dtype=float),
        "segmento_filtrado_blob": segmento_filtrado_rep if segmento_filtrado_rep is not None else np.array([], dtype=float),

        "frecuencias_original_blob": f_original_rep if f_original_rep is not None else np.array([], dtype=float),
        "espectro_original_blob": espectro_original_rep if espectro_original_rep is not None else np.array([], dtype=float),

        "frecuencias_filtrado_blob": f_filtrado_rep if f_filtrado_rep is not None else np.array([], dtype=float),
        "espectro_filtrado_blob": espectro_filtrado_rep if espectro_filtrado_rep is not None else np.array([], dtype=float),
    }

    if energias_bandas_original is not None:
        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            datos[col] = energias_bandas_original.get(col)
    else:
        for col in COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD:
            datos[col] = None

    if tipo_patron == "senyal_completa":
        for col in [
            "duracion_s_min", "duracion_s_max",
            "rms_original_max",
            "pico_pico_original_max",
            "rizo_pico_pico_original_max",
            "dtw_distancia_max",
            "offset_abs_original_max",
        ]:
            datos[col] = None

    elif tipo_patron in ("oscilatorio", "reposo"):
        datos["duracion_s_min"] = None
        datos["duracion_s_max"] = None
        datos["dtw_distancia_max"] = None

        if tipo_patron != "reposo":
            datos["offset_abs_original_max"] = None

    elif tipo_patron == "desplazamiento":
        datos["offset_abs_original_max"] = None

    return datos


def leer_representantes_bd_para_guardado_dtw(cur, eje):
    """
    Lee los grupos de desplazamiento ya existentes en la BD para el eje dado.
    Se usa al principio de la clasificación global, antes de insertar los
    segmentos del experimento actual.
    """
    cur.execute(
        """
        SELECT
            id,
            forma_normalizada_blob,
            segmento_filtrado_blob
        FROM patrones_normales_dtw
        WHERE tipo_patron = 'desplazamiento' AND eje = ?
        ORDER BY id ASC
        """,
        (eje,)
    )

    grupos_bd = []

    for id_patron, forma_blob, segmento_filtrado_blob in cur.fetchall():
        forma = _blob_a_array_seguro(forma_blob)
        segmento_filtrado = _blob_a_array_seguro(segmento_filtrado_blob)

        if len(forma) == 0:
            continue

        grupos_bd.append({
            "tipo_item": "grupo_bd",
            "patron_id": int(id_patron),
            "etiqueta": f"G{int(id_patron)}",
            "forma": forma,
            "metrica": calcular_metricas_array_1d(segmento_filtrado),
        })

    return grupos_bd


def crear_segmentos_individuales_desplazamiento_bd(
        acc_sin_filtrar,
        acc_filtrada,
        info_eje,
        eje
):
    """
    Prepara todos los segmentos normales del experimento actual como elementos
    individuales para clasificarlos globalmente contra la BD.
    """
    segmentos_validos = info_eje.get("segmentos_validos", [])
    etiquetas_validas = info_eje.get("etiquetas", [])
    formas = info_eje.get("formas", [])
    metricas = info_eje.get("metricas", [])

    idx_eje = IDX_EJE[eje]
    segmentos_individuales = []

    for idx_seg, segmento in enumerate(segmentos_validos):
        if idx_seg >= len(formas):
            continue

        forma = np.asarray(formas[idx_seg], dtype=float)

        if len(forma) == 0:
            continue

        ini, fin = segmento
        etiqueta_seg = (
            etiquetas_validas[idx_seg]
            if idx_seg < len(etiquetas_validas)
            else f"{eje.upper()}?"
        )

        seg_original = acc_sin_filtrar[ini:fin + 1, idx_eje]
        seg_filtrada = acc_filtrada[ini:fin + 1, idx_eje]

        metrica_seg = metricas[idx_seg] if idx_seg < len(metricas) else None

        if metrica_seg is None:
            metrica_seg = calcular_metricas_array_1d(seg_filtrada)

        f_original_seg, espectro_original_seg = calcular_espectro_segmento_1d(seg_original)
        f_filtrado_seg, espectro_filtrado_seg = calcular_espectro_segmento_1d(seg_filtrada)

        segmentos_individuales.append({
            "tipo_item": "segmento",
            "patron_id": None,
            "ini": int(ini),
            "fin": int(fin),
            "etiqueta": etiqueta_seg,
            "forma": forma,
            "metrica": metrica_seg,
            "segmento_original": seg_original,
            "segmento_filtrado": seg_filtrada,
            "forma_normalizada": forma,
            "f_original": f_original_seg,
            "espectro_original": espectro_original_seg,
            "f_filtrado": f_filtrado_seg,
            "espectro_filtrado": espectro_filtrado_seg,
        })

    return segmentos_individuales


def crear_datos_patron_desplazamiento_desde_segmentos_bd(
        eje,
        segmentos_individuales,
        dtw_max=None,
        idx_representante_local=None
):
    """
    Crea la fila de patrón de desplazamiento a partir de uno o varios segmentos.
    Si hay varios segmentos, el representante inicial es el medoide del grupo.
    """
    if len(segmentos_individuales) == 0:
        return None

    formas = [np.asarray(seg["forma_normalizada"], dtype=float) for seg in segmentos_individuales]
    metricas = [seg.get("metrica") for seg in segmentos_individuales]

    if idx_representante_local is None:
        if len(formas) == 1:
            idx_representante_local = 0
            if dtw_max is None:
                dtw_max = 0.0
        else:
            D = calcular_matriz_distancias_dtw_1d(formas, metricas)
            idx_representante_local = obtener_medoide_cluster(
                np.arange(len(formas)),
                D
            )
            if dtw_max is None:
                dtw_max = float(np.max(D[:, int(idx_representante_local)]))

    idx_representante_local = int(idx_representante_local)
    representante = segmentos_individuales[idx_representante_local]

    duraciones = []
    rms_original = []
    pico_pico_original = []
    rizo_pico_pico_original = []
    energias_bandas_original = []

    for seg in segmentos_individuales:
        seg_original = seg["segmento_original"]
        seg_filtrada = seg["segmento_filtrado"]

        features_original, _, _ = calcular_features_patron_1d(seg_original)
        energias_original, _, _ = calcular_energia_bandas_1d(seg_original)

        duraciones.append(float(len(seg_original) / Fs))
        rms_original.append(features_original["rms"])
        pico_pico_original.append(features_original["pico_pico"])
        rizo_pico_pico_original.append(
            calcular_rizo_pico_pico_1d(seg_original, seg_filtrada)
        )
        energias_bandas_original.append(energias_original)

    dur_min, dur_max = min_max_bd(duraciones)
    rms_ori_max = max_bd(rms_original)
    pp_ori_max = max_bd(pico_pico_original)

    datos_patron = crear_datos_patron_bd(
        tipo_patron="desplazamiento",
        eje=eje,
        duracion_min=dur_min,
        duracion_max=dur_max,
        rms_ori_max=rms_ori_max,
        pp_ori_max=pp_ori_max,
        dtw_max=dtw_max,
        offset_abs_original_max=None,
        forma_representante=representante["forma_normalizada"],
        segmento_original_rep=representante["segmento_original"],
        segmento_filtrado_rep=representante["segmento_filtrado"],
        f_original_rep=representante["f_original"],
        espectro_original_rep=representante["espectro_original"],
        f_filtrado_rep=representante["f_filtrado"],
        espectro_filtrado_rep=representante["espectro_filtrado"],
        rizo_pico_pico_original_max=max_bd(rizo_pico_pico_original),
        energias_bandas_original=combinar_maximos_dict(
            energias_bandas_original,
            COLUMNAS_ENERGIA_BANDAS_ORIGINAL_BD
        )
    )

    datos_patron["metrica_representante"] = representante.get("metrica")

    return datos_patron


def insertar_segmentos_individuales_en_patron_bd(
        cur,
        id_patron,
        tipo_patron,
        eje,
        segmentos_individuales
):
    """
    Inserta en segmentos_patrones_dtw todos los segmentos individuales asignados
    a un patrón y devuelve cuántos se han guardado.
    """
    n_guardados = 0

    for seg in segmentos_individuales:
        insertado = insertar_segmento_patron_bd(
            cur=cur,
            patron_id=id_patron,
            tipo_patron=tipo_patron,
            eje=eje,
            segmento_original=seg["segmento_original"],
            segmento_filtrado=seg["segmento_filtrado"],
            forma_normalizada=seg["forma_normalizada"],
            f_original=seg["f_original"],
            espectro_original=seg["espectro_original"],
            f_filtrado=seg["f_filtrado"],
            espectro_filtrado=seg["espectro_filtrado"],
        )

        if insertado:
            n_guardados += 1

    return n_guardados


def clasificar_segmentos_con_bd_por_dendrograma(
        cur,
        eje,
        segmentos_individuales
):
    """
    Clasifica globalmente los segmentos del experimento contra los grupos de la
    BD usando la misma matriz de distancias y el mismo dendrograma que se
    visualiza en la pestaña BD.

    Reglas:
    - Si un cluster contiene un único grupo BD, todos sus segmentos se asignan a
      ese grupo.
    - Si un cluster contiene varios grupos BD, cada segmento se asigna al grupo
      BD más cercano dentro de ese cluster.
    - Si un cluster no contiene ningún grupo BD, sus segmentos crean un grupo BD
      nuevo conjunto.
    """
    if len(segmentos_individuales) == 0:
        return []

    grupos_bd = leer_representantes_bd_para_guardado_dtw(cur, eje)
    items = list(grupos_bd) + list(segmentos_individuales)
    n_grupos_bd = len(grupos_bd)
    umbral = obtener_umbral_fusion_bd(eje)

    if len(items) == 1:
        labels = np.array([1], dtype=int)
        D = np.zeros((1, 1), dtype=float)
    else:
        D = calcular_matriz_items_dtw(items)
        D_condensada = squareform(D, checks=False)
        Z = linkage(D_condensada, method=LINKAGE_DTW)
        labels = fcluster(Z, t=umbral, criterion="distance")

    clusters = {}

    for idx_item, label in enumerate(labels):
        label = int(label)

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(idx_item)

    operaciones = []

    for label in sorted(clusters.keys()):
        indices_cluster = clusters[label]

        indices_bd = [idx for idx in indices_cluster if idx < n_grupos_bd]
        indices_segmentos_global = [idx for idx in indices_cluster if idx >= n_grupos_bd]

        if len(indices_segmentos_global) == 0:
            continue

        segmentos_cluster = [items[idx] for idx in indices_segmentos_global]

        if len(indices_bd) == 0:
            operaciones.append({
                "accion": "crear",
                "id_patron": None,
                "segmentos": segmentos_cluster,
                "distancia_max": None,
            })
            continue

        if len(indices_bd) == 1:
            id_patron = int(items[indices_bd[0]]["patron_id"])
            distancias = [float(D[idx_seg, indices_bd[0]]) for idx_seg in indices_segmentos_global]
            operaciones.append({
                "accion": "actualizar",
                "id_patron": id_patron,
                "segmentos": segmentos_cluster,
                "distancia_max": max_bd(distancias),
            })
            continue

        asignaciones = {}
        distancias_por_grupo = {}

        for idx_seg_global in indices_segmentos_global:
            idx_bd_cercano = min(
                indices_bd,
                key=lambda idx_bd: D[idx_seg_global, idx_bd]
            )
            id_patron = int(items[idx_bd_cercano]["patron_id"])
            asignaciones.setdefault(id_patron, []).append(items[idx_seg_global])
            distancias_por_grupo.setdefault(id_patron, []).append(float(D[idx_seg_global, idx_bd_cercano]))

        for id_patron in sorted(asignaciones.keys()):
            operaciones.append({
                "accion": "actualizar",
                "id_patron": id_patron,
                "segmentos": asignaciones[id_patron],
                "distancia_max": max_bd(distancias_por_grupo.get(id_patron, [])),
            })

    return operaciones


def guardar_senyal_completa_bd(cur, acc_sin_filtrar):
    """
    Guarda/actualiza información espectral general de la señal completa.
    No representa un movimiento ni un segmento, solo el máximo de energía por
    bandas de cada eje para el programa/carpeta procesado.
    """
    for eje in ["x", "y", "z"]:
        idx_eje = IDX_EJE[eje]
        senyal = acc_sin_filtrar[:, idx_eje]

        energias, f_original, espectro_original = calcular_energia_bandas_1d(senyal)

        datos_patron = crear_datos_patron_bd(
            tipo_patron="senyal_completa",
            eje=eje,
            energias_bandas_original=energias,
            segmento_original_rep=senyal,
            segmento_filtrado_rep=np.array([], dtype=float),
            f_original_rep=f_original,
            espectro_original_rep=espectro_original,
            f_filtrado_rep=np.array([], dtype=float),
            espectro_filtrado_rep=np.array([], dtype=float)
        )

        cur.execute(
            """
            SELECT id
            FROM patrones_normales_dtw
            WHERE tipo_patron = 'senyal_completa' AND eje = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (eje,)
        )

        fila = cur.fetchone()

        if fila is None:
            insertar_patron_normal_bd(cur, datos_patron)
        else:
            actualizar_patron_normal_bd(cur, int(fila[0]), datos_patron)


def anadir_etiquetas_a_resumen_bd(resumen, id_patron, etiquetas):
    """
    Añade etiquetas de segmentos a un resumen por ID real de grupo/patrón de la BD.
    Evita repetir etiquetas dentro del mismo grupo si aparecen duplicadas.
    """
    if id_patron is None:
        return

    id_patron = int(id_patron)

    if id_patron not in resumen:
        resumen[id_patron] = []

    for etiqueta in etiquetas:
        if etiqueta is None:
            continue

        etiqueta = str(etiqueta)

        if etiqueta not in resumen[id_patron]:
            resumen[id_patron].append(etiqueta)


def imprimir_resumen_segmentos_bd(titulo, resumen):
    """
    Imprime los segmentos agrupados por el ID real del grupo/patrón en la BD.
    """
    print(f"\n{titulo}:")

    if len(resumen) == 0:
        print("Cap.")
        return

    for id_patron in sorted(resumen.keys()):
        etiquetas = resumen[id_patron]
        print(f"Grup {id_patron}: {', '.join(etiquetas)}")


def obtener_ids_patrones_bd(cur):
    """
    Devuelve los ID de patrones que existen en la BD en este momento.

    Se usa para distinguir entre:
    - grupos que ya existían antes de empezar el análisis actual;
    - grupos creados durante esta ejecución.
    """
    cur.execute("SELECT id FROM patrones_normales_dtw")
    return set(int(fila[0]) for fila in cur.fetchall())


def guardar_patrones_normales_dtw_bd(
        acc_sin_filtrar,
        acc_filtrada,
        info_clasificacion_dtw,
        permitir_borrado_bd=True
):
    """
    Guarda patrones normales en SQLite de forma acumulativa.
    """
    if info_clasificacion_dtw is None:
        print("\nNo hi ha classificació DTW. No es guarda la base de dades.")
        return

    conn = crear_conexion_bd_patrones()
    inicializar_bd_patrones_normales(conn)

    cur = conn.cursor()

    if BORRAR_BD_PATRONES_ANTES_DE_GUARDAR and permitir_borrado_bd:
        cur.execute("DELETE FROM segmentos_patrones_dtw")
        cur.execute("DELETE FROM patrones_normales_dtw")

        cur.execute("DELETE FROM sqlite_sequence WHERE name = 'segmentos_patrones_dtw'")
        cur.execute("DELETE FROM sqlite_sequence WHERE name = 'patrones_normales_dtw'")

        conn.commit()

    n_insertados = 0
    n_segmentos_guardados = 0

    ids_patrones_existentes_inicio = obtener_ids_patrones_bd(cur)
    ids_patrones_creados_esta_ejecucion = set()
    ids_patrones_existentes_actualizados = set()

    resumen_segmentos_anadidos = {}
    resumen_segmentos_nuevos = {}

    guardar_senyal_completa_bd(cur, acc_sin_filtrar)

    for eje in EJES_CLASIFICACION_DTW:
        info_eje = info_clasificacion_dtw.get(eje)

        if info_eje is None:
            continue

        grupos_fijos = info_eje.get("grupos_fijos", [])

        segmentos_individuales = crear_segmentos_individuales_desplazamiento_bd(
            acc_sin_filtrar=acc_sin_filtrar,
            acc_filtrada=acc_filtrada,
            info_eje=info_eje,
            eje=eje
        )

        operaciones = clasificar_segmentos_con_bd_por_dendrograma(
            cur=cur,
            eje=eje,
            segmentos_individuales=segmentos_individuales
        )

        for operacion in operaciones:
            segmentos_op = operacion.get("segmentos", [])

            if len(segmentos_op) == 0:
                continue

            etiquetas_op = [seg.get("etiqueta") for seg in segmentos_op]
            datos_patron = crear_datos_patron_desplazamiento_desde_segmentos_bd(
                eje=eje,
                segmentos_individuales=segmentos_op,
                dtw_max=operacion.get("distancia_max")
            )

            if datos_patron is None:
                continue

            if operacion.get("accion") == "crear":
                id_patron = insertar_patron_normal_bd(cur, datos_patron)
                id_patron = int(id_patron)
                n_insertados += 1
                ids_patrones_creados_esta_ejecucion.add(id_patron)

            else:
                id_patron = int(operacion.get("id_patron"))
                actualizar_patron_normal_bd(
                    cur=cur,
                    id_patron=id_patron,
                    datos_patron=datos_patron,
                    distancia_representantes=operacion.get("distancia_max")
                )

                if id_patron not in ids_patrones_creados_esta_ejecucion:
                    ids_patrones_existentes_actualizados.add(id_patron)

            n_segmentos_guardados += insertar_segmentos_individuales_en_patron_bd(
                cur=cur,
                id_patron=id_patron,
                tipo_patron="desplazamiento",
                eje=eje,
                segmentos_individuales=segmentos_op
            )

            recalcular_representante_patron_bd(cur, id_patron)

            if id_patron in ids_patrones_creados_esta_ejecucion:
                anadir_etiquetas_a_resumen_bd(
                    resumen_segmentos_nuevos,
                    id_patron,
                    etiquetas_op
                )
            else:
                anadir_etiquetas_a_resumen_bd(
                    resumen_segmentos_anadidos,
                    id_patron,
                    etiquetas_op
                )

        for grupo_fijo in grupos_fijos:
            tipo_fijo = grupo_fijo.get("tipo")
            segmentos_fijos = grupo_fijo.get("segmentos", [])
            etiquetas_fijas = grupo_fijo.get("etiquetas", [])

            if tipo_fijo not in ("oscilatorio", "reposo"):
                continue

            if len(segmentos_fijos) == 0:
                continue

            stats = preparar_estadisticas_segmentos_bd(
                acc_sin_filtrar=acc_sin_filtrar,
                acc_filtrada=acc_filtrada,
                segmentos=segmentos_fijos,
                eje=eje
            )

            rms_ori_max = stats["rms_original_max"]
            pp_ori_max = stats["pico_pico_original_max"]

            if tipo_fijo == "reposo":
                offset_abs_original_max = stats["offset_abs_original_max"]
            else:
                offset_abs_original_max = None

            datos_patron = crear_datos_patron_bd(
                tipo_patron=tipo_fijo,
                eje=eje,
                duracion_min=None,
                duracion_max=None,
                rms_ori_max=rms_ori_max,
                pp_ori_max=pp_ori_max,
                dtw_max=None,
                offset_abs_original_max=offset_abs_original_max,
                forma_representante=stats["forma_representante"],
                segmento_original_rep=stats["segmento_original_rep"],
                segmento_filtrado_rep=stats["segmento_filtrado_rep"],
                f_original_rep=stats["f_original_rep"],
                espectro_original_rep=stats["espectro_original_rep"],
                f_filtrado_rep=stats["f_filtrado_rep"],
                espectro_filtrado_rep=stats["espectro_filtrado_rep"],
                rizo_pico_pico_original_max=stats["rizo_pico_pico_original_max"],
                energias_bandas_original=stats["energias_bandas_original"]
            )

            accion, id_patron, _ = guardar_o_actualizar_patron_normal_bd(cur, datos_patron)
            id_patron = int(id_patron)

            if accion == "insertado":
                n_insertados += 1
                ids_patrones_creados_esta_ejecucion.add(id_patron)
            elif id_patron not in ids_patrones_creados_esta_ejecucion:
                ids_patrones_existentes_actualizados.add(id_patron)

            if id_patron in ids_patrones_creados_esta_ejecucion:
                anadir_etiquetas_a_resumen_bd(
                    resumen_segmentos_nuevos,
                    id_patron,
                    etiquetas_fijas
                )
            else:
                anadir_etiquetas_a_resumen_bd(
                    resumen_segmentos_anadidos,
                    id_patron,
                    etiquetas_fijas
                )

            idx_eje_fijo = IDX_EJE[eje]

            segmentos_fijos_individuales = []

            for etiqueta_fija, (ini_fijo, fin_fijo) in zip(etiquetas_fijas, segmentos_fijos):
                seg_original_fijo = acc_sin_filtrar[ini_fijo:fin_fijo + 1, idx_eje_fijo]
                seg_filtrada_fijo = acc_filtrada[ini_fijo:fin_fijo + 1, idx_eje_fijo]

                f_ori_fijo, esp_ori_fijo = calcular_espectro_segmento_1d(seg_original_fijo)
                f_fil_fijo, esp_fil_fijo = calcular_espectro_segmento_1d(seg_filtrada_fijo)

                forma_fijo = calcular_forma_normalizada_bd(seg_filtrada_fijo)

                segmentos_fijos_individuales.append({
                    "etiqueta": etiqueta_fija,
                    "segmento_original": seg_original_fijo,
                    "segmento_filtrado": seg_filtrada_fijo,
                    "forma_normalizada": forma_fijo,
                    "f_original": f_ori_fijo,
                    "espectro_original": esp_ori_fijo,
                    "f_filtrado": f_fil_fijo,
                    "espectro_filtrado": esp_fil_fijo,
                })

            n_segmentos_guardados += insertar_segmentos_individuales_en_patron_bd(
                cur=cur,
                id_patron=id_patron,
                tipo_patron=tipo_fijo,
                eje=eje,
                segmentos_individuales=segmentos_fijos_individuales
            )

            recalcular_representante_patron_bd(cur, id_patron)

    conn.commit()
    conn.close()

    imprimir_resumen_segmentos_bd(
        "Segments afegits",
        resumen_segmentos_anadidos
    )
    imprimir_resumen_segmentos_bd(
        "Segments nous",
        resumen_segmentos_nuevos
    )

    print(
        f"\nBase de dades de patrons normals actualitzada:\n"
        f"{RUTA_SCRIPT / NOMBRE_BD_PATRONES_NORMALES}"
    )
    print(f"Patrons nous inserits: {n_insertados}")
    print(f"Patrons existents actualitzats: {len(ids_patrones_existentes_actualizados)}")
    print(f"Segments individuals guardats en 'Segments BD': {n_segmentos_guardados}")

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
        return [], [], f"Error llegint la base de dades:\n{e}"


def formatear_valor_bd(valor, columna=None):
    if valor is None:
        return ""

    if columna == "tipo_patron":
        return traduir_tipus_patron(valor)

    if columna == "eje":
        return str(valor).upper()

    if isinstance(valor, float):
        return f"{valor:.6g}"

    return str(valor)


def traduir_tipus_patron(valor):
    mapa = {
        "desplazamiento": "desplaçament",
        "oscilatorio": "oscil·latori",
        "reposo": "repòs",
        "senyal_completa": "senyal_completa",
    }

    return mapa.get(str(valor), str(valor))


def traduir_columna_bd(columna):
    mapa = {
        "id": "ID",
        "tipo_patron": "Tipus de patró",
        "eje": "Eix",
        "duracion_s_min": "Duració mín. [s]",
        "duracion_s_max": "Duració màx. [s]",
        "rms_original_max": "RMS original màx.",
        "pico_pico_original_max": "Pic pic original màx.",
        "rizo_pico_pico_original_max": "Ondulació pic pic original màx.",
        "dtw_distancia_max": "Distància DTW màx.",
        "offset_abs_original_max": "Offset abs. original màx.",
    }

    for f_ini, f_fin in BANDAS_FRECUENCIA_BD:
        original = f"energia_banda_{f_ini}_{f_fin}_original_max"
        mapa[original] = f"Energia banda {f_ini}-{f_fin} Hz original màx."

    return mapa.get(str(columna), str(columna))


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
            tree.heading(col, text=traduir_columna_bd(col))

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
            valores = [formatear_valor_bd(v, col) for v, col in zip(fila, columnas)]
            tree.insert("", tk.END, values=valores)

        lbl_info.config(
            text=f"Patrons normals guardats: {len(filas)}"
        )

    btn_actualizar.config(command=cargar_tabla)
    cargar_tabla()


def _blob_a_array_seguro(blob):
    try:
        return blob_a_array(blob)
    except Exception:
        return np.array([], dtype=float)


def leer_segmentos_bd_para_visualizacion():
    """
    Devuelve una lista con un elemento por grupo/patrón guardado en la BD
    (uno por fila de patrones_normales_dtw), incluyendo:
    - el representante del grupo (arrays guardados en patrones_normales_dtw)
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
        return [], f"Error llegint segments de la base de dades:\n{e}"

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
    Devuelve la duración máxima de todos los segmentos de desplazamiento
    guardados en la BD.

    Se usa para que, en la pestaña Segmentos BD, todos los grupos de
    desplazamiento tengan la misma escala horizontal en la gráfica:
    'Segmento original guardado (tiempo real relativo)'.

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
        f"Grup BD ID {id_patron} | {traduir_tipus_patron(tipo_patron)} | eix {eje.upper()} | "
        f"{len(segmentos)} segment(s) guardat(s)",
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

    ax1.set_title("Segment original guardat (temps real relatiu)")
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

    ax2.set_title("Forma normalitzada guardada (utilitzada en DTW)")
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
        ax3.plot(f_fil_rep, e_fil_rep, linewidth=2.0, color="tab:orange", label="Filtrada (representativa)")

    ax3.set_title("Espectre guardat")
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
                text="No hi ha patrons guardats en la base de dades.",
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
                 f"| {traduir_tipus_patron(grupo_bd['tipo_patron'])} | eix {grupo_bd['eje']} "
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
        return {}, f"Error llegint les bandes de freqüència:\n{e}"

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

        x = np.arange(len(ETIQUETAS_BANDAS_FRECUENCIA_BD))
        ax.bar(x, valores)

        ax.set_title(f"Eix {eje.upper()}")
        ax.set_ylabel("Energia")
        ax.grid(True, axis="y")

        ax.set_xticks(x)
        ax.set_xticklabels(
            ETIQUETAS_BANDAS_FRECUENCIA_BD,
            rotation=30,
            ha="right",
            fontsize=8
        )

        if ymax is not None:
            ax.set_ylim(0, ymax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
                text="Encara no hi ha dades de senyal_completa guardades.",
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


##################################################
# TKINTER
##################################################


def mostrar_ventana_bd_final(parent=None):
    """
    Abre una ventana final con solo las pestañas de la base de datos.

    Se usa al terminar el guardado masivo para revisar directamente:
    - Base de datos
    - Segmentos BD
    """
    if parent is None:
        root = tk.Tk()
        usar_mainloop = True
    else:
        root = tk.Toplevel(parent)
        usar_mainloop = False

    root.title("Base de dades")
    root.geometry("1300x850")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    frame_bd = ttk.Frame(notebook)
    frame_segmentos_bd = ttk.Frame(notebook)
    frame_bandas_bd = ttk.Frame(notebook)

    notebook.add(frame_bd, text="Taula de dades")
    notebook.add(frame_segmentos_bd, text="Segments BD")
    notebook.add(frame_bandas_bd, text="Bandes freqüència")

    insertar_bd_en_frame(frame_bd)
    insertar_segmentos_bd_paginados_en_frame(frame_segmentos_bd)
    insertar_bandas_frecuencia_bd_en_frame(frame_bandas_bd)

    if usar_mainloop:
        root.mainloop()


def parsear_lista_experimentos(texto):
    """
    Convierte un texto tipo:
        132, 133, 142, 143
    en una lista de enteros sin duplicados.

    También acepta espacios, saltos de línea, punto y coma y rangos simples:
        132-135, 140
    """
    texto = str(texto).strip()

    if texto == "":
        raise ValueError("Introdueix almenys un número d'experiment.")

    tokens = re.split(r"[,;\s]+", texto)
    experimentos = []

    for token in tokens:
        token = token.strip()

        if token == "":
            continue

        if "-" in token:
            partes = token.split("-")

            if len(partes) != 2:
                raise ValueError(f"Rang no vàlid: {token}")

            try:
                ini = int(partes[0].strip())
                fin = int(partes[1].strip())
            except ValueError:
                raise ValueError(f"Rang no vàlid: {token}")

            if fin < ini:
                ini, fin = fin, ini

            for num in range(ini, fin + 1):
                experimentos.append(num)

        else:
            try:
                experimentos.append(int(token))
            except ValueError:
                raise ValueError(f"Experiència no vàlid: {token}")

    vistos = set()
    resultado = []

    for num in experimentos:
        if num not in vistos:
            vistos.add(num)
            resultado.append(num)

    if len(resultado) == 0:
        raise ValueError("No s'ha trobat cap experiència vàlida.")

    return resultado


def preparar_bd_para_guardado_lote():
    """
    Inicializa la base de datos antes de un guardado masivo.
    """
    conn = crear_conexion_bd_patrones()
    inicializar_bd_patrones_normales(conn)
    cur = conn.cursor()

    if BORRAR_BD_PATRONES_ANTES_DE_GUARDAR:
        cur.execute("DELETE FROM patrones_normales_dtw")
        cur.execute("DELETE FROM segmentos_patrones_dtw")
        cur.execute("DELETE FROM sqlite_sequence WHERE name = 'segmentos_patrones_dtw'")
        cur.execute("DELETE FROM sqlite_sequence WHERE name = 'patrones_normales_dtw'")

        conn.commit()

    conn.close()


def obtener_carpetas_experimento_para_guardado_bd(num_experimento):
    """
    Devuelve todas las carpetas/programas con timeblock*.txt de un experimento.
    """
    ruta_base = obtener_ruta_base(num_experimento)

    if not ruta_base.exists():
        raise FileNotFoundError(
            f"No existeix la ruta de l'experiència {num_experimento}:\n{ruta_base}"
        )

    carpetas = obtener_carpetas_con_timeblocks(ruta_base)

    if len(carpetas) == 0:
        raise FileNotFoundError(
            f"No s'han trobat timeblock*.txt en l'experiència {num_experimento}:\n{ruta_base}"
        )

    return carpetas


def procesar_carpeta_experimento_para_bd(
        num_experimento,
        carpeta,
        cargar_todos_los_bloques,
        num_bloque,
        estado_callback=None
):
    """
    Procesa una carpeta/programa de un experimento y guarda sus segmentos en BD.
    """
    global experimento, bloque, CARGAR_TODOS_LOS_BLOQUES

    experimento = int(num_experimento)
    bloque = int(num_bloque)
    CARGAR_TODOS_LOS_BLOQUES = bool(cargar_todos_los_bloques)

    if estado_callback is not None:
        estado_callback(f"Carregant experiència {experimento} | {carpeta.name}...")

    (
        _t,
        acc_sin_filtrar,
        acc_filtrada_suave,
        acc_filtrada_agresiva,
        _unidad_acc
    ) = cargar_datos_carpeta(carpeta)

    acc_segmentacion = acc_filtrada_suave
    acc_validacion_desplazamiento = acc_filtrada_agresiva
    acc_clasificacion_dtw = acc_validacion_desplazamiento

    if USAR_SENAL_SIN_FILTRAR_PARA_IMPACTOS:
        acc_impactos = acc_sin_filtrar
    else:
        acc_impactos = acc_segmentacion

    if estado_callback is not None:
        estado_callback(f"Segmentant experiència {experimento} | {carpeta.name}...")

    info_seg = segmentar_xy_sin_z(
        acc_segmentacion_suave=acc_segmentacion,
        acc_validacion_desplazamiento=acc_validacion_desplazamiento,
        acc_impactos=acc_impactos
    )

    if not USAR_CLASIFICACION_DTW:
        return

    if estado_callback is not None:
        estado_callback(f"Comparant amb la BD experiència {experimento} | {carpeta.name}...")

    info_clasificacion_dtw = clasificar_segmentos_dtw_xy(
        acc=acc_clasificacion_dtw,
        info_seg=info_seg
    )

    if GUARDAR_BD_PATRONES_NORMALES and info_clasificacion_dtw is not None:
        guardar_patrones_normales_dtw_bd(
            acc_sin_filtrar=acc_sin_filtrar,
            acc_filtrada=acc_clasificacion_dtw,
            info_clasificacion_dtw=info_clasificacion_dtw,
            permitir_borrado_bd=False
        )

    del acc_sin_filtrar
    del acc_filtrada_suave
    del acc_filtrada_agresiva
    del acc_segmentacion
    del acc_validacion_desplazamiento
    del acc_clasificacion_dtw
    del acc_impactos
    del info_seg
    del info_clasificacion_dtw

    gc.collect()

    if PAUSA_CPU_BATCH_S > 0:
        time.sleep(PAUSA_CPU_BATCH_S)


def ejecutar_guardado_lote_bd_desde_seleccion(seleccion, parent=None, estado_callback=None, progreso_callback=None):
    """
    Procesa una lista de experimentos y guarda todos sus segmentos en la BD.
    Cada experimento puede contener una o varias carpetas/programas; se procesan todas.
    """
    experimentos_lote = seleccion["experimentos"]
    cargar_todos = seleccion["cargar_todos_los_bloques"]
    num_bloque = seleccion["bloque"]

    if estado_callback is not None:
        estado_callback("Calculant el llindar de repòs des de l'experiència de referència...")

    actualizar_umbral_reposo_desde_experimento_referencia()
    preparar_bd_para_guardado_lote()

    tareas = []
    errores = []

    for num_exp in experimentos_lote:
        try:
            carpetas = obtener_carpetas_experimento_para_guardado_bd(num_exp)

            for carpeta in carpetas:
                tareas.append((num_exp, carpeta))

        except Exception as e:
            errores.append(f"Experiència {num_exp}: {e}")

    total = len(tareas)

    if total == 0:
        mensaje = "No s'ha pogut processar cap carpeta amb timeblock*.txt."

        if len(errores) > 0:
            mensaje += "\n\nErrors:\n" + "\n".join(errores[:8])

        raise RuntimeError(mensaje)

    if progreso_callback is not None:
        progreso_callback(0, total, "Preparat per a processar")

    procesadas = 0

    for i, (num_exp, carpeta) in enumerate(tareas, start=1):
        try:
            texto_progreso = f"Processant {i}/{total}: Experiència {num_exp} | {carpeta.name}"

            if estado_callback is not None:
                estado_callback(texto_progreso)

            if progreso_callback is not None:
                progreso_callback(i - 1, total, texto_progreso)

            procesar_carpeta_experimento_para_bd(
                num_experimento=num_exp,
                carpeta=carpeta,
                cargar_todos_los_bloques=cargar_todos,
                num_bloque=num_bloque,
                estado_callback=estado_callback
            )

            procesadas += 1

        except Exception as e:
            errores.append(f"Experiència {num_exp} | {carpeta.name}: {e}")

        if progreso_callback is not None:
            progreso_callback(i, total, f"Processades {i}/{total} carpeta(es)")

        if parent is not None:
            parent.update_idletasks()
            parent.update()

    if estado_callback is not None:
        estado_callback(
            f"Guardat finalitzat. Carpetes processades: {procesadas}/{total}."
        )

    if progreso_callback is not None:
        progreso_callback(total, total, "Guardat finalitzat")

    return {
        "procesadas": procesadas,
        "total": total,
        "errores": errores,
    }


def pedir_configuracion_inicial():
    """
    Ventana inicial V7 para seleccionar varios experimentos y guardarlos
    directamente en la base de datos.
    """
    root = tk.Tk()
    root.title("Crear base de dades")
    root.geometry("760x560")
    root.resizable(False, False)

    bloque_var = tk.StringVar(value=str(bloque))
    todos_var = tk.BooleanVar(value=CARGAR_TODOS_LOS_BLOQUES)
    estado_var = tk.StringVar(value="")
    progreso_var = tk.DoubleVar(value=0.0)
    porcentaje_var = tk.StringVar(value="0 %")
    borrar_bd_var = tk.BooleanVar(
    value=bool(BORRAR_BD_PATRONES_ANTES_DE_GUARDAR)
)

    main_frame = ttk.Frame(root, padding=18)
    main_frame.pack(fill=tk.BOTH, expand=True)

    titulo = ttk.Label(
        main_frame,
        text="Seleccionar experiències per a guardar en la base de dades",
        font=("Segoe UI", 13, "bold")
    )
    titulo.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

    ttk.Label(
        main_frame,
        text="Experiències:",
    ).grid(row=1, column=0, sticky="nw", pady=6)

    txt_experimentos = tk.Text(
        main_frame,
        width=66,
        height=5,
        wrap="word"
    )
    txt_experimentos.grid(row=1, column=1, columnspan=2, sticky="w", pady=6)
    txt_experimentos.insert("1.0", str(experimento))

    ayuda = ttk.Label(
        main_frame,
        text="Exemple: 132, 133, 142, 143, 145   |   També accepta rangs: 132-135",
        foreground="gray"
    )
    ayuda.grid(row=2, column=1, columnspan=2, sticky="w", pady=(0, 8))

    check_todos = ttk.Checkbutton(
        main_frame,
        text="Carregar tots els blocs",
        variable=todos_var
    )
    check_todos.grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 6))

    ttk.Label(main_frame, text="Bloc:").grid(row=4, column=0, sticky="w", pady=6)
    entry_bloque = ttk.Entry(main_frame, textvariable=bloque_var, width=14)
    entry_bloque.grid(row=4, column=1, sticky="w", pady=6)

    check_borrar_bd = ttk.Checkbutton(
        main_frame,
        text="Esborrar patrons abans de guardar",
        variable=borrar_bd_var
    )
    check_borrar_bd.grid(row=5, column=0, columnspan=2, sticky="w", pady=6)

    lbl_aviso = ttk.Label(
        main_frame,
        text="Es processen totes les carpetes/programes amb timeblock*.txt de cada experiment.",
        foreground="gray"
    )
    lbl_aviso.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

    label_estado = ttk.Label(main_frame, textvariable=estado_var, foreground="gray")
    label_estado.grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 6))

    frame_progreso = ttk.Frame(main_frame)
    frame_progreso.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 6))

    barra_progreso = ttk.Progressbar(
        frame_progreso,
        orient=tk.HORIZONTAL,
        mode="determinate",
        maximum=100.0,
        variable=progreso_var,
        length=580
    )
    barra_progreso.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    lbl_porcentaje = ttk.Label(
        frame_progreso,
        textvariable=porcentaje_var,
        width=8,
        anchor="e"
    )
    lbl_porcentaje.pack(side=tk.RIGHT)

    botones = ttk.Frame(main_frame)
    botones.grid(row=9, column=0, columnspan=3, sticky="e", pady=(18, 0))

    btn_cancelar = ttk.Button(botones, text="Cancel·lar")
    btn_cancelar.pack(side=tk.RIGHT, padx=(8, 0))

    btn_aceptar = ttk.Button(botones, text="Guardar en BD")
    btn_aceptar.pack(side=tk.RIGHT, padx=(8, 0))

    btn_ver_bd = ttk.Button(botones, text="Veure base de dades")
    btn_ver_bd.pack(side=tk.RIGHT)

    def actualizar_estado_bloque(*_):
        if todos_var.get():
            entry_bloque.configure(state="disabled")
        else:
            entry_bloque.configure(state="normal")

    def set_estado(texto):
        estado_var.set(texto)
        root.update_idletasks()

    def set_progreso(actual, total, texto=None):
        total = max(1, int(total))
        actual = max(0, min(int(actual), total))
        porcentaje = 100.0 * actual / total
        progreso_var.set(porcentaje)
        porcentaje_var.set(f"{porcentaje:5.1f} %")

        if texto is not None:
            estado_var.set(texto)

        root.update_idletasks()

    def ver_base_datos():
        """
        Abre la ventana de la base de datos sin procesar experimentos
        ni guardar ningún dato nuevo.
        """
        mostrar_ventana_bd_final(parent=root)

    def aceptar():
        try:
            lista_exps = parsear_lista_experimentos(
                txt_experimentos.get("1.0", tk.END)
            )
        except ValueError as e:
            messagebox.showerror("Experiències no vàlids", str(e), parent=root)
            return

        cargar_todos = bool(todos_var.get())
        borrar_bd = bool(borrar_bd_var.get())

        try:
            num_bloque = int(bloque_var.get().strip())
        except ValueError:
            messagebox.showerror("Bloc no vàlid", "Introdueix un número de bloc vàlid.", parent=root)
            return

        if num_bloque < 0:
            messagebox.showerror("Bloc no vàlid", "El bloc no pot ser negatiu.", parent=root)
            return

        confirmacion = messagebox.askyesno(
            "Confirmar guardat massiu",
            f"Es processaran {len(lista_exps)} experiment(s):\n"
            f"{', '.join(str(e) for e in lista_exps)}\n\n"
            "Açò pot tardar bastant. Vols continuar?",
            parent=root
        )

        if not confirmacion:
            return

        if borrar_bd:
            confirmar_borrado = messagebox.askyesno(
                "Avís: s'esborrarà la base de dades",
                "Has activat 'Esborrar patrons abans de guardar'.\n\n"
                "Abans d'iniciar el guardat s'eliminaran els patrons i segments "
                "guardats actualment en la base de dades.\n\n"
                "Segur que vols continuar?",
                parent=root
            )

            if not confirmar_borrado:
                return

        global BORRAR_BD_PATRONES_ANTES_DE_GUARDAR
        BORRAR_BD_PATRONES_ANTES_DE_GUARDAR = borrar_bd

        seleccion_actual = {
            "experimentos": lista_exps,
            "cargar_todos_los_bloques": cargar_todos,
            "bloque": num_bloque,
            "borrar_bd_antes_de_guardar": borrar_bd,
        }

        progreso_var.set(0.0)
        porcentaje_var.set("0 %")

        btn_aceptar.configure(state="disabled")
        btn_ver_bd.configure(state="disabled")
        btn_cancelar.configure(state="disabled")
        check_todos.configure(state="disabled")
        entry_bloque.configure(state="disabled")
        check_borrar_bd.configure(state="disabled")
        txt_experimentos.configure(state="disabled")
        root.update_idletasks()

        try:
            resultado = ejecutar_guardado_lote_bd_desde_seleccion(
                seleccion=seleccion_actual,
                parent=root,
                estado_callback=set_estado,
                progreso_callback=set_progreso
            )

            errores = resultado.get("errores", [])
            procesadas = resultado.get("procesadas", 0)
            total = resultado.get("total", 0)

            if len(errores) == 0:
                messagebox.showinfo(
                    "Guardat finalitzat",
                    f"Base de dades actualitzada correctament.\n"
                    f"Carpetes processades: {procesadas}/{total}",
                    parent=root
                )
            else:
                texto_errores = "\n".join(errores[:10])

                if len(errores) > 10:
                    texto_errores += f"\n... i {len(errores) - 10} error(s) més."

                messagebox.showwarning(
                    "Guardat finalitzat amb avisos",
                    f"Carpetes processades: {procesadas}/{total}\n\n"
                    f"Errors/avisos:\n{texto_errores}",
                    parent=root
                )

            set_estado(
                f"Finalitzat. Carpetes processades: {procesadas}/{total}."
            )

            mostrar_ventana_bd_final(parent=root)

        except Exception as e:
            set_estado("Error durant el guardat massiu.")
            messagebox.showerror("Error durant el guardat", str(e), parent=root)

        finally:
            btn_aceptar.configure(state="normal")
            btn_ver_bd.configure(state="normal")
            btn_cancelar.configure(state="normal")
            check_todos.configure(state="normal")
            check_borrar_bd.configure(state="normal")
            txt_experimentos.configure(state="normal")
            actualizar_estado_bloque()

    def cancelar():
        root.destroy()

    btn_aceptar.configure(command=aceptar)
    btn_ver_bd.configure(command=ver_base_datos)
    btn_cancelar.configure(command=cancelar)
    check_todos.configure(command=actualizar_estado_bloque)
    root.protocol("WM_DELETE_WINDOW", cancelar)

    actualizar_estado_bloque()
    txt_experimentos.focus_set()
    root.mainloop()


##################################################
# MAIN
##################################################


def main():
    pedir_configuracion_inicial()


if __name__ == "__main__":
    main()
