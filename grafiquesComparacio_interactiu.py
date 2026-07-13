
"""
grafiquesComparacio_interactiu.py

Permet comparar diversos experiments.
Obre una finestra amb dades temporal y un altra amb les corresponents en el domini de la freqüència.

"""

import re
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend


################################################## CONFIGURACIÓN GENERAL

Fs = 1000

EXPERIMENTOS_DEFECTO = "86, 84, 82, 73"
BLOQUE_DEFECTO = "0"
CARPETA_DATA_DEFECTO = "0"  # Puede ser índice: 0, 1, 2... o nombre exacto de carpeta
TITULO_DEFECTO = "Comparació"

RUTA_SCRIPT = Path(__file__).resolve().parent
RUTA_DATOS = RUTA_SCRIPT.parent


################################################## FUNCIONES DE DATOS

def obtener_config_experimento(experimento):
    """Devuelve ruta de datos, hoja de Excel y columnas según el número de experimento."""

    if experimento < 59: # perfil aluminio
        ruta_base = (
            RUTA_DATOS
            / "pruebas_lab"
            / "pruebasExperimentalesDocumentadas"
            / f"Experimento{experimento}"
        )
        nombre_hoja = "PerfilAluminio"
        columnas_excel = "A:H"

    elif 64 < experimento < 114: # portico
        ruta_base = (
            RUTA_DATOS
            / "pruebas_lab"
            / "pruebasExperimentalesDocumentadas"
            / f"Experimento{experimento}"
        )
        nombre_hoja = "Portico"
        columnas_excel = "A:K"

    elif 113 < experimento < 120  or 122 < experimento: # planta
        ruta_base = RUTA_DATOS / "pruebas_planta" / f"Experimento{experimento}"
        nombre_hoja = "Planta"
        columnas_excel = "A:F"


    elif 119 < experimento < 123: # guitarra
        ruta_base = RUTA_DATOS / "pruebas_planta" / f"Experimento{experimento}"
        nombre_hoja = "Guitarra"
        columnas_excel = "A:E"

    else:
        raise ValueError(f"Experimento {experimento} no contemplado")

    return ruta_base, nombre_hoja, columnas_excel


def obtener_nombre_experimento(experimento):
    """Obtiene una etiqueta descriptiva del experimento desde Registro_Experimentos.xlsx."""

    _, nombre_hoja, columnas_excel = obtener_config_experimento(experimento)

    archivo_excel = (
        RUTA_DATOS
        / "pruebas_lab"
        / "pruebasExperimentalesDocumentadas"
        / "Registro_Experimentos.xlsx"
    )

    if not archivo_excel.exists():
        raise FileNotFoundError(f"No s'ha pogut trobar l'arxiu: {archivo_excel}")

    df = pd.read_excel(
        archivo_excel,
        sheet_name=nombre_hoja,
        header=4,
        usecols=columnas_excel,
        engine="openpyxl",
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )

    col_exp = "Número del experimento (cronológico)"

    if col_exp not in df.columns:
        raise KeyError(f"No existe la columna '{col_exp}' en la hoja '{nombre_hoja}'")

    fila = df.loc[df[col_exp] == experimento]

    if fila.empty:
        return f"Experimento {experimento}"

    fila = fila.iloc[0]

    if experimento < 59: # perfil aluminio
        titulo_experimento = (
            f"càrrega = {fila['Càrrega (g)']} g | "
            f"Posició = {fila['Posició de la càrrega (cm)']} cm | "
            f"V = {fila['Voltatge del motor (V)']} V"
        )

    elif 64 < experimento < 114: # portico
        titulo_experimento = (
            f"v = {fila['Velocitat (mm/s)']} mm/s | "
            f"Moviment: {fila['Moviment (x, y, z)']} | "
            f"Màquina: {fila['Estat màquina']}"
        )

    elif 113 < experimento < 120  or 122 < experimento: # planta
        titulo_experimento = (
            f"Moviment: {fila['Moviment (x, y, z)']} | "
            f"Fondo escala: {fila['Fons Escala']} g"
        )


    elif 119 < experimento < 123: # guitarra
        titulo_experimento = (
            f"Nota: {fila['Nota']} | "
            f"Fons escala: {fila['Fons Escala']}"
        )

    else:
        titulo_experimento = f"Experimento {experimento}"

    if "Observacions" in fila.index and pd.notna(fila["Observacions"]):
        titulo_experimento += f" | Obs.: {fila['Observacions']}"

    return titulo_experimento


def obtener_valor_columna_b(experimento):
    """Obtiene el valor de la columna B del Excel para el experimento indicado."""

    _, nombre_hoja, columnas_excel = obtener_config_experimento(experimento)

    archivo_excel = (
        RUTA_DATOS
        / "pruebas_lab"
        / "pruebasExperimentalesDocumentadas"
        / "Registro_Experimentos.xlsx"
    )

    if not archivo_excel.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {archivo_excel}")

    df = pd.read_excel(
        archivo_excel,
        sheet_name=nombre_hoja,
        header=4,
        usecols=columnas_excel,
        engine="openpyxl",
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )

    col_exp = "Número del experimento (cronológico)"

    if col_exp not in df.columns:
        raise KeyError(f"No existe la columna '{col_exp}' en la hoja '{nombre_hoja}'")

    fila = df.loc[df[col_exp] == experimento]

    if fila.empty:
        return str(experimento)

    if df.shape[1] < 2:
        raise ValueError(f"La hoja '{nombre_hoja}' no tiene columna B dentro del rango leído")

    valor_columna_b = fila.iloc[0, 1]

    if pd.isna(valor_columna_b):
        return str(experimento)

    if isinstance(valor_columna_b, float) and valor_columna_b.is_integer():
        return str(int(valor_columna_b))

    return str(valor_columna_b).strip()


def parsear_experimentos(texto):
    """Convierte '86, 84, 82' o '86 84 82' en [86, 84, 82]."""

    partes = [p for p in re.split(r"[,;\s]+", texto.strip()) if p]

    if not partes:
        raise ValueError("Introduce al menos un experimento")

    experimentos = []
    for parte in partes:
        try:
            experimentos.append(int(parte))
        except ValueError:
            raise ValueError(f"'{parte}' no es un número de experimento válido")

    return experimentos


def seleccionar_carpeta(ruta_base, selector_carpeta):
    """
    Selecciona una única carpeta de datos.
    selector_carpeta puede ser:
      - un índice: 0, 1, 2...
      - el nombre exacto de la carpeta
      - vacío: primera carpeta
    """

    carpetas = sorted([p for p in ruta_base.iterdir() if p.is_dir()])

    if not carpetas:
        raise FileNotFoundError(f"No hay carpetas de datos en {ruta_base}")

    selector_carpeta = selector_carpeta.strip()

    if selector_carpeta == "":
        return carpetas[0]

    if selector_carpeta.isdigit():
        indice = int(selector_carpeta)
        if indice < 0 or indice >= len(carpetas):
            raise IndexError(
                f"Índice de carpeta {indice} fuera de rango en {ruta_base}. "
                f"Hay {len(carpetas)} carpeta(s)."
            )
        return carpetas[indice]

    for carpeta in carpetas:
        if carpeta.name == selector_carpeta:
            return carpeta

    nombres = ", ".join(c.name for c in carpetas)
    raise FileNotFoundError(
        f"No existe la carpeta '{selector_carpeta}' en {ruta_base}. "
        f"Carpetas disponibles: {nombres}"
    )


def leer_espectro(spec, exp, t):
    """Devuelve frecuencia y espectros X/Y/Z."""

    if exp == 120:
        nsamples = len(t)
        f_max = 500
        espectro_dim = int(f_max * nsamples / Fs)

        f_dsp = np.arange(espectro_dim) * Fs / nsamples
        x_dsp = spec[:espectro_dim, 0]
        y_dsp = spec[:espectro_dim, 1]
        z_dsp = spec[:espectro_dim, 2]
    else:
        f_dsp = spec[:, 0]
        x_dsp = spec[:, 1]
        y_dsp = spec[:, 2]
        z_dsp = spec[:, 3]

    return f_dsp, x_dsp, y_dsp, z_dsp


ventana_control_visibilidad = None


def crear_panel_visibilidad_tkinter(root, fig_ac, fig_frec, axs_ac, axs_frec, registros_curvas):
    """Crea una ventana Tkinter con checkboxes de estilo nativo para mostrar/ocultar experimentos."""

    global ventana_control_visibilidad

    if not registros_curvas:
        return

    if ventana_control_visibilidad is not None and ventana_control_visibilidad.winfo_exists():
        ventana_control_visibilidad.destroy()

    panel = tk.Toplevel(root)
    ventana_control_visibilidad = panel
    panel.title("Mostrar / ocultar experiments")
    panel.geometry("430x360")
    panel.minsize(380, 280)

    # Estilo tipo Windows mediante ttk.
    # En Windows suele estar disponible el tema "vista" o "xpnative".
    style = ttk.Style(panel)
    temas = style.theme_names()
    if "vista" in temas:
        style.theme_use("vista")
    elif "xpnative" in temas:
        style.theme_use("xpnative")

    style.configure("Titulo.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("Texto.TLabel", font=("Segoe UI", 9))
    style.configure("Experimento.TCheckbutton", font=("Segoe UI", 10), padding=(4, 6))
    style.configure("Boton.TButton", font=("Segoe UI", 9), padding=(8, 4))

    panel.columnconfigure(0, weight=1)
    panel.rowconfigure(2, weight=1)

    frame_principal = ttk.Frame(panel, padding=(18, 16, 18, 16))
    frame_principal.grid(row=0, column=0, sticky="nsew")
    frame_principal.columnconfigure(0, weight=1)
    frame_principal.rowconfigure(2, weight=1)

    titulo = ttk.Label(
        frame_principal,
        text="CheckBox options:",
        style="Titulo.TLabel",
        anchor="w",
    )
    titulo.grid(row=0, column=0, sticky="w", pady=(0, 8))

    barra_botones = ttk.Frame(frame_principal)
    barra_botones.grid(row=1, column=0, sticky="ew", pady=(0, 10))

    def actualizar_leyenda(ax):
        leyenda_actual = ax.get_legend()
        if leyenda_actual is not None:
            leyenda_actual.remove()

        handles, labels = ax.get_legend_handles_labels()
        visibles = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if handle.get_visible()
        ]

        if visibles:
            handles_visibles, labels_visibles = zip(*visibles)
            ax.legend(handles_visibles, labels_visibles, fontsize=8)

    def actualizar_leyendas():
        actualizar_leyenda(axs_ac[0])
        actualizar_leyenda(axs_frec[0])

    def aplicar_visibilidad(indice):
        registro = registros_curvas[indice]
        visible = registro["var"].get()
        registro["visible"] = visible

        for linea in registro["lineas"]:
            linea.set_visible(visible)

        actualizar_leyendas()
        fig_ac.canvas.draw_idle()
        fig_frec.canvas.draw_idle()

    def establecer_todos(visible):
        for indice, registro in enumerate(registros_curvas):
            registro["var"].set(visible)
            registro["visible"] = visible

            for linea in registro["lineas"]:
                linea.set_visible(visible)

        actualizar_leyendas()
        fig_ac.canvas.draw_idle()
        fig_frec.canvas.draw_idle()

    boton_mostrar_todo = ttk.Button(
        barra_botones,
        text="Mostrar tot",
        style="Boton.TButton",
        command=lambda: establecer_todos(True),
    )
    boton_mostrar_todo.pack(side="left", padx=(0, 8))

    boton_ocultar_todo = ttk.Button(
        barra_botones,
        text="Ocultar tot",
        style="Boton.TButton",
        command=lambda: establecer_todos(False),
    )
    boton_ocultar_todo.pack(side="left")

    contenedor = ttk.Frame(frame_principal)
    contenedor.grid(row=2, column=0, sticky="nsew")
    contenedor.columnconfigure(0, weight=1)
    contenedor.rowconfigure(0, weight=1)

    canvas = tk.Canvas(
        contenedor,
        highlightthickness=0,
        borderwidth=0,
    )
    scrollbar = ttk.Scrollbar(contenedor, orient="vertical", command=canvas.yview)
    frame_scroll = ttk.Frame(canvas, padding=(0, 2, 0, 2))

    ventana_canvas = canvas.create_window((0, 0), window=frame_scroll, anchor="nw")

    def actualizar_scrollregion(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def ajustar_ancho_frame(event):
        canvas.itemconfigure(ventana_canvas, width=event.width)

    frame_scroll.bind("<Configure>", actualizar_scrollregion)
    canvas.bind("<Configure>", ajustar_ancho_frame)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    for indice, registro in enumerate(registros_curvas):
        registro["var"] = tk.BooleanVar(value=registro["visible"])

        texto = registro["etiqueta_menu"].replace(" | ", "  |  ")
        check = ttk.Checkbutton(
            frame_scroll,
            text=texto,
            variable=registro["var"],
            command=lambda i=indice: aplicar_visibilidad(i),
            style="Experimento.TCheckbutton",
        )
        check.grid(row=indice, column=0, sticky="w", padx=(0, 0), pady=3)
        registro["check"] = check

    def rueda_raton(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def activar_rueda(event):
        canvas.bind_all("<MouseWheel>", rueda_raton)

    def desactivar_rueda(event):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", activar_rueda)
    canvas.bind("<Leave>", desactivar_rueda)

    panel.lift()
    panel.focus_force()

def hacer_etiqueta_unica(etiqueta_base, etiquetas_existentes):
    """Evita etiquetas repetidas en el menú interactivo."""

    etiqueta = etiqueta_base
    contador = 2

    while etiqueta in etiquetas_existentes:
        etiqueta = f"{etiqueta_base} ({contador})"
        contador += 1

    return etiqueta


def comparar_experimentos(experimentos, bloque, selector_carpeta, titulo_figura):
    """Carga una carpeta de datos para varios experimentos y dibuja aceleración y espectro."""

    plt.close("all")

    fig_ac, axs_ac = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_ac.suptitle(f"{titulo_figura} - Acceleració en el domini del temps")

    fig_frec, axs_frec = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_frec.suptitle(f"{titulo_figura} - Espectre de la freqüència")

    ymax_ac = 0.0
    ymax_frec = 0.0
    avisos = []
    curvas_cargadas = 0
    registros_curvas = []
    etiquetas_menu = []

    for exp in experimentos:
        try:
            ruta_base, _, _ = obtener_config_experimento(exp)

            if not ruta_base.exists():
                avisos.append(f"Experimento {exp}: no existe la ruta {ruta_base}")
                continue

            carpeta = seleccionar_carpeta(ruta_base, selector_carpeta)

            archivo_time = carpeta / f"timeblock{bloque}.txt"
            archivo_spectrum = carpeta / f"spectrum{bloque}.txt"

            if not archivo_time.exists() or not archivo_spectrum.exists():
                avisos.append(f"Experimento {exp}: faltan archivos en {carpeta}")
                continue

            acc = np.loadtxt(archivo_time)
            spec = np.loadtxt(archivo_spectrum)

            # Se elimina la media en X e Y.
            # type="constant" quita solo el offset DC, no la tendencia lineal.
            x = detrend(acc[:, 2], type="constant")
            y = detrend(acc[:, 3], type="constant")
            z = acc[:, 4]

            t = np.arange(len(x)) / Fs
            f_dsp, X, Y, Z = leer_espectro(spec, exp, t)

            ymax_ac = max(
                ymax_ac,
                np.max(np.abs(x)),
                np.max(np.abs(y)),
                np.max(np.abs(z)),
            )

            ymax_frec = max(
                ymax_frec,
                np.max(np.abs(X)),
                np.max(np.abs(Y)),
                np.max(np.abs(Z)),
            )

            valor_columna_b = obtener_valor_columna_b(exp)
            nombre_exp = obtener_nombre_experimento(exp)
            label = f"Exp. {valor_columna_b} | {nombre_exp} | {carpeta.name}"

            line_ac_x, = axs_ac[0].plot(t, x, lw=0.8, label=label)
            line_ac_y, = axs_ac[1].plot(t, y, lw=0.8, label=label)
            line_ac_z, = axs_ac[2].plot(t, z, lw=0.8, label=label)

            line_frec_x, = axs_frec[0].plot(f_dsp, X, lw=0.8, label=label)
            line_frec_y, = axs_frec[1].plot(f_dsp, Y, lw=0.8, label=label)
            line_frec_z, = axs_frec[2].plot(f_dsp, Z, lw=0.8, label=label)

            etiqueta_base = f"Exp. {valor_columna_b} | {carpeta.name}"
            etiqueta_menu = hacer_etiqueta_unica(etiqueta_base, etiquetas_menu)
            etiquetas_menu.append(etiqueta_menu)

            registros_curvas.append(
                {
                    "visible": True,
                    "etiqueta_menu": etiqueta_menu,
                    "lineas": [
                        line_ac_x,
                        line_ac_y,
                        line_ac_z,
                        line_frec_x,
                        line_frec_y,
                        line_frec_z,
                    ],
                }
            )

            curvas_cargadas += 1

        except Exception as e:
            avisos.append(f"Experimento {exp}: {e}")

    if curvas_cargadas == 0:
        plt.close(fig_ac)
        plt.close(fig_frec)
        raise RuntimeError("No s'ha pogut carregar cap experiment.\n\n" + "\n".join(avisos))

    configurar_ejes_aceleracion(axs_ac, ymax_ac)
    configurar_ejes_frecuencia(axs_frec, ymax_frec)

    axs_ac[0].legend(fontsize=8)
    axs_frec[0].legend(fontsize=8)

    fig_ac.tight_layout(rect=[0, 0, 1, 0.95])
    fig_frec.tight_layout(rect=[0, 0, 1, 0.95])

    crear_panel_visibilidad_tkinter(
        ventana,
        fig_ac,
        fig_frec,
        axs_ac,
        axs_frec,
        registros_curvas,
    )

    if avisos:
        messagebox.showwarning("Avisos", "\n".join(avisos))

    plt.show(block=False)


def configurar_ejes_aceleracion(axs_ac, ymax_ac):
    for ax in axs_ac:
        ax.grid(True)
        ax.set_ylabel("Amplitud (g)")

    axs_ac[0].set_title("Eix X")
    axs_ac[1].set_title("Eix Y")
    axs_ac[2].set_title("Eix Z")
    axs_ac[2].set_xlabel("Temps (s)")

    if ymax_ac < 1e-12:
        ymax_ac = 1.0

    ymax_ac *= 1.10

    for ax in axs_ac:
        ax.set_ylim(-ymax_ac, ymax_ac)


def configurar_ejes_frecuencia(axs_frec, ymax_frec):
    for ax in axs_frec:
        ax.grid(True)
        ax.set_ylabel("Amplitud (g)")

    axs_frec[0].set_title("Eix X")
    axs_frec[1].set_title("Eix Y")
    axs_frec[2].set_title("Eix Z")
    axs_frec[2].set_xlabel("Freqüència (Hz)")

    if ymax_frec < 1e-12:
        ymax_frec = 1.0

    ymax_frec *= 1.10

    for ax in axs_frec:
        ax.set_ylim(0, ymax_frec)


################################################## INTERFAZ TKINTER

def buscar():
    try:
        experimentos = parsear_experimentos(entry_experimentos.get())

        try:
            bloque = int(entry_bloque.get().strip())
        except ValueError:
            raise ValueError("El bloc ha de ser un número enter")

        selector_carpeta = entry_carpeta.get()
        titulo_figura = entry_titulo.get().strip() or TITULO_DEFECTO

        estado_var.set("Carregant datdes...")
        ventana.update_idletasks()

        comparar_experimentos(experimentos, bloque, selector_carpeta, titulo_figura)

        estado_var.set(
            f"Carregats {len(experimentos)} experiment(s), bloc {bloque}, carpeta '{selector_carpeta or '0'}'."
        )

    except Exception as e:
        estado_var.set("Error al carregar les dades.")
        messagebox.showerror("Error", str(e))


ventana = tk.Tk()
ventana.title("Comparador d'experiments")
ventana.resizable(False, False)

frame = tk.Frame(ventana, padx=20, pady=20)
frame.grid(row=0, column=0)

label_experimentos = tk.Label(frame, text="Experiments:")
label_experimentos.grid(row=0, column=0, sticky="e", padx=(0, 8), pady=6)

entry_experimentos = tk.Entry(frame, width=34)
entry_experimentos.insert(0, EXPERIMENTOS_DEFECTO)
entry_experimentos.grid(row=0, column=1, sticky="w", pady=6)

label_bloque = tk.Label(frame, text="Bloc:")
label_bloque.grid(row=1, column=0, sticky="e", padx=(0, 8), pady=6)

entry_bloque = tk.Entry(frame, width=10)
entry_bloque.insert(0, BLOQUE_DEFECTO)
entry_bloque.grid(row=1, column=1, sticky="w", pady=6)

label_carpeta = tk.Label(frame, text="Carpeta data:")
label_carpeta.grid(row=2, column=0, sticky="e", padx=(0, 8), pady=6)

entry_carpeta = tk.Entry(frame, width=20)
entry_carpeta.insert(0, CARPETA_DATA_DEFECTO)
entry_carpeta.grid(row=2, column=1, sticky="w", pady=6)

label_titulo = tk.Label(frame, text="Títol:")
label_titulo.grid(row=3, column=0, sticky="e", padx=(0, 8), pady=6)

entry_titulo = tk.Entry(frame, width=46)
entry_titulo.insert(0, TITULO_DEFECTO)
entry_titulo.grid(row=3, column=1, sticky="w", pady=6)

boton_buscar = tk.Button(frame, text="Buscar", command=buscar)
boton_buscar.grid(row=0, column=2, rowspan=2, padx=(12, 0), pady=6, sticky="ns")

texto_ayuda = (
    "Experiments separades per comes o espais. \n"
    "Carpeta data: utilitza 0 per a la primera carpeta o escriu el nom."
)
label_ayuda = tk.Label(frame, text=texto_ayuda, justify="left")
label_ayuda.grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 4))

estado_var = tk.StringVar(value="Introdueix les dades i polsa Buscar")
label_estado = tk.Label(frame, textvariable=estado_var, justify="left")
label_estado.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

ventana.bind("<Return>", lambda event: buscar())

ventana.mainloop()
