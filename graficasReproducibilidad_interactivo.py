import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


################################################## CONFIGURACIÓN

Fs = 1000
BLOQUE_POR_DEFECTO = 0


def obtener_configuracion_experimento(experimento: int):
    """
    Devuelve ruta base, hoja de Excel y columnas según el número de experimento.
    """
    raiz = Path(__file__).resolve().parent.parent

    if experimento < 59: # perfil aluminio
        ruta_base = (
            raiz
            / "pruebas_lab"
            / "pruebasExperimentalesDocumentadas"
            / f"Experimento{experimento}"
        )
        nombre_hoja = "PerfilAluminio"
        columnas_excel = "A:H"

    elif 64 < experimento < 114: # portico
        ruta_base = (
            raiz
            / "pruebas_lab"
            / "pruebasExperimentalesDocumentadas"
            / f"Experimento{experimento}"
        )
        nombre_hoja = "Portico"
        columnas_excel = "A:K"

    elif 113 < experimento < 120 or 122 < experimento: # planta
        ruta_base = (
            raiz
            / "pruebas_planta"
            / f"Experimento{experimento}"
        )
        nombre_hoja = "Planta"
        columnas_excel = "A:F"

    elif 119 < experimento < 123: # guitarra
        ruta_base = (
            raiz
            / "pruebas_planta"
            / f"Experimento{experimento}"
        )
        nombre_hoja = "Guitarra"
        columnas_excel = "A:E"

    else:
        raise ValueError(
            f"El experimento {experimento} no entra en ningún rango configurado."
        )

    return ruta_base, nombre_hoja, columnas_excel


def leer_titulo_experimento(experimento: int, nombre_hoja: str, columnas_excel: str) -> str:
    """
    Lee Registro_Experimentos.xlsx y genera el título descriptivo del experimento.
    """
    ruta_script = Path(__file__).resolve().parent

    archivo_excel = (
        ruta_script.parent
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
        engine="openpyxl"
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )

    col_exp = "Número del experimento (cronológico)"
    fila = df.loc[df[col_exp] == experimento]

    if fila.empty:
        return f"Experiencia {experimento}"

    fila = fila.iloc[0]

    if experimento < 59: # perfil aluminio
        titulo_experimento = (
            f"Càrrega = {fila['Càrrega (g)']} g | "
            f"Posició de la càrrega = {fila['Posició de la càrrega (cm)']} | \n"
            f"V: {fila['Voltatge del motor (V)']} V | "
            f"Fons escala: {fila['Fons Escala']} | "
        )

    elif 64 < experimento < 114: # portico
        titulo_experimento = (
            f"v = {fila['Velocitat (mm/s)']} mm/s | "
            f"Moviment: {fila['Moviment (x, y, z)']} |\n"
            f"Màquina: {fila['Estat màquina']} | "
        )

    elif 113 < experimento < 120 or 122 < experimento: # planta
        titulo_experimento = (
            f"Moviment: {fila['Moviment (x, y, z)']} |\n"
            f"Fons escala: {fila['Fons Escala']} g | "
        )

    elif 119 < experimento < 123: # guitarra
        titulo_experimento = (
            f"Nota: {fila['Nota']} |\n"
            f"Fons escala: {fila['Fons Escala']} | "
        )

    if "Observacions" in fila.index and pd.notna(fila["Observacions"]):
        titulo_experimento += f"\nObs.: {fila['Observacions']}"

    return titulo_experimento



ventana_control_visibilidad = None


def crear_panel_visibilidad_tkinter(root, fig_ac, fig_frec, axs_ac, axs_frec, registros_curvas):
    """Crea un panel Tkinter con checkboxes de estilo Windows para mostrar/ocultar series."""

    global ventana_control_visibilidad

    if not registros_curvas:
        return

    if ventana_control_visibilidad is not None and ventana_control_visibilidad.winfo_exists():
        ventana_control_visibilidad.destroy()

    panel = tk.Toplevel(root)
    ventana_control_visibilidad = panel
    panel.title("Mostrar / ocultar series")
    panel.geometry("430x360")
    panel.minsize(380, 280)

    # Estilo nativo con ttk. En Windows suele usar "vista" o "xpnative".
    style = ttk.Style(panel)
    temas = style.theme_names()
    if "vista" in temas:
        style.theme_use("vista")
    elif "xpnative" in temas:
        style.theme_use("xpnative")

    style.configure("Titulo.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("Serie.TCheckbutton", font=("Segoe UI", 10), padding=(4, 6))
    style.configure("Boton.TButton", font=("Segoe UI", 9), padding=(8, 4))

    panel.columnconfigure(0, weight=1)
    panel.rowconfigure(0, weight=1)

    frame_principal = ttk.Frame(panel, padding=(18, 16, 18, 16))
    frame_principal.grid(row=0, column=0, sticky="nsew")
    frame_principal.columnconfigure(0, weight=1)
    frame_principal.rowconfigure(2, weight=1)

    titulo = ttk.Label(
        frame_principal,
        text="Series disponibles:",
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
        for registro in registros_curvas:
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

    canvas = tk.Canvas(contenedor, highlightthickness=0, borderwidth=0)
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

        check = ttk.Checkbutton(
            frame_scroll,
            text=registro["etiqueta_menu"],
            variable=registro["var"],
            command=lambda i=indice: aplicar_visibilidad(i),
            style="Serie.TCheckbutton",
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

def cargar_y_graficar_experimento(experimento: int, bloque: int):
    """
    Carga los datos del experimento y genera las figuras de aceleración y espectro.
    No aplica detrend: usa directamente las columnas originales de aceleración.
    """
    ruta_base, nombre_hoja, columnas_excel = obtener_configuracion_experimento(experimento)

    if not ruta_base.exists():
        raise FileNotFoundError(f"No se encontró la carpeta del experimento: {ruta_base}")

    carpetas = sorted([p for p in ruta_base.iterdir() if p.is_dir()])

    if not carpetas:
        raise FileNotFoundError(f"No hay subcarpetas de datos en: {ruta_base}")

    titulo_experimento = leer_titulo_experimento(
        experimento,
        nombre_hoja,
        columnas_excel
    )

    # Cierra figuras anteriores para que al buscar otro experimento no se acumulen ventanas.
    plt.close("all")

    ################################################## FIGURA ACELERACIÓN

    fig_ac, axs_ac = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_ac.suptitle(f"{titulo_experimento} \n\nAceleració en el domini del temps")

    axs_ac[0].set_title("Eix X")
    axs_ac[1].set_title("Eix Y")
    axs_ac[2].set_title("Eix Z")

    axs_ac[0].set_ylabel("Amplitud (g)")
    axs_ac[1].set_ylabel("Amplitud (g)")
    axs_ac[2].set_ylabel("Amplitud (g)")
    axs_ac[2].set_xlabel("Temps (s)")

    for ax in axs_ac:
        ax.grid(True)

    ################################################## FIGURA ESPECTRO

    fig_frec, axs_frec = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_frec.suptitle(f"{titulo_experimento} \nEspectre de la freqüència")

    axs_frec[0].set_title("Eix X")
    axs_frec[1].set_title("Eix Y")
    axs_frec[2].set_title("Eix Z")

    axs_frec[0].set_ylabel("Amplitud (g)")
    axs_frec[1].set_ylabel("Amplitud (g)")
    axs_frec[2].set_ylabel("Amplitud (g)")
    axs_frec[2].set_xlabel("Freqüència (Hz)")

    for ax in axs_frec:
        ax.grid(True)

    ################################################## ESCALAS COMUNES

    ymax_ac = 0.0
    ymax_frec = 0.0
    datos_cargados = 0
    registros_curvas = []

    ################################################## RECORRER CARPETAS

    for carpeta in carpetas:
        archivo_time = carpeta / f"timeblock{bloque}.txt"
        archivo_spectrum = carpeta / f"spectrum{bloque}.txt"

        if not archivo_time.exists() or not archivo_spectrum.exists():
            print(f"No se encontraron los archivos en {carpeta.name}")
            continue

        try:
            acceleration_data = np.loadtxt(archivo_time)
            spectrum_dsp_data = np.loadtxt(archivo_spectrum)
        except Exception as e:
            print(f"Error leyendo archivos en {carpeta.name}: {e}")
            continue

        etiqueta = carpeta.name

        # Aceleración SIN detrend.
        x_ac = acceleration_data[:, 2]
        y_ac = acceleration_data[:, 3]
        z_ac = acceleration_data[:, 4]

        ymax_ac = max(
            ymax_ac,
            np.max(np.abs(x_ac)),
            np.max(np.abs(y_ac)),
            np.max(np.abs(z_ac))
        )

        t = np.arange(len(x_ac)) / Fs

        line_ac_x, = axs_ac[0].plot(t, x_ac, lw=0.5, label=etiqueta)
        line_ac_y, = axs_ac[1].plot(t, y_ac, lw=0.5, label=etiqueta)
        line_ac_z, = axs_ac[2].plot(t, z_ac, lw=0.5, label=etiqueta)

        # Espectro.
        if experimento == 120:
            NSAMPLES = len(t)
            F_max = 500

            ESPECTRO_DIM = int(F_max * NSAMPLES / Fs)

            f_dsp = np.arange(ESPECTRO_DIM) * Fs / NSAMPLES

            armonicosX_dsp = spectrum_dsp_data[:ESPECTRO_DIM, 0]
            armonicosY_dsp = spectrum_dsp_data[:ESPECTRO_DIM, 1]
            armonicosZ_dsp = spectrum_dsp_data[:ESPECTRO_DIM, 2]
        else:
            f_dsp = spectrum_dsp_data[:, 0]
            armonicosX_dsp = spectrum_dsp_data[:, 1]
            armonicosY_dsp = spectrum_dsp_data[:, 2]
            armonicosZ_dsp = spectrum_dsp_data[:, 3]

        ymax_frec = max(
            ymax_frec,
            np.max(np.abs(armonicosX_dsp)),
            np.max(np.abs(armonicosY_dsp)),
            np.max(np.abs(armonicosZ_dsp))
        )

        line_frec_x, = axs_frec[0].plot(f_dsp, armonicosX_dsp, lw=0.5, label=etiqueta)
        line_frec_y, = axs_frec[1].plot(f_dsp, armonicosY_dsp, lw=0.5, label=etiqueta)
        line_frec_z, = axs_frec[2].plot(f_dsp, armonicosZ_dsp, lw=0.5, label=etiqueta)

        registros_curvas.append(
            {
                "visible": True,
                "etiqueta_menu": etiqueta,
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

        datos_cargados += 1

    if datos_cargados == 0:
        plt.close(fig_ac)
        plt.close(fig_frec)
        raise FileNotFoundError(
            f"No s'ha pogut carregar cap par timeblock{bloque}.txt / spectrum{bloque}.txt."
        )

    ################################################## LEYENDAS Y AJUSTE FINAL

    axs_ac[0].legend(fontsize=8)
    axs_frec[0].legend(fontsize=8)

    ################################################## APLICAR MISMA ESCALA EN LOS SUBPLOTS

    if ymax_ac < 1e-12:
        ymax_ac = 1.0

    ymax_ac *= 1.10

    for ax in axs_ac:
        ax.set_ylim(-ymax_ac, ymax_ac)

    if ymax_frec < 1e-12:
        ymax_frec = 1.0

    ymax_frec *= 1.10

    for ax in axs_frec:
        ax.set_ylim(0, ymax_frec)

    fig_ac.tight_layout()
    fig_frec.tight_layout()

    crear_panel_visibilidad_tkinter(
        ventana,
        fig_ac,
        fig_frec,
        axs_ac,
        axs_frec,
        registros_curvas,
    )

    # No bloquea la ventana Tkinter.
    plt.show(block=False)

    return datos_cargados


################################################## INTERFAZ TKINTER

def buscar_experimento():
    texto_experimento = entrada_experimento.get().strip()
    texto_bloque = entrada_bloque.get().strip()

    if not texto_experimento:
        messagebox.showwarning("Camp buit", "Introdueix un número d'experiència")
        return

    if not texto_bloque:
        messagebox.showwarning("Camp buit", "Introdueix un número de bloc")
        return

    try:
        experimento = int(texto_experimento)
    except ValueError:
        messagebox.showerror(
            "Valor no vàlid",
            "El experimento ha de ser un número enter."
        )
        return

    try:
        bloque = int(texto_bloque)
    except ValueError:
        messagebox.showerror(
            "Valor no vàlid",
            "El bloc ha de ser un número enter."
        )
        return

    if bloque < 0:
        messagebox.showerror(
            "Valor no vàlid",
            "El bloc ha de ser major o igual que 0."
        )
        return

    etiqueta_estado.config(text=f"Carregant experiment {experimento}, bloc {bloque}...")
    ventana.update_idletasks()

    try:
        n_datos = cargar_y_graficar_experimento(experimento, bloque)
    except Exception as e:
        etiqueta_estado.config(text="Error al carregar l'experiment.")
        messagebox.showerror("Error", str(e))
        return

    etiqueta_estado.config(
        text=f"Experiment {experimento}, bloc {bloque} carregat correctament. Series carregades: {n_datos}."
    )


ventana = tk.Tk()
ventana.title("Buscador d'experiments")
ventana.geometry("520x200")
ventana.resizable(False, False)

frame = tk.Frame(ventana, padx=20, pady=20)
frame.pack(fill="both", expand=True)

etiqueta_experimento = tk.Label(frame, text="Experiment:")
etiqueta_experimento.grid(row=0, column=0, sticky="w")

entrada_experimento = tk.Entry(frame, width=15)
entrada_experimento.grid(row=0, column=1, padx=10, pady=5)
entrada_experimento.insert(0, "37")

etiqueta_bloque = tk.Label(frame, text="Bloc | Bloque:")
etiqueta_bloque.grid(row=1, column=0, sticky="w")

entrada_bloque = tk.Entry(frame, width=15)
entrada_bloque.grid(row=1, column=1, padx=10, pady=5)
entrada_bloque.insert(0, str(BLOQUE_POR_DEFECTO))

boton_buscar = tk.Button(frame, text="Buscar", command=buscar_experimento)
boton_buscar.grid(row=0, column=2, rowspan=2, padx=(10, 0))

etiqueta_estado = tk.Label(
    frame,
    text="Introdueix l'experiència i bloc, a continuació polsa Buscar.",
    anchor="w",
    justify="left"
)
etiqueta_estado.grid(row=2, column=0, columnspan=3, sticky="w", pady=(20, 0))

ventana.bind("<Return>", lambda event: buscar_experimento())

ventana.mainloop()
