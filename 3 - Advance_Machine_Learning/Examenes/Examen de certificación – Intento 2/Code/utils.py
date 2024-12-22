import pandas as pd
import numpy as np
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_na_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of non-missing values, missing values, and the percentage of missing values
    for each column in a DataFrame, and return them as a sorted DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame for which to calculate NA statistics.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns representing:
        - 'datos sin NAs en q': Number of non-missing values for each column
        - 'Na en q': Number of missing values for each column
        - 'Na en %': Percentage of missing values for each column, sorted in descending order.
    """
    qsna = df.shape[0] - df.isnull().sum(axis=0)
    qna = df.isnull().sum(axis=0)
    ppna = np.round(100 * (df.isnull().sum(axis=0) / df.shape[0]), 2)
    aux = {'datos sin NAs en q': qsna, 'Na en q': qna, 'Na en %': ppna}
    na = pd.DataFrame(data=aux)
    return na.sort_values(by='Na en %', ascending=False)

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Return True for outliers
    return (data < lower_bound) | (data > upper_bound)

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def limpiar_cadena(cadena):
    """
    Limpia una cadena de texto realizando las siguientes operaciones:
    1. Convierte todo el texto a minúsculas.
    2. Elimina caracteres no imprimibles antes de la primera letra y después de la última letra,
       pero mantiene los caracteres internos.
    3. Elimina paréntesis y su contenido al final de la cadena.
    
    Parámetros:
    - cadena (str): La cadena de texto a limpiar.
    
    Retorna:
    - str: La cadena limpia.
    """
    if isinstance(cadena, str):
        # 1. Convertir todo a minúsculas
        cadena = cadena.lower()
        
        # 2. Eliminar paréntesis y su contenido al final de la cadena
        cadena = re.sub(r'\s*\([^)]*\)\s*$', '', cadena)
        
        # 3. Eliminar caracteres no imprimibles antes de la primera letra y después de la última letra
        # Buscar la posición de la primera letra (a-z)
        primer_letra = re.search(r'[a-z]', cadena)
        # Buscar la posición de la última letra (a-z)
        ultima_letra = re.search(r'[a-z](?!.*[a-z])', cadena)
        
        if primer_letra and ultima_letra:
            inicio = primer_letra.start()
            fin = ultima_letra.end()
            cadena = cadena[inicio:fin]
        else:
            # Si no se encuentran letras, eliminar espacios en blanco
            cadena = cadena.strip()
        
        return cadena
    return cadena


def calcular_estadisticas(column, data):
    """
    Calcula estadísticas descriptivas para una columna numérica,
    omitiendo los valores nulos.

    Parámetros:
    - column (str): Nombre de la columna.
    - data (pd.Series): Serie de pandas con los datos de la columna.

    Retorna:
    - dict: Diccionario con las estadísticas calculadas.
    """
    estadisticas = {
        'Cuenta': int(np.sum(~np.isnan(data))),
        'Media': np.nanmean(data),
        'Mediana': np.nanmedian(data),
        'Desviación Estándar': np.nanstd(data, ddof=1),
        'Mínimo': np.nanmin(data),
        'Máximo': np.nanmax(data),
        '25% Percentil': np.nanpercentile(data, 25),
        '75% Percentil': np.nanpercentile(data, 75)
    }
    return estadisticas

def validar_tipos(df, diccionario):
    """
    Valida que cada columna en df tenga el tipo de dato especificado en diccionario.
    
    Parámetros:
    - df: DataFrame de pandas.
    - diccionario: Diccionario con columnas como llaves y tipos de datos como valores.
    
    Retorna:
    - mismatches: Lista de tuplas con (columna, tipo_actual, tipo_esperado) para discrepancias.
    """
    mismatches = []
    for columna, tipo_esperado in diccionario.items():
        if columna in df.columns:
            tipo_actual = str(df[columna].dtype)
            # Algunos dtypes pueden ser equivalentes pero diferentes en nombre
            # Por ejemplo, 'string' en pandas puede ser 'string[python]'
            # Comparar solo las partes relevantes
            if tipo_esperado.startswith('datetime') and tipo_actual.startswith('datetime'):
                continue  # Considerar igual si ambos son datetime
            elif tipo_actual != tipo_esperado:
                mismatches.append((columna, tipo_actual, tipo_esperado))
        else:
            mismatches.append((columna, 'No existe en el DataFrame', tipo_esperado))
    return mismatches


def graph_correlations(pearson, spearman, kendall, title="Correlation Heatmaps", 
                       cmap=['coolwarm', 'viridis', 'plasma'], 
                       figsize=(20, 16), 
                       annot_size=8):
    """
    Genera gráficos de correlación usando métodos Pearson, Spearman y Kendall
    
    Parámetros:
    - pearson: DataFrame de correlación de Pearson
    - spearman: DataFrame de correlación de Spearman
    - kendall: DataFrame de correlación de Kendall
    - title: Título general del gráfico
    - cmap: Paletas de color para cada mapa de calor
    - figsize: Tamaño de la figura
    - annot_size: Tamaño de la anotación de valores
    """
    # Crear máscara para la parte superior del triángulo
    mask_pearson = np.triu(np.ones_like(pearson, dtype=bool))
    mask_spearman = np.triu(np.ones_like(spearman, dtype=bool))
    mask_kendall = np.triu(np.ones_like(kendall, dtype=bool))
    
    # Crear figura
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    
    # Gráfico de Pearson
    sns.heatmap(
        pearson, 
        annot=True, 
        cmap=cmap[0],
        center=0, 
        ax=axs[0,0], 
        mask=mask_pearson,
        annot_kws={"size": annot_size},
        cbar_kws={"shrink": .8},
    )
    axs[0,0].set_title("Pearson Correlation", fontsize=12)
    
    # Gráfico de Spearman
    sns.heatmap(
        spearman, 
        annot=True, 
        cmap=cmap[1], 
        center=0, 
        ax=axs[0,1], 
        mask=mask_spearman,
        annot_kws={"size": annot_size},
        cbar_kws={"shrink": .8}
    )
    axs[0,1].set_title("Spearman Correlation", fontsize=12)
    
    # Gráfico de Kendall
    sns.heatmap(
        kendall, 
        annot=True, 
        cmap=cmap[2], 
        center=0, 
        ax=axs[1,0], 
        mask=mask_kendall,
        annot_kws={"size": annot_size},
        cbar_kws={"shrink": .8}
    )
    axs[1,0].set_title("Kendall Correlation", fontsize=12)
    
    # Remover el cuarto subplot
    fig.delaxes(axs[1,1])
    
    # Título general
    plt.suptitle(title, fontsize=16)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar gráfico
    plt.show()

def analizar_distribucion_avanzada(serie, nombre_columna, alpha=0.05):
    """
    Analiza diferentes tipos de distribuciones con múltiples pruebas estadísticas
    
    Parámetros:
    - serie: Serie de datos a analizar
    - nombre_columna: Nombre de la columna para etiquetas
    - alpha: Nivel de significancia para pruebas
    
    Retorna un diccionario con resultados del análisis
    """
    # Preprocesamiento
    serie = serie.dropna()
    
    # Visualización
    plt.figure(figsize=(20, 6))
    
    # Histograma con KDE
    plt.subplot(1, 4, 1)
    sns.histplot(data=serie, kde=True)
    plt.title(f'Distribución de {nombre_columna}')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    
    # Box Plot
    plt.subplot(1, 4, 2)
    sns.boxplot(x=serie)
    plt.title('Box Plot')
    
    # Q-Q Plot para Normal
    plt.subplot(1, 4, 3)
    stats.probplot(serie, dist="norm", plot=plt)
    plt.title('Q-Q Plot Normal')
    
    # Violin Plot
    plt.subplot(1, 4, 4)
    sns.violinplot(x=serie)
    plt.title('Violin Plot')
    
    plt.tight_layout()
    
    # Pruebas de distribución
    distribuciones = {
        'Normal': stats.kstest(serie, 'norm')[1],
        'Exponencial': stats.kstest(serie, 'expon')[1],
        'Uniforme': stats.kstest(serie, 'uniform')[1],
        'Log-Normal': stats.kstest(np.log(serie[serie > 0]), 'norm')[1],
        'Gamma': stats.kstest(serie[serie > 0], lambda x: stats.gamma.cdf(x, *stats.gamma.fit(serie[serie > 0])))[1],
        'Weibull': stats.kstest(serie[serie > 0], lambda x: stats.weibull_min.cdf(x, *stats.weibull_min.fit(serie[serie > 0])))[1]
    }
    
    # Pruebas adicionales
    shapiro_test = stats.shapiro(serie)
    anderson_test = stats.anderson(serie)
    
    # Estadísticos descriptivos
    descriptivos = {
        'media': serie.mean(),
        'mediana': serie.median(),
        'desv_est': serie.std(),
        'asimetria': serie.skew(),
        'kurtosis': serie.kurtosis(),
        'min': serie.min(),
        'max': serie.max()
    }
    
    # Selección de la mejor distribución
    mejor_dist = max(distribuciones.items(), key=lambda x: x[1])
    
    # Impresión de resultados
    print(f"\n📊 Análisis de Distribución para {nombre_columna}")
    print("-" * 50)
    
    print("\n🔍 Estadísticos Descriptivos:")
    for key, value in descriptivos.items():
        print(f"- {key.capitalize()}: {value:.4f}")
    
    print("\n📈 Pruebas de Distribución:")
    for dist, p_valor in distribuciones.items():
        print(f"- {dist}: p-valor = {p_valor:.4f}")
    
    print("\n⚖️ Características de Distribución:")
    if descriptivos['asimetria'] > 0.5:
        print("- Asimetría positiva (cola hacia la derecha)")
    elif descriptivos['asimetria'] < -0.5:
        print("- Asimetría negativa (cola hacia la izquierda)")
    else:
        print("- Distribución aproximadamente simétrica")
    
    if descriptivos['kurtosis'] > 0.5:
        print("- Distribución leptocúrtica (más puntiaguda)")
    elif descriptivos['kurtosis'] < -0.5:
        print("- Distribución platicúrtica (más plana)")
    else:
        print("- Distribución mesocúrtica (similar a normal)")
    
    print(f"\n🏆 Mejor distribución: {mejor_dist[0]}")
    if mejor_dist[1] < alpha:
        print("⚠️ Advertencia: Ninguna distribución se ajusta perfectamente")
    
    return {
        'mejor_distribucion': mejor_dist[0],
        'p_valores': distribuciones,
        'estadisticos': descriptivos,
        'shapiro_test': shapiro_test,
        'anderson_test': anderson_test
    }

def plot_categorical_distributions(data, 
                                   nrows=None, 
                                   ncols=None, 
                                   figsize=None, 
                                   color="skyblue", 
                                   edgecolor="black", 
                                   title="Distribuciones Categóricas del Dataset", 
                                   top_n=7, 
                                   alpha=0.8,
                                   palette="Set2", 
                                   grid=True):
    """
    Genera gráficos de barras para todas las columnas categóricas del dataset, mostrando las N categorías más frecuentes.

    Parámetros:
    -----------
    data : DataFrame
        Dataset que contiene los datos.
    nrows : int, opcional
        Número de filas en la cuadrícula de subplots.
    ncols : int, opcional
        Número de columnas en la cuadrícula de subplots.
    figsize : tuple, opcional
        Tamaño de la figura (ancho, alto).
    color : str, opcional
        Color de las barras del gráfico.
    edgecolor : str, opcional
        Color del borde de las barras.
    title : str, opcional
        Título general del gráfico.
    top_n : int, opcional
        Número de categorías más frecuentes a mostrar por columna.
    alpha : float, opcional
        Transparencia de las barras (0.0 a 1.0).
    grid : bool, opcional
        Si es True, muestra una cuadrícula en los gráficos.
    """
    import matplotlib.pyplot as plt
    import math

    plt.close('all')
    
    # Seleccionar columnas categóricas
    categorical_cols = data.select_dtypes(include=["category", "object"]).columns

    # Calcular dimensiones automáticamente si no se especifican
    if nrows is None or ncols is None:
        n_plots = len(categorical_cols)
        nrows = math.ceil(n_plots / math.ceil(math.sqrt(n_plots)))
        ncols = math.ceil(math.sqrt(n_plots))
    
    # Calcular tamaño de figura si no se especifica
    if figsize is None:
        figsize = (4*ncols, 3*nrows)

    # Crear subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    # Generar gráficos de barras
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            ax = axes[i]
            
            # Contar las categorías más frecuentes
            top_categories = data[col].value_counts().nlargest(top_n)
            
            # Crear gráfico de barras
            top_categories.plot(kind="bar", color=sns.color_palette(palette, len(top_categories)), 
                    edgecolor=edgecolor, alpha=alpha, ax=ax)
            ax.set_title(f"Top {top_n} en '{col}'", fontsize=12, pad=5)
            ax.set_xlabel("")  # Quitar el título del eje X
            ax.set_ylabel("Frecuencia")
            ax.tick_params(axis='x', rotation=45, labelsize=8, colors='black')
            ax.tick_params(axis='y', labelsize=8, colors='black')

            # Mostrar cuadrícula si está activada
            if grid:
                ax.grid(axis="y", linestyle="--", linewidth=0.6, color="gray", alpha=0.5)
            
            # Estética de bordes
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_color("black")
                ax.spines[spine].set_linewidth(0.5)

    # Eliminar ejes sobrantes
    for j in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[j])
    
    # Título general y ajuste
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    return fig

def plot_distributions2(data, 
                       nrows=None, 
                       ncols=None, 
                       figsize=None, 
                       color_palette="Set2", 
                       title='Distribuciones del Dataset',
                       bins='sqrt',
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5,
                       kde=False,
                       kde_kws=None,
                       show_iqr_and_legend=True):
    """
    Genera histogramas para todas las columnas numéricas de un dataset, incluyendo métricas IQR y KDE opcional.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import math

    # Validación del tipo de dato de 'data'
    if not isinstance(data, pd.DataFrame):
        raise TypeError("El argumento 'data' debe ser un DataFrame de pandas.")

    plt.close('all')

    # Seleccionar solo columnas numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Calcular dimensiones automáticamente si no se especifican
    if nrows is None or ncols is None:
        n_plots = len(numeric_cols)
        nrows = math.ceil(n_plots / math.ceil(math.sqrt(n_plots)))
        ncols = math.ceil(math.sqrt(n_plots))
    
    # Calcular tamaño de figura si no se especifica
    if figsize is None:
        figsize = (4*ncols, 3*nrows)
    
    # Crear subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    # Paleta de colores
    colors = sns.color_palette(color_palette, len(numeric_cols))
    
    # Generar histogramas con métricas IQR
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(
                data[col].dropna(),
                bins=bins,
                kde=kde,
                color=colors[i],
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                ax=ax,
                kde_kws=kde_kws if kde else {}
            )
            ax.set_xlabel("")
            
            if show_iqr_and_legend:
                Q1 = np.percentile(data[col].dropna(), 25)
                Q2 = np.percentile(data[col].dropna(), 50)
                Q3 = np.percentile(data[col].dropna(), 75)
                IQR = Q3 - Q1
                
                ax.axvline(Q1, color="blue", linestyle="--", linewidth=0.7, label=f"Q1 ({Q1:.2f})")
                ax.axvline(Q2, color="red", linestyle="--", linewidth=0.7, label=f"Q2 ({Q2:.2f})")
                ax.axvline(Q3, color="purple", linestyle="--", linewidth=0.7, label=f"Q3 ({Q3:.2f})")
                ax.axvspan(Q1, Q3, color="gray", alpha=0.1, label=f"IQR ({IQR:.2f})")
            
            ax.set_title(col, fontsize=12, pad=5)
            ax.grid(axis='y', linestyle='--', linewidth=0.6, color='gray', alpha=0.5)
            ax.set_facecolor('white')
            
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_color('black')
                ax.spines[spine].set_linewidth(0.5)
            ax.tick_params(axis='x', rotation=45, labelsize=8, colors='black')
            ax.tick_params(axis='y', labelsize=8, colors='black')

            if show_iqr_and_legend:
                ax.legend(fontsize=8, loc='upper right')

    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_boxplots(data, 
                  nrows=None, 
                  ncols=None, 
                  figsize=None, 
                  color_palette="Set2",
                  grid=False,
                  grid_style=None,
                  median_color='black',
                  box_alpha=0.7,
                  title=None):  # Nuevo parámetro:
    """
    Genera boxplots para todas las columnas numéricas de un dataset.
    
    Parámetros:
    -----------
    data : pandas.DataFrame
        Dataset a visualizar
    nrows : int, opcional
        Número de filas en la cuadrícula de subplots. 
        Si no se especifica, se calcula automáticamente.
    ncols : int, opcional
        Número de columnas en la cuadrícula de subplots. 
        Si no se especifica, se calcula automáticamente.
    figsize : tuple, opcional
        Tamaño de la figura (ancho, alto). 
        Si no se especifica, se calcula automáticamente.
    color_palette : str, opcional
        Paleta de colores de Seaborn a utilizar
    grid : bool, opcional
        Mostrar grid en los subplots
    grid_style : dict, opcional
        Estilos personalizados para la cuadrícula
    median_color : str, opcional
        Color de la línea de la mediana
    box_alpha : float, opcional
        Transparencia de las cajas de los boxplots
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figura con los boxplots generados
    """
    plt.close('all')  # Cerrar todas las figuras previas
    
    # Seleccionar solo columnas numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Calcular dimensiones automáticamente si no se especifican
    if nrows is None or ncols is None:
        n_plots = len(numeric_cols)
        ncols = int(np.ceil(np.sqrt(n_plots)))  # Ajuste cuadrado por defecto
        nrows = int(np.ceil(n_plots / ncols))   # Redondear hacia arriba
    
    # Calcular tamaño de figura si no se especifica
    if figsize is None:
        figsize = (4*ncols, 4*nrows)
    
    # Crear subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    # Paleta de colores
    colors = sns.color_palette(color_palette, len(numeric_cols))
    
    # Configuración por defecto de la cuadrícula
    default_grid_style = {
        'linestyle': '--', 
        'linewidth': 0.6, 
        'color': 'gray', 
        'alpha': 0.5
    }
    
    # Combinar estilos de cuadrícula
    grid_style = grid_style or default_grid_style
    
    # Generar boxplots
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Convertir a numérico, eliminando valores no numéricos
            datos_columna = pd.to_numeric(data[col], errors='coerce')
            
            # Graficar solo si hay datos válidos
            if datos_columna.dropna().shape[0] > 0:
                # Boxplot
                axes[i].boxplot(datos_columna.dropna(), 
                                vert=True, 
                                patch_artist=True, 
                                boxprops=dict(facecolor=colors[i], alpha=box_alpha),
                                medianprops=dict(color=median_color))
                
                
                # Estética
                axes[i].set_title(col, fontsize=12, pad=5)
                axes[i].set_xticks([])
                #axes[i].set_xlabel('')
                
                # Configurar grid de manera segura
                if grid:
                    axes[i].grid(True, **grid_style)
                
                axes[i].set_facecolor('white')
                
                # Configurar bordes
                for spine in ['top', 'right', 'left', 'bottom']:
                    axes[i].spines[spine].set_color('black')
                    axes[i].spines[spine].set_linewidth(0.5)
                
                # Configurar ticks
                #axes[i].tick_params(axis='x', rotation=45, labelsize=8, colors='black')
                axes[i].tick_params(axis='y', labelsize=8, colors='black')
    # Eliminar ejes sobrantes
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    
    # Configurar título principal si se proporciona
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1)
    # Ajuste
    plt.tight_layout()
    
    return fig