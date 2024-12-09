import pandas as pd
import numpy as np
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math




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


def analizar_distribucion_simple(serie, nombre_columna, alpha=0.05):
    """
    Analiza el tipo de distribución para una columna específica
    """
    # Visualización
    plt.figure(figsize=(15, 5))
    
    # Histograma con KDE
    plt.subplot(1, 3, 1)
    sns.histplot(data=serie, kde=True)
    plt.title(f'Distribución de {nombre_columna}')
    
    # Box Plot
    plt.subplot(1, 3, 2)
    sns.boxplot(data=serie)
    plt.title('Box Plot')
    
    # Q-Q Plot
    plt.subplot(1, 3, 3)
    stats.probplot(serie, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    
    # Tests de distribución
    # Test de normalidad
    _, normal_pval = stats.normaltest(serie)
    _, ks_normal_pval = stats.kstest(serie, 'norm')
    
    # Test para distribución exponencial
    _, ks_exp_pval = stats.kstest(serie, 'expon')
    
    # Test para distribución uniforme
    _, ks_unif_pval = stats.kstest(serie, 'uniform')
    
    # Estadísticos descriptivos
    print(f"\nAnálisis de distribución para {nombre_columna}")
    print("-" * 50)
    print(f"Media: {serie.mean():.4f}")
    print(f"Mediana: {serie.median():.4f}")
    print(f"Desviación estándar: {serie.std():.4f}")
    print(f"Asimetría: {serie.skew():.4f}")
    print(f"Kurtosis: {serie.kurtosis():.4f}")
    
    print("\nResultados de los tests:")
    print("-" * 50)
    print(f"Test de Normalidad (p-valor): {normal_pval:.4f}")
    print(f"KS test para Normal (p-valor): {ks_normal_pval:.4f}")
    print(f"KS test para Exponencial (p-valor): {ks_exp_pval:.4f}")
    print(f"KS test para Uniforme (p-valor): {ks_unif_pval:.4f}")
    
    # Determinar el tipo de distribución
    distribuciones = {
        'Normal': ks_normal_pval,
        'Exponencial': ks_exp_pval,
        'Uniforme': ks_unif_pval
    }
    
    mejor_dist = max(distribuciones.items(), key=lambda x: x[1])
    
    print("\nCaracterísticas de la distribución:")
    print("-" * 50)
    if serie.skew() > 0.5:
        print("- Asimetría positiva (cola hacia la derecha)")
    elif serie.skew() < -0.5:
        print("- Asimetría negativa (cola hacia la izquierda)")
    else:
        print("- Aproximadamente simétrica")
        
    if serie.kurtosis() > 0.5:
        print("- Leptocúrtica (más puntiaguda que la normal)")
    elif serie.kurtosis() < -0.5:
        print("- Platicúrtica (más plana que la normal)")
    else:
        print("- Mesocúrtica (similar a la normal)")
    
    print(f"\nLa distribución que mejor se ajusta es: {mejor_dist[0]}")
    if mejor_dist[1] < alpha:
        print("Nota: Ninguna distribución se ajusta bien a los datos")
    
    return {
        'mejor_distribucion': mejor_dist[0],
        'p_valores': distribuciones,
        'estadisticos': {
            'media': serie.mean(),
            'mediana': serie.median(),
            'std': serie.std(),
            'skewness': serie.skew(),
            'kurtosis': serie.kurtosis()
        }
    }


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


def plot_distribution_combo(data, category_col, value_col, 
                          figsize=(6, 10),
                          title=None,
                          palette="Set1",
                          violin_width=0.6,
                          box_width=0.20,
                          bw_method=0.10,
                          scatter_size=10,
                          grid_alpha=0.2,
                          ax=None):
    """
    Crea un gráfico combinado de violín, caja y dispersión.
    
    Parámetros:
    -----------
    (igual que antes, más ax)
    ax : matplotlib.axes.Axes, opcional
        Un eje de Matplotlib sobre el que dibujar el gráfico.
    """
    # Configuración inicial
    sns.set_style("ticks")
    if ax is None:  # Crear una figura si no se proporciona un eje
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None  # La figura no se devuelve si se proporciona un eje externo

    # Crear paleta de colores
    categories = data[category_col].unique()
    if isinstance(palette, str):
        palette = sns.color_palette(palette, len(categories))
    
    # Preparar datos
    categories_data = {cat: group[value_col].values 
                      for cat, group in data.groupby(category_col)}
    
    # Posiciones base
    positions = np.arange(1, len(categories) + 1)
    
    # Crear gráficos
    for idx, (cat_name, measurements) in enumerate(categories_data.items()):
        pos = positions[idx]
        
        # Violin plot
        vp = ax.violinplot(measurements,
                          positions=[pos - 0.15],
                          showmeans=False,
                          showextrema=False,
                          showmedians=False,
                          vert=True,
                          widths=violin_width,
                          bw_method=bw_method)
        
        # Estilizar violín
        for body in vp['bodies']:
            body.set_alpha(0.5)
            body.set_color(palette[idx])
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], None, np.mean(vertices[:, 0]))
        
        # Box plot
        bp = ax.boxplot(measurements,
                       positions=[pos],
                       patch_artist=True,
                       vert=True,
                       widths=box_width,
                       showfliers=True,
                       notch=True,
                       boxprops=dict(facecolor='white', color='black', linewidth=1),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(linewidth=1),
                       capprops=dict(linewidth=1),
                       flierprops=dict(marker='o', markerfacecolor='red', alpha=0.5))
        
        bp['boxes'][0].set_facecolor(palette[idx])
        bp['boxes'][0].set_alpha(0.5)
        
        # Scatter plot
        x_jitter = np.random.uniform(pos + 0.15, pos + 0.40, size=len(measurements))
        ax.scatter(x_jitter, measurements, s=scatter_size, color=palette[idx], 
                  alpha=0.6, rasterized=True)
    
    # Configurar diseño
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel(value_col)
    if title:
        ax.set_title(title)
    
    # Optimizar diseño
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.5, len(categories) + 0.5)
    ax.grid(True, axis='y', alpha=grid_alpha, color='black')
    
    plt.tight_layout()
    
    return fig, ax



def crear_multiple_boxplots(
    df, 
    ncols=4, 
    figsize_width=4, 
    figsize_height=12, 
    color_palette="Set2",
    include_numeric_only=True,
    grid=True,
    titulo_prefijo="Boxplot de ",
    flier_marker='o',
    flier_size=8,
    median_color='black',
    custom_columns=None
):
    """
    Crea múltiples boxplots para las columnas de un DataFrame.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos a graficar
    ncols : int, opcional (default=4)
        Número de columnas en la disposición de los gráficos
    figsize_width : int, opcional (default=4)
        Ancho de cada subplot
    figsize_height : int, opcional (default=12)
        Alto de cada fila de subplots
    color_palette : str, opcional (default="Set2")
        Nombre de la paleta de colores de seaborn
    include_numeric_only : bool, opcional (default=True)
        Si es True, solo incluye columnas numéricas
    grid : bool, opcional (default=True)
        Si es True, muestra la cuadrícula en los gráficos
    titulo_prefijo : str, opcional (default="Boxplot de ")
        Prefijo para el título de cada gráfico
    flier_marker : str, opcional (default='o')
        Marcador para los outliers
    flier_size : int, opcional (default=8)
        Tamaño de los marcadores de outliers
    median_color : str, opcional (default='black')
        Color de la línea de la mediana
    custom_columns : list, opcional (default=None)
        Lista de columnas específicas a graficar. Si es None, usa todas las columnas numéricas
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figura de matplotlib con todos los boxplots
    axes : numpy.ndarray
        Array con los ejes de cada subplot
    """
    try:
        # Seleccionar columnas a graficar
        if custom_columns is not None:
            cols_to_plot = custom_columns
            if not all(col in df.columns for col in cols_to_plot):
                raise ValueError("Algunas columnas especificadas no existen en el DataFrame")
        else:
            if include_numeric_only:
                cols_to_plot = df.select_dtypes(include=[np.number]).columns
            else:
                cols_to_plot = df.columns
        
        num_cols = len(cols_to_plot)
        if num_cols == 0:
            raise ValueError("No hay columnas para graficar")
            
        # Calcular el número de filas necesarias
        nrows = -(-num_cols // ncols)  # Redondeo hacia arriba
        
        # Crear la figura y los subplots
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=ncols, 
            figsize=(figsize_width * ncols, figsize_height * nrows)
        )
        
        # Asegurarse de que axes sea un array incluso si solo hay un subplot
        if num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
        
        # Obtener la paleta de colores
        colors = sns.color_palette(color_palette, num_cols)
        
        # Crear los boxplots
        for i, col in enumerate(cols_to_plot):
            # Convertir la columna a numérica
            datos_columna = pd.to_numeric(df[col], errors='coerce')
            
            if datos_columna.notna().any():  # Verificar que haya datos válidos
                axes[i].boxplot(
                    datos_columna,
                    vert=True,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[i]),
                    medianprops=dict(color=median_color),
                    flierprops=dict(marker=flier_marker, markersize=flier_size)
                )
                axes[i].set_title(f'{titulo_prefijo}{col}')
                axes[i].set_xlabel('Valor')
                if grid:
                    axes[i].grid()
        
        # Ocultar los subplots vacíos
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Ajustar el diseño
        plt.tight_layout()
        
        return fig, axes
        
    except Exception as e:
        print(f"Error al crear los boxplots: {str(e)}")
        return None, None
    



def plot_distributions(data, 
                       nrows=None, 
                       ncols=None, 
                       figsize=None, 
                       color_palette="Set2", 
                       title='Distribuciones del Dataset',
                       bins='sqrt',
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5):
    """
    Genera histogramas para todas las columnas numéricas de un dataset.
    
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
    title : str, opcional
        Título general de la figura
    bins : int o str, opcional
        Número de bins o método de cálculo (por defecto 'sqrt')
    alpha : float, opcional
        Transparencia de los histogramas
    edgecolor : str, opcional
        Color de los bordes de los histogramas
    linewidth : float, opcional
        Ancho de los bordes de los histogramas
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figura con los histogramas generados
    """
    plt.close('all')  # Cerrar todas las figuras previas
    
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
    
    # Generar histogramas
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Histograma
            axes[i].hist(data[col], bins=bins, color=colors[i], 
                         alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
            
            # Estética
            axes[i].set_title(col, fontsize=12, pad=5)
            axes[i].grid(axis='y', linestyle='--', linewidth=0.6, 
                         color='gray', alpha=0.5)
            axes[i].set_facecolor('white')
            
            # Configurar bordes
            for spine in ['top', 'right', 'left', 'bottom']:
                axes[i].spines[spine].set_color('black')
                axes[i].spines[spine].set_linewidth(0.5)
            
            # Configurar ticks
            axes[i].tick_params(axis='x', rotation=45, labelsize=8, colors='black')
            axes[i].tick_params(axis='y', labelsize=8, colors='black')

    # Eliminar ejes sobrantes
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    # Título general y ajuste
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1)
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
                axes[i].set_xlabel('Valor')
                
                # Configurar grid de manera segura
                if grid:
                    axes[i].grid(True, **grid_style)
                
                axes[i].set_facecolor('white')
                
                # Configurar bordes
                for spine in ['top', 'right', 'left', 'bottom']:
                    axes[i].spines[spine].set_color('black')
                    axes[i].spines[spine].set_linewidth(0.5)
                
                # Configurar ticks
                axes[i].tick_params(axis='x', rotation=45, labelsize=8, colors='black')
                axes[i].tick_params(axis='y', labelsize=8, colors='black')
    # Eliminar ejes sobrantes
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    
    # Configurar título principal si se proporciona
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    # Ajuste
    plt.tight_layout()
    
    return fig


def triangular_pairplot(data, hue=None, title="Triangular Pairplot", 
                        palette="husl", kind="scatter", 
                        diag_kind="auto", height=2.5, aspect=1, 
                        corner=True, figsize=(15, 10),
                        xtick_rotation=45, ytick_rotation=0, 
                        tick_labelsize=8, tick_color='black'):
    """
    Genera un gráfico de dispersión (pairplot) con formato triangular y personalización de ticks.

    Parámetros:
    -----------
    - data: pandas.DataFrame
        Conjunto de datos a graficar.
    - hue: str, opcional
        Columna que define los colores de los gráficos (variable categórica).
    - title: str, opcional
        Título del gráfico.
    - palette: str, opcional
        Estilo de color para los gráficos.
    - kind: str, opcional
        Tipo de gráfico para las relaciones ('scatter', 'reg').
    - diag_kind: str, opcional
        Tipo de gráfico para la diagonal ('auto', 'kde', 'hist').
    - height: float, opcional
        Altura de cada gráfico.
    - aspect: float, opcional
        Relación de aspecto de cada gráfico.
    - corner: bool, opcional
        Si True, muestra solo la parte inferior del triángulo.
    - figsize: tuple, opcional
        Tamaño de la figura.
    - xtick_rotation: int, opcional
        Rotación de etiquetas del eje x.
    - ytick_rotation: int, opcional
        Rotación de etiquetas del eje y.
    - tick_labelsize: int, opcional
        Tamaño de la fuente para las etiquetas de los ejes.
    - tick_color: str, opcional
        Color de las etiquetas de los ejes.

    Retorna:
    --------
    seaborn.PairGrid
        Objeto de gráfico de dispersión configurado.
    """
    # Crear un objeto PairGrid
    pairplot = sns.pairplot(data, 
                            hue=hue, 
                            palette=palette, 
                            kind=kind, 
                            diag_kind=diag_kind, 
                            height=height, 
                            aspect=aspect, 
                            corner=corner)

    # Ajustar el tamaño del gráfico
    pairplot.fig.set_size_inches(figsize)

    # Configurar el título principal si se proporciona
    if title:
        pairplot.fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    # Configurar los ticks para cada subgráfico
    for ax in pairplot.axes.flat:
        if ax is not None:  # Ignorar subgráficos vacíos
            ax.tick_params(axis='x', rotation=xtick_rotation)
            ax.tick_params(axis='y', rotation=ytick_rotation)
    
    # Ajustar el diseño
    plt.tight_layout()

    return pairplot
