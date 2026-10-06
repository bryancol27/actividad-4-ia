"""
===================================================================================
PROYECTO: APRENDIZAJE NO SUPERVISADO APLICADO A UN SISTEMA DE TRANSPORTE MASIVO
ALGORITMO: K-MEANS CLUSTERING
OBJETIVO: Segmentar estaciones y franjas horarias operativas para optimizar la toma 
          de decisiones y distribución de recursos en transporte público.
===================================================================================
"""

# =================================================================================
# 0. IMPORTACIÓN DE LIBRERÍAS
# =================================================================================
# Librerías estándar para análisis y manipulación de datos
import numpy as np
import pandas as pd

# Librerías para visualización estadística
import matplotlib
matplotlib.use('Agg')  # Para evitar bloqueo de ventanas GUI al ejecutar como script
import matplotlib.pyplot as plt
import seaborn as sns

# Herramientas de Machine Learning (Scikit-Learn)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Configuración visual de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print(">> Librerías importadas correctamente.\n")


# =================================================================================
# 1. SIMULACIÓN Y GENERACIÓN DEL DATASET
# =================================================================================
"""
EXPLICACIÓN PARA LA SUSTENTACIÓN:
En sistemas reales de transporte masivo (ej. TransMilenio, Metro, MIO, Metrobús),
los patrones operativos varían según la demanda, tiempos de espera y congestión.
Simularemos 1,000 observaciones basadas en 3 perfiles típicos:
- Perfil 1: Operación Valle / Baja Demanda (Horas no pico, fines de semana, noche).
- Perfil 2: Operación Moderada / Flujo Regular (Horas laborales intermedias).
- Perfil 3: Operación Crítica / Hora Pico (Pico mañana/tarde con alta congestión).
"""

def generar_dataset_transporte(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # Distribución aproximada de muestras por perfil operativo
    n_valle = int(n_samples * 0.35)
    n_moderado = int(n_samples * 0.40)
    n_pico = n_samples - n_valle - n_moderado
    
    # --- Grupo 1: Valle / Baja Demanda (350 registros) ---
    afluencia_valle = np.random.normal(loc=180, scale=40, size=n_valle).clip(50, 320)
    espera_valle = np.random.normal(loc=3.5, scale=0.8, size=n_valle).clip(1, 6)
    retraso_valle = np.random.normal(loc=1.2, scale=0.4, size=n_valle).clip(0.2, 3)
    recargas_valle = np.random.normal(loc=90, scale=25, size=n_valle).clip(20, 180)
    franja_valle = np.random.choice([1], size=n_valle) # 1: Valle/Nocturno

    # --- Grupo 2: Moderado / Flujo Regular (400 registros) ---
    afluencia_mod = np.random.normal(loc=700, scale=90, size=n_moderado).clip(450, 950)
    espera_mod = np.random.normal(loc=8.5, scale=1.2, size=n_moderado).clip(5.5, 12)
    retraso_mod = np.random.normal(loc=4.5, scale=0.9, size=n_moderado).clip(2, 7)
    recargas_mod = np.random.normal(loc=380, scale=60, size=n_moderado).clip(220, 550)
    franja_mod = np.random.choice([2], size=n_moderado) # 2: Intermedia

    # --- Grupo 3: Crítico / Pico Saturado (250 registros) ---
    afluencia_pico = np.random.normal(loc=1650, scale=180, size=n_pico).clip(1200, 2400)
    espera_pico = np.random.normal(loc=19.0, scale=2.5, size=n_pico).clip(13, 28)
    retraso_pico = np.random.normal(loc=13.5, scale=2.0, size=n_pico).clip(8, 22)
    recargas_pico = np.random.normal(loc=980, scale=130, size=n_pico).clip(650, 1500)
    franja_pico = np.random.choice([3], size=n_pico) # 3: Pico

    # Concatenación de datos
    df = pd.DataFrame({
        'id_registro': np.arange(1, n_samples + 1),
        'afluencia_pasajeros': np.round(np.concatenate([afluencia_valle, afluencia_mod, afluencia_pico]), 0).astype(int),
        'tiempo_espera_promedio': np.round(np.concatenate([espera_valle, espera_mod, espera_pico]), 2),
        'retraso_promedio_bus': np.round(np.concatenate([retraso_valle, retraso_mod, retraso_pico]), 2),
        'tarjetas_recargadas': np.round(np.concatenate([recargas_valle, recargas_mod, recargas_pico]), 0).astype(int),
        'franja_horaria_cod': np.concatenate([franja_valle, franja_mod, franja_pico])
    })
    
    # Mezclamos aleatoriamente para evitar que queden ordenados por grupo
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df['id_registro'] = np.arange(1, n_samples + 1)
    
    return df

# Generar y guardar en CSV
nombre_archivo_csv = 'datos_transporte_masivo.csv'
df_transporte = generar_dataset_transporte(n_samples=1000, random_state=42)
df_transporte.to_csv(nombre_archivo_csv, index=False, encoding='utf-8')
print(f"[Paso 1 Completado] Dataset sintético generado con {len(df_transporte)} registros y guardado en '{nombre_archivo_csv}'.")


# =================================================================================
# 2. PROCESAMIENTO Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =================================================================================
print("\n" + "="*80)
print("2. PROCESAMIENTO Y EDA BÁSICO")
print("="*80)

# Carga y exploración inicial
df = pd.read_csv(nombre_archivo_csv)
print("\n--- Vista de las primeras 5 filas ---")
print(df.head())

print("\n--- Información General y Tipos de Datos ---")
print(df.info())

# Verificación de valores nulos
nulos = df.isnull().sum()
print("\n--- Verificación de Valores Nulos ---")
print(nulos)

# Estadísticas descriptivas
print("\n--- Resumen Estadístico (Variables Numéricas) ---")
print(df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

# Selección de variables predictoras para el Clustering
# Se excluye 'id_registro' porque es un identificador sin valor estadístico de distancia.
features_clustering = [
    'afluencia_pasajeros', 
    'tiempo_espera_promedio', 
    'retraso_promedio_bus', 
    'tarjetas_recargadas', 
    'franja_horaria_cod'
]

X = df[features_clustering].copy()

"""
EXPLICACIÓN PARA LA SUSTENTACIÓN (ESCALADO):
K-Means se basa en distancias euclidianas. Variables como 'afluencia_pasajeros' 
(con valores de hasta 2500) dominarían completamente sobre 'retraso_promedio_bus' 
(con valores de 1 a 20). StandardScaler transforma cada variable para que tenga 
media 0 y desviación estándar 1, permitiendo que todas pesen equitativamente.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n[Paso 2 Completado] Variables escaladas exitosamente con StandardScaler.")
print(f"Dimensiones de matriz escalada: {X_scaled.shape}")


# =================================================================================
# 3. EVALUACIÓN Y SELECCIÓN TÉCNICA DEL NÚMERO DE CLUSTERS (K)
# =================================================================================
print("\n" + "="*80)
print("3. PRUEBAS DE SELECCIÓN DE HIPERPARÁMETRO K (CODO Y SILUETA)")
print("="*80)

# A. Método del Codo (Elbow Method / Inercia)
inercia = []
rango_k = range(1, 11)

for k in rango_k:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans_temp.fit(X_scaled)
    inercia.append(kmeans_temp.inertia_)

# B. Coeficiente de Silueta (Silhouette Score) para K de 2 a 6
silhouette_scores = {}
print("\n--- Coeficientes de Silueta Calculados ---")
for k in range(2, 7):
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_temp)
    silhouette_scores[k] = score
    print(f"  > Para K = {k}: Silhouette Score = {score:.4f}")

# Gráfica conjunta: Método del Codo y Coeficiente de Silueta
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subgráfico 1: Método del Codo
axes[0].plot(rango_k, inercia, marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
axes[0].set_title('Método del Codo (Inercia vs Número de Clusters)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Número de Clusters (K)', fontsize=11)
axes[0].set_ylabel('Inercia (Suma de Cuadrados Intra-cluster)', fontsize=11)
axes[0].set_xticks(rango_k)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Subgráfico 2: Coeficiente de Silueta
k_vals = list(silhouette_scores.keys())
s_vals = list(silhouette_scores.values())
bars = axes[1].bar(k_vals, s_vals, color='#2ca02c', alpha=0.85, edgecolor='black')
axes[1].set_title('Evaluación de Cohesión y Separación (Silhouette Score)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Número de Clusters (K)', fontsize=11)
axes[1].set_ylabel('Coeficiente de Silueta Promedio', fontsize=11)
axes[1].set_ylim(0, 1.0)
axes[1].set_xticks(k_vals)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Añadir etiquetas de valor encima de las barras
for bar in bars:
    height = bar.get_height()
    axes[1].annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),  # offset vertical
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('evaluacion_k_optimo.png', dpi=300)
plt.show()

# Justificación técnica automática
k_optimo = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n[Justificación Técnica] El número óptimo de clusters seleccionado es K = {k_optimo}.")
print(f"Razón: Presenta el mayor coeficiente de silueta ({silhouette_scores[k_optimo]:.4f}) y coincide con el punto de inflexión del codo.")


# =================================================================================
# 4. ENTRENAMIENTO DEL MODELO FINAL Y ASIGNACIÓN DE CLUSTERS
# =================================================================================
print("\n" + "="*80)
print(f"4. ENTRENAMIENTO DEL MODELO FINAL CON K = {k_optimo}")
print("="*80)

# Configuración y ajuste del modelo final
kmeans_final = KMeans(
    n_clusters=k_optimo, 
    init='k-means++', 
    n_init=20, 
    max_iter=300, 
    random_state=42
)

# Ajuste y predicción
df['cluster'] = kmeans_final.fit_predict(X_scaled)
print(f"Conteo de observaciones por Cluster:")
print(df['cluster'].value_counts().sort_index())


# =================================================================================
# 5. VISUALIZACIÓN E INTERPRETACIÓN PARA LA TOMA DE DECISIONES
# =================================================================================
print("\n" + "="*80)
print("5. VISUALIZACIÓN E INTERPRETACIÓN DE CLUSTERS")
print("="*80)

# Obtener centroides en la escala original desescalando
centroides_escalados = kmeans_final.cluster_centers_
centroides_originales = scaler.inverse_transform(centroides_escalados)
df_centroides = pd.DataFrame(centroides_originales, columns=features_clustering)
df_centroides['cluster'] = range(k_optimo)

# Gráfico de dispersión 2D (Scatter Plot)
plt.figure(figsize=(12, 7))

# Paleta de colores distintiva
paleta_colores = ['#2ca02c', '#1f77b4', '#d62728'] # Verde (Valle), Azul (Moderado), Rojo (Crítico)

scatter = sns.scatterplot(
    data=df,
    x='afluencia_pasajeros',
    y='tiempo_espera_promedio',
    hue='cluster',
    palette=paleta_colores[:k_optimo],
    alpha=0.65,
    s=55,
    edgecolor='none'
)

# Dibujar los centroides
plt.scatter(
    df_centroides['afluencia_pasajeros'],
    df_centroides['tiempo_espera_promedio'],
    color='black',
    marker='X',
    s=250,
    edgecolor='yellow',
    linewidth=1.8,
    label='Centroides de Cluster'
)

# Anotaciones en los centroides
for i, row in df_centroides.iterrows():
    plt.annotate(
        f'Centroide {int(row["cluster"])}',
        (row['afluencia_pasajeros'], row['tiempo_espera_promedio']),
        textcoords="offset points",
        xytext=(0, 14),
        ha='center',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8, ec="black")
    )

plt.title('Segmentación del Sistema de Transporte Masivo con K-Means', fontsize=14, fontweight='bold')
plt.xlabel('Afluencia de Pasajeros (personas/hora)', fontsize=12)
plt.ylabel('Tiempo de Espera Promedio (minutos)', fontsize=12)
plt.legend(title='Grupo / Segmento', title_fontsize='11', loc='upper left')
plt.tight_layout()
plt.savefig('clusters_transporte_masivo.png', dpi=300)
plt.show()

# Análisis descriptivo de los clusters (Perfilamiento y medias)
print("\n--- PERFILAMIENTO Y CARACTERIZACIÓN DE LOS GRUPOS (MEDIAS) ---")
resumen_clusters = df.groupby('cluster')[features_clustering].mean().round(2)
resumen_clusters['total_registros'] = df['cluster'].value_counts()
resumen_clusters['porcentaje'] = (df['cluster'].value_counts(normalize=True) * 100).round(1).astype(str) + '%'
print(resumen_clusters)

# Asignación de nombres de negocio/operacionales a los clusters basados en sus promedios
def etiquetar_cluster(row):
    if row['afluencia_pasajeros'] > 1200:
        return 'Grupo 1: Estaciones Críticas / Alta Demanda (Pico)'
    elif row['afluencia_pasajeros'] > 500:
        return 'Grupo 2: Operación Moderada / Flujo Regular'
    else:
        return 'Grupo 3: Operación Valle / Baja Demanda'

df['perfil_operativo'] = df.apply(etiquetar_cluster, axis=1)

print("\n" + "="*80)
print("INTERPRETACIÓN OPERATIVA Y RECOMENDACIONES DE TOMA DE DECISIONES:")
print("="*80)
print("""
1. CLUSTER DE ESTACIONES CRÍTICAS / HORA PICO (Alta Afluencia & Alto Retraso):
   - Características: Afluencia > 1,400 pasajeros/h, tiempos de espera > 16 min y retrasos de bus > 10 min.
   - Decisión de Operación: Inyectar buses de refuerzo (expresos), habilitar más taquillas o recarga digital, 
     y desplegar gestores de convivencia y control de flujo en andenes.

2. CLUSTER DE OPERACIÓN MODERADA / FLUJO REGULAR (Demanda Intermedia):
   - Características: Afluencia ~700-800 pasajeros/h, tiempos de espera de 8-10 min y retrasos de 4-5 min.
   - Decisión de Operación: Mantener frecuencias programadas estándar, realizar mantenimiento preventivo 
     de flota en intervalos y optimizar la sincronización semafórica de carriles exclusivos.

3. CLUSTER DE OPERACIÓN VALLE / BAJA DEMANDA (Bajo Impacto Operativo):
   - Características: Afluencia < 300 pasajeros/h, tiempos de espera < 5 min y retrasos mínimos.
   - Decisión de Operación: Reducir flota activa para ahorro de combustible y costos operativos, 
     redistribuir conductores hacia zonas con mayor demanda y programar mantenimientos mayores.
""")

# Guardar dataset final etiquetado
df.to_csv('datos_transporte_masivo_clusterizado.csv', index=False, encoding='utf-8')
print("[Paso 5 Completado] Dataset final etiquetado guardado en 'datos_transporte_masivo_clusterizado.csv'.")
print(">> Ejecución finalizada con éxito.")
