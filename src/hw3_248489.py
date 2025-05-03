#!/usr/bin/env python
# coding: utf-8

# # Creación de env e instalación de paquetes

# ### Create new environment:
# 
# conda creare -name TAREA_TRES
# 
# ### Activate new environment
# 
# conda activate TAREA_TRES
# 
# ### Install package:
# 
# conda install -c conda-forge python=3 geopandas
# 
# ### Importar librerías

# In[29]:


# Manipulación de datos
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Visualización
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import seaborn as sns
from IPython.display import display, HTML

# Geodatos y análisis espacial
import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial.distance import cdist

# Mapas interactivos
import folium
from folium import Circle, CircleMarker
from folium.plugins import MarkerCluster

# Otros
import os


# ### Lectura de la bbdd de  instituciones

# In[30]:


cv_data = pd.read_html('listado_iiee.xls')[0]
print(cv_data.head())


# In[31]:


# Cambiamos el nombre de Ubigeo para poder hacer el merge luego
cv_data = cv_data.rename({'Ubigeo':'UBIGEO'}, axis =1 )
cv_data.head(5)


# ### Importación del Shape File por distritos

# In[32]:


maps = gpd.read_file('DISTRITOS.shp')
maps.head(5)


# In[33]:


# Nos quedamos con las columnas relevantes y cambiamos el nombre de IDDIST para hacer el merge luego
maps = maps[['IDDIST', 'geometry']]
maps = maps.rename({'IDDIST':'UBIGEO'}, axis =1 )
maps.head(5)


# In[34]:


# Para poder hacer el merge, ambas columnas deben estar en el mismo formato: Object or srting to int
maps['UBIGEO'] = maps['UBIGEO'].astype(str).astype(int)
maps.head(5)


# In[35]:


# Vemos los valores que tienen para saber como proceder con el filtrado
valores_unicos = cv_data['Nivel / Modalidad'].unique().tolist()
valores_unicos


# # TASK 1

# ### Creamos las funciones que nos ayudarán a filtrar instituciones

# In[36]:


# Función para filtrar escuelas por nivel
def filtrar_escuelas_por_nivel(cv_data, nivel):

    if nivel == 'Inicial':
        # Filtra todas las filas que contienen 'Inicial' en la columna Nivel_Modalidad
        base_nivel= cv_data[cv_data['Nivel / Modalidad'].str.contains('Inicial')]
    elif nivel == 'Primaria':
        # Filtra todas las filas que contienen 'Primaria' en la columna Nivel_Modalidad
        base_nivel= cv_data[cv_data['Nivel / Modalidad'].str.contains('Primaria')]
    elif nivel == 'Secundaria':
        # Filtra todas las filas que contienen 'Secundaria' en la columna Nivel_Modalidad
        base_nivel= cv_data[cv_data['Nivel / Modalidad'].str.contains('Secundaria')]
    else:
        raise ValueError(f"Nivel '{nivel}' no reconocido.")
    
    return base_nivel

# Función para contar escuelas por distrito
def contar_escuelas_por_distrito(base_nivel):
    # Agrupar y contar
    conteo = base_nivel.groupby(['Departamento', 'Provincia', 'Distrito', 'UBIGEO']).size().reset_index(name='Cantidad')
    print(f"Tamaño antes del merge: {conteo.shape}")
    #Saber si UBIGEO es único antes del merge
    print(f"UBIGEO no tiene repretidos: {conteo['UBIGEO'].is_unique}")

    # Hacemos el merge para unir la geometría
    conteo_con_geom = maps.merge(conteo, how="inner", on="UBIGEO")
    print(f"Tamaño después del merge: {conteo_con_geom.shape}")

    # Calcular total
    total_escuelas = conteo_con_geom['Cantidad'].sum()

    # Mostrar un resumen
    print(f"Total de escuelas: {total_escuelas}")
    display(conteo_con_geom[['Departamento', 'Provincia', 'Distrito', 'Cantidad', 'geometry']])  # Mostrar primeras 10 filas
    return conteo_con_geom, total_escuelas


# ### Mapa estático para INICIAL

# Creamos la base de datos

# In[37]:


nivel = 'Inicial'  # O el nivel que quieras
base_inicial = filtrar_escuelas_por_nivel(cv_data, nivel)
conteo_con_geom, total_escuelas = contar_escuelas_por_distrito(base_inicial)


# Creamos el histograma y el mapa estático

# In[38]:


import matplotlib.pyplot as plt

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Graficar el mapa con la variable 'Cantidad'
conteo_con_geom.plot(column='Cantidad', cmap='Reds', 
                     ax=ax, legend=True, 
                     linestyle='-', edgecolor='gray')

# Añadir título
ax.set_title("Distribución de las Variables (Mapa)")

# Quitar ejes si deseas
ax.axis('off')

# Guardar el gráfico como imagen PNG
plt.savefig("Distribución de escuelas iniciales.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()



# ### Mapa estático para PRIMARIA

# Creamos la base de datos

# In[39]:


nivel = 'Primaria'  # O el nivel que quieras
base_primaria = filtrar_escuelas_por_nivel(cv_data, nivel)
conteo_con_geom, total_escuelas = contar_escuelas_por_distrito(base_primaria)


# Creamos el histograma y el mapa estático

# In[40]:


import matplotlib.pyplot as plt

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Graficar el mapa con la variable 'Cantidad'
conteo_con_geom.plot(column='Cantidad', cmap='Reds', 
                     ax=ax, legend=True, 
                     linestyle='-', edgecolor='gray')

# Añadir título
ax.set_title("Distribución de las Variables (Mapa)")

# Quitar ejes si deseas
ax.axis('off')

# Guardar el gráfico como imagen PNG
plt.savefig("Distribución de escuelas primarias.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()



# ### Mapa estático para SECUNDARIA

# Creamos la base de datos

# In[41]:


nivel = 'Secundaria'  # O el nivel que quieras
base_secundaria = filtrar_escuelas_por_nivel(cv_data, nivel)
conteo_con_geom, total_escuelas = contar_escuelas_por_distrito(base_secundaria)


# Creamos el histograma y el mapa estático

# In[42]:


import matplotlib.pyplot as plt

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Graficar el mapa con la variable 'Cantidad'
conteo_con_geom.plot(column='Cantidad', cmap='Reds', 
                     ax=ax, legend=True, 
                     linestyle='-', edgecolor='gray')

# Añadir título
ax.set_title("Distribución de las Variables (Mapa)")

# Quitar ejes si deseas
ax.axis('off')

# Guardar el gráfico como imagen PNG
plt.savefig("Distribución de escuelas secundarias.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()



# <h1>TASK 2</h1>

# Creamos la base de datos de Primarias solo para Huancavelica y Huancayo

# In[43]:


dataset_cv = pd.merge(maps, cv_data, how="inner", on="UBIGEO")

# First, create a copy of the dataset to avoid modifying the original
school_data = dataset_cv.copy()

# Create a new column for school level
school_data['nivel_escolar'] = np.nan  # Initialize with NaN values

# Classify schools based on 'Nivel / Modalidad'
school_data.loc[school_data['Nivel / Modalidad'].str.contains('Inicial', case=False, na=False), 'nivel_escolar'] = 'Inicial'
school_data.loc[school_data['Nivel / Modalidad'].str.contains('Primaria', case=False, na=False), 'nivel_escolar'] = 'Primaria'
school_data.loc[school_data['Nivel / Modalidad'].str.contains('Secundaria', case=False, na=False), 'nivel_escolar'] = 'Secundaria'

# Filter the target regions
target_regions = ['HUANCAVELICA', 'AYACUCHO']
regional_schools = school_data[school_data['Departamento'].isin(target_regions)].copy()

# Convert latitude and longitude to numeric values
regional_schools['Latitud'] = pd.to_numeric(regional_schools['Latitud'], errors='coerce')
regional_schools['Longitud'] = pd.to_numeric(regional_schools['Longitud'], errors='coerce')

# Remove rows with missing coordinates
valid_schools = regional_schools.dropna(subset=['Latitud', 'Longitud']).copy()

# Create point geometry from coordinates
valid_schools['school_point'] = valid_schools.apply(
    lambda row: Point(row['Longitud'], row['Latitud']), axis=1
)

# Create a GeoDataFrame
school_gdf = gpd.GeoDataFrame(valid_schools, geometry='school_point', crs="EPSG:4326")

# Project to UTM for accurate distance calculations
school_gdf = school_gdf.to_crs(epsg=32718)  # UTM Zone 18S for Peru

# Count schools by level and region
print("School counts by region and level:")
print(school_gdf.groupby(['Departamento', 'nivel_escolar']).size())


# ### Análisis para HUANCAVELICA

# In[44]:


# Filter schools in Huancavelica
huancavelica_schools = school_gdf[school_gdf['Departamento'] == 'HUANCAVELICA'].copy()

# Separate by level
huancavelica_primary = huancavelica_schools[huancavelica_schools['nivel_escolar'] == 'Primaria'].copy()
huancavelica_secondary = huancavelica_schools[huancavelica_schools['nivel_escolar'] == 'Secundaria'].copy()

print(f"Number of primary schools in Huancavelica: {len(huancavelica_primary)}")
print(f"Number of secondary schools in Huancavelica: {len(huancavelica_secondary)}")

# Create 5km buffer around each primary school
# The geometry is already the school point (no need to calculate centroids)
huancavelica_primary['buffer_5km'] = huancavelica_primary['school_point'].buffer(5000)

# Count secondary schools within each buffer
nearby_counts = []

for idx, primary in huancavelica_primary.iterrows():
    buffer = primary['buffer_5km']
    # Count secondary schools within the buffer
    count = sum(huancavelica_secondary['school_point'].within(buffer))
    nearby_counts.append(count)

huancavelica_primary['secondary_count'] = nearby_counts

# Display statistics
print("\nHuancavelica statistics:")
print(f"Average secondary schools within 5km: {huancavelica_primary['secondary_count'].mean():.2f}")
print(f"Maximum secondary schools: {huancavelica_primary['secondary_count'].max()}")
print(f"Minimum secondary schools: {huancavelica_primary['secondary_count'].min()}")

# Find schools with minimum and maximum counts
huancavelica_min_school = huancavelica_primary.loc[huancavelica_primary['secondary_count'].idxmin()]
huancavelica_max_school = huancavelica_primary.loc[huancavelica_primary['secondary_count'].idxmax()]

print("\nPrimary school with fewest secondary schools nearby:")
print(f"Name: {huancavelica_min_school['Nombre de SS.EE.']}")
print(f"District: {huancavelica_min_school['Distrito']}")
print(f"Secondary schools within 5km: {huancavelica_min_school['secondary_count']}")

print("\nPrimary school with most secondary schools nearby:")
print(f"Name: {huancavelica_max_school['Nombre de SS.EE.']}")
print(f"District: {huancavelica_max_school['Distrito']}")
print(f"Secondary schools within 5km: {huancavelica_max_school['secondary_count']}")


# #### Huancavelica - Minimum Case

# In[45]:


# Plot for Huancavelica school with minimum secondary schools
# Get the primary school with minimum secondary schools
min_school = huancavelica_min_school
min_buffer = min_school['buffer_5km']

# Find secondary schools within the buffer
sec_in_buffer = huancavelica_secondary[huancavelica_secondary['school_point'].within(min_buffer)]

# Create figure
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot buffer (5km radius)
gpd.GeoSeries([min_buffer]).plot(
    ax=ax,
    color='lightblue',
    alpha=0.5,
    edgecolor='blue',
    linewidth=2
)

# Plot primary school
gpd.GeoSeries([min_school['school_point']]).plot(
    ax=ax,
    color='red',
    markersize=100,
    marker='*',
    label='Primary School'
)

# Plot secondary schools if any
if len(sec_in_buffer) > 0:
    gpd.GeoSeries(sec_in_buffer['school_point'].tolist()).plot(
        ax=ax,
        color='green',
        markersize=50,
        marker='o',
        label='Secondary Schools'
    )

# Title and styling
plt.title(
    f"Primary School with Fewest Nearby Secondary Schools\n"
    f"{min_school['Nombre de SS.EE.']} (HUANCAVELICA)",
    fontsize=14,
    pad=10
)

# Information box
plt.figtext(
    0.5, 0.05,
    f"Secondary schools within 5km: {min_school['secondary_count']}",
    ha='center',
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.legend(loc='lower right')
plt.axis('off')  # Hide axis

# Save the figure with high quality
plt.savefig(
    'huancavelica_min_secondary.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.5
)

plt.close()  # Close the figure to free memory
print("Plot saved as 'huancavelica_min_secondary.png'")


# #### Huancavelica - Maximum Case

# In[46]:


# Plot for Huancavelica school with maximum secondary schools
# Get the primary school with maximum secondary schools
max_school = huancavelica_max_school
max_buffer = max_school['buffer_5km']

# Find secondary schools within the buffer
sec_in_buffer = huancavelica_secondary[huancavelica_secondary['school_point'].within(max_buffer)]

# Create figure
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot buffer (5km radius)
gpd.GeoSeries([max_buffer]).plot(
    ax=ax,
    color='lightyellow',
    alpha=0.5,
    edgecolor='orange',
    linewidth=2
)

# Plot primary school
gpd.GeoSeries([max_school['school_point']]).plot(
    ax=ax,
    color='red',
    markersize=100,
    marker='*',
    label='Primary School'
)

# Plot secondary schools
gpd.GeoSeries(sec_in_buffer['school_point'].tolist()).plot(
    ax=ax,
    color='green',
    markersize=50,
    marker='o',
    label='Secondary Schools'
)

# Title and styling
plt.title(
    f"Primary School with Most Nearby Secondary Schools\n"
    f"{max_school['Nombre de SS.EE.']} (HUANCAVELICA)",
    fontsize=14,
    pad=10
)

# Information box
plt.figtext(
    0.5, 0.05,
    f"Secondary schools within 5km: {max_school['secondary_count']}",
    ha='center',
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.legend(loc='lower right')
plt.axis('off')  # Hide axis

# Save the figure with high quality
plt.savefig(
    'huancavelica_max_secondary.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.5
)

plt.close()  # Close the figure to free memory
print("Plot saved as 'huancavelica_max_secondary.png'")


# ### Análisis para AYACUCHO

# In[47]:


# Filter schools in Ayacucho
ayacucho_schools = school_gdf[school_gdf['Departamento'] == 'AYACUCHO'].copy()

# Separate by level
ayacucho_primary = ayacucho_schools[ayacucho_schools['nivel_escolar'] == 'Primaria'].copy()
ayacucho_secondary = ayacucho_schools[ayacucho_schools['nivel_escolar'] == 'Secundaria'].copy()

print(f"Number of primary schools in Ayacucho: {len(ayacucho_primary)}")
print(f"Number of secondary schools in Ayacucho: {len(ayacucho_secondary)}")

# Create 5km buffer around each primary school
ayacucho_primary['buffer_5km'] = ayacucho_primary['school_point'].buffer(5000)

# Count secondary schools within each buffer
nearby_counts = []

for idx, primary in ayacucho_primary.iterrows():
    buffer = primary['buffer_5km']
    # Count secondary schools within the buffer
    count = sum(ayacucho_secondary['school_point'].within(buffer))
    nearby_counts.append(count)

ayacucho_primary['secondary_count'] = nearby_counts

# Display statistics
print("\nAyacucho statistics:")
print(f"Average secondary schools within 5km: {ayacucho_primary['secondary_count'].mean():.2f}")
print(f"Maximum secondary schools: {ayacucho_primary['secondary_count'].max()}")
print(f"Minimum secondary schools: {ayacucho_primary['secondary_count'].min()}")

# Find schools with minimum and maximum counts
ayacucho_min_school = ayacucho_primary.loc[ayacucho_primary['secondary_count'].idxmin()]
ayacucho_max_school = ayacucho_primary.loc[ayacucho_primary['secondary_count'].idxmax()]

print("\nPrimary school with fewest secondary schools nearby:")
print(f"Name: {ayacucho_min_school['Nombre de SS.EE.']}")
print(f"District: {ayacucho_min_school['Distrito']}")
print(f"Secondary schools within 5km: {ayacucho_min_school['secondary_count']}")

print("\nPrimary school with most secondary schools nearby:")
print(f"Name: {ayacucho_max_school['Nombre de SS.EE.']}")
print(f"District: {ayacucho_max_school['Distrito']}")
print(f"Secondary schools within 5km: {ayacucho_max_school['secondary_count']}")


# #### Ayacucho - Minimum Case

# In[48]:


# Plot for Ayacucho school with minimum secondary schools
# Get the primary school with minimum secondary schools
min_school = ayacucho_min_school
min_buffer = min_school['buffer_5km']

# Find secondary schools within the buffer
sec_in_buffer = ayacucho_secondary[ayacucho_secondary['school_point'].within(min_buffer)]

# Create figure
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot buffer (5km radius)
gpd.GeoSeries([min_buffer]).plot(
    ax=ax,
    color='mistyrose',
    alpha=0.5,
    edgecolor='darkred',
    linewidth=2
)

# Plot primary school
gpd.GeoSeries([min_school['school_point']]).plot(
    ax=ax,
    color='purple',
    markersize=100,
    marker='*',
    label='Primary School'
)

# Plot secondary schools if any
if len(sec_in_buffer) > 0:
    gpd.GeoSeries(sec_in_buffer['school_point'].tolist()).plot(
        ax=ax,
        color='darkorange',
        markersize=50,
        marker='o',
        label='Secondary Schools'
    )

# Title and styling
plt.title(
    f"Primary School with Fewest Nearby Secondary Schools\n"
    f"{min_school['Nombre de SS.EE.']} (AYACUCHO)",
    fontsize=14,
    pad=10
)

# Information box
plt.figtext(
    0.5, 0.05,
    f"Secondary schools within 5km: {min_school['secondary_count']}",
    ha='center',
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.legend(loc='lower right')
plt.axis('off')  # Hide axis

# Save the figure with high quality
plt.savefig(
    'ayacucho_min_secondary.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.5
)

plt.close()  # Close the figure to free memory
print("Plot saved as 'ayacucho_min_secondary.png'")


# #### Ayacucho - Maximum Case

# In[49]:


# Plot for Ayacucho school with maximum secondary schools
# Get the primary school with maximum secondary schools
max_school = ayacucho_max_school
max_buffer = max_school['buffer_5km']

# Find secondary schools within the buffer
sec_in_buffer = ayacucho_secondary[ayacucho_secondary['school_point'].within(max_buffer)]

# Create figure
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot buffer (5km radius)
gpd.GeoSeries([max_buffer]).plot(
    ax=ax,
    color='honeydew',
    alpha=0.5,
    edgecolor='darkgreen',
    linewidth=2
)

# Plot primary school
gpd.GeoSeries([max_school['school_point']]).plot(
    ax=ax,
    color='purple',
    markersize=100,
    marker='*',
    label='Primary School'
)

# Plot secondary schools
gpd.GeoSeries(sec_in_buffer['school_point'].tolist()).plot(
    ax=ax,
    color='darkorange',
    markersize=50,
    marker='o',
    label='Secondary Schools'
)

# Title and styling
plt.title(
    f"Primary School with Most Nearby Secondary Schools\n"
    f"{max_school['Nombre de SS.EE.']} (AYACUCHO)",
    fontsize=14,
    pad=10
)

# Information box
plt.figtext(
    0.5, 0.05,
    f"Secondary schools within 5km: {max_school['secondary_count']}",
    ha='center',
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
)

plt.legend(loc='lower right')
plt.axis('off')  # Hide axis

# Save the figure with high quality
plt.savefig(
    'ayacucho_max_secondary.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.5
)

plt.close()  # Close the figure to free memory
print("Plot saved as 'ayacucho_max_secondary.png'")


# ### Resumen

# In[50]:


# Create a summary dataframe
summary_data = {
    'Region': ['HUANCAVELICA', 'HUANCAVELICA', 'AYACUCHO', 'AYACUCHO'],
    'Analysis Type': ['Minimum', 'Maximum', 'Minimum', 'Maximum'],
    'School Name': [
        huancavelica_min_school['Nombre de SS.EE.'], 
        huancavelica_max_school['Nombre de SS.EE.'], 
        ayacucho_min_school['Nombre de SS.EE.'], 
        ayacucho_max_school['Nombre de SS.EE.']
    ],
    'District': [
        huancavelica_min_school['Distrito'], 
        huancavelica_max_school['Distrito'], 
        ayacucho_min_school['Distrito'], 
        ayacucho_max_school['Distrito']
    ],
    'Secondary Schools (5km)': [
        huancavelica_min_school['secondary_count'], 
        huancavelica_max_school['secondary_count'], 
        ayacucho_min_school['secondary_count'], 
        ayacucho_max_school['secondary_count']
    ]
}

# Create dataframe
summary_df = pd.DataFrame(summary_data)

# Display summary
print("Summary of Results:")
print(summary_df)

# Calculate additional statistics
huancavelica_zeros = len(huancavelica_primary[huancavelica_primary['secondary_count'] == 0])
ayacucho_zeros = len(ayacucho_primary[ayacucho_primary['secondary_count'] == 0])

print("\nAdditional Insights:")
print(f"1. Primary schools in Huancavelica with no secondary schools within 5km: {huancavelica_zeros} ({huancavelica_zeros/len(huancavelica_primary)*100:.1f}%)")
print(f"2. Primary schools in Ayacucho with no secondary schools within 5km: {ayacucho_zeros} ({ayacucho_zeros/len(ayacucho_primary)*100:.1f}%)")


# # TASK 3

# ### PARTE 1: Choropleth Maps for All Districts

# In[51]:


maps = gpd.read_file('DISTRITOS.shp')
cv_data = pd.read_html('listado_iiee.xls')[0]

# Filtering by educational level
initial_schools = cv_data[cv_data['Nivel / Modalidad'].str.contains('Inicial', case=False, na=False)]
primary_schools = cv_data[cv_data['Nivel / Modalidad'].str.contains('Primaria', case=False, na=False)]
secondary_schools = cv_data[cv_data['Nivel / Modalidad'].str.contains('Secundaria', case=False, na=False)]

# Counting schools by district
initial_counts = initial_schools.groupby('Ubigeo').size().reset_index(name='initial_count')
primary_counts = primary_schools.groupby('Ubigeo').size().reset_index(name='primary_count')
secondary_counts = secondary_schools.groupby('Ubigeo').size().reset_index(name='secondary_count')

# Preparing data for merging
maps['IDDIST'] = maps['IDDIST'].astype(str)
initial_counts['Ubigeo'] = initial_counts['Ubigeo'].astype(str)
primary_counts['Ubigeo'] = primary_counts['Ubigeo'].astype(str)
secondary_counts['Ubigeo'] = secondary_counts['Ubigeo'].astype(str)

# Merging school counts with district geometries
maps = maps.merge(initial_counts, left_on='IDDIST', right_on='Ubigeo', how='left').drop(columns='Ubigeo')
maps = maps.merge(primary_counts, left_on='IDDIST', right_on='Ubigeo', how='left').drop(columns='Ubigeo')
maps = maps.merge(secondary_counts, left_on='IDDIST', right_on='Ubigeo', how='left').drop(columns='Ubigeo')

# Replacing NaN values with 0
maps.fillna(0, inplace=True)

# Simplifying geometry to improve performance
maps = maps.to_crs(epsg=4326)
maps['geometry'] = maps['geometry'].simplify(tolerance=0.01)

# Converting to GeoJSON format
districts_geojson = maps.to_json()

# Creating base map centered on Peru
peru_map = folium.Map(location=[-9.19, -75.0152], zoom_start=5, tiles='cartodbpositron')

# Function to add choropleth layers
def add_choropleth_layer(layer_name, data_column, color_scheme, legend_title):
    folium.Choropleth(
        geo_data=districts_geojson,
        name=layer_name,
        data=maps,
        columns=['IDDIST', data_column],
        key_on='feature.properties.IDDIST',
        fill_color=color_scheme,
        fill_opacity=0.7,
        line_opacity=0.8,
        line_color='black',
        legend_name=legend_title,
    ).add_to(peru_map)

# Adding the three educational level layers with different colors
add_choropleth_layer("Initial Education", "initial_count", "BuPu", "Initial Schools Count")
add_choropleth_layer("Primary Education", "primary_count", "OrRd", "Primary Schools Count")
add_choropleth_layer("Secondary Education", "secondary_count", "GnBu", "Secondary Schools Count")

# Adding layer control
folium.LayerControl().add_to(peru_map)

# Display the map directly in the notebook
display(peru_map)


# ### PARTE 2: Proximity maps for Huancavelica and Ayacucho

# In[52]:


# Loading district shapefile
maps = gpd.read_file('DISTRITOS.shp')
cv_data = pd.read_html('listado_iiee.xls')[0]
maps = maps.to_crs(epsg=4326)

# Loading educational institutions data

cv_data = cv_data.dropna(subset=["Latitud", "Longitud"]).copy()

# Creating geometry points from coordinates
cv_data["geometry"] = cv_data.apply(lambda row: Point(row["Longitud"], row["Latitud"]), axis=1)
schools_gdf = gpd.GeoDataFrame(cv_data, geometry="geometry", crs="EPSG:4326").to_crs(epsg=32718)
schools_gdf = schools_gdf.to_crs(epsg=4326)

def create_and_display_region_map(region_name):
    print(f"Generating map for {region_name}...")
    
    # Filter data for the specified region
    region_schools = schools_gdf[schools_gdf['Departamento'].str.upper() == region_name.upper()]
    
    # Filter by school type
    primaries = region_schools[region_schools['Nivel / Modalidad'].str.contains("Primaria", case=False, na=False)].copy()
    secondaries = region_schools[region_schools['Nivel / Modalidad'].str.contains("Secundaria", case=False, na=False)].copy()

    # Basic stats
    print(f"Number of primary schools in {region_name}: {len(primaries)}")
    print(f"Number of secondary schools in {region_name}: {len(secondaries)}")
    
    # Convert to UTM projection for buffer analysis
    primaries_utm = primaries.to_crs(epsg=32718)
    secondaries_utm = secondaries.to_crs(epsg=32718)

    # Create 5km buffers around primary schools
    primaries_utm['buffer_5km'] = primaries_utm.geometry.buffer(5000)
    secondary_counts = []
    secondary_ids = []

    # Count secondary schools within each buffer
    for idx, primary in primaries_utm.iterrows():
        buffer_geom = primary['buffer_5km']
        schools_within = secondaries_utm[secondaries_utm.geometry.within(buffer_geom)]
        secondary_counts.append(len(schools_within))
        secondary_ids.append(schools_within.index.tolist())

    # Add counts and IDs to original dataframe
    primaries['nearby_secondaries'] = secondary_counts
    primaries['secondary_school_ids'] = secondary_ids

    # Find minimum and maximum cases
    min_count = primaries['nearby_secondaries'].min()
    max_count = primaries['nearby_secondaries'].max()
    
    print(f"Minimum secondary schools within 5km: {min_count}")
    print(f"Maximum secondary schools within 5km: {max_count}")

    min_primaries = primaries[primaries['nearby_secondaries'] == min_count]
    max_primaries = primaries[primaries['nearby_secondaries'] == max_count]
    
    print(f"Number of primary schools with minimum ({min_count}) secondary schools: {len(min_primaries)}")
    print(f"Number of primary schools with maximum ({max_count}) secondary schools: {len(max_primaries)}")

    # Create map centered on the region
    center = primaries.geometry.centroid.unary_union.centroid
    region_map = folium.Map(location=[center.y, center.x], zoom_start=8)

    def add_schools(schools_group, marker_color, label):
        for _, school in schools_group.iterrows():
            lat, lon = school.geometry.y, school.geometry.x

            # Add marker for the primary school
            folium.Marker(
                location=[lat, lon],
                popup=f"{label}: {school['Nombre de SS.EE.']}<br>Nearby secondary schools: {school['nearby_secondaries']}",
                icon=folium.Icon(color=marker_color)
            ).add_to(region_map)

            # Add 5km radius circle
            Circle(
                location=[lat, lon],
                radius=5000,
                color=marker_color,
                fill=True,
                fill_opacity=0.1
            ).add_to(region_map)

            # Add nearby secondary schools (if any exist)
            if school['nearby_secondaries'] > 0:
                ids = school['secondary_school_ids']
                nearby_schools = secondaries.loc[ids]
                for _, sec_school in nearby_schools.iterrows():
                    CircleMarker(
                        location=[sec_school.geometry.y, sec_school.geometry.x],
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(region_map)

    # Add schools to the map
    add_schools(min_primaries, 'red', 'Primary with fewest secondary schools')
    add_schools(max_primaries, 'green', 'Primary with most secondary schools')
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 250px; height: 90px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color: white; padding: 10px;
        border-radius: 5px;">
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div>
            <div>Primary with fewest secondary schools ({0})</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: green; margin-right: 5px;"></div>
            <div>Primary with most secondary schools ({1})</div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 10px; height: 10px; background-color: blue; border-radius: 50%; margin-right: 5px;"></div>
            <div>Secondary schools</div>
        </div>
    </div>
    '''.format(min_count, max_count)
    
    region_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map directly in the notebook
    display(region_map)
    print(f"✅ {region_name} map displayed.")
    print("-" * 50)


# In[53]:


# Display maps for both regions
create_and_display_region_map('HUANCAVELICA')


# In[54]:


create_and_display_region_map('AYACUCHO')


# In[56]:


get_ipython().system('jupyter nbconvert --to script CODIGO_FINAL.ipynb')

