# app.py

# --- Librerías necesarias ---
import streamlit as st
import pandas as pd
import folium
from folium import Choropleth, LayerControl
import matplotlib.pyplot as plt
import seaborn as sns  # Añadido seaborn que faltaba
from streamlit_folium import folium_static
import numpy as np

# --- Configurar página de Streamlit ---
st.set_page_config(page_title="Análisis Geoespacial de Colegios en Perú", layout="wide")

# --- Cargar los datos ---
@st.cache_data
def load_data():
    # 1. Cargar colegios
    cv_data = pd.read_html('listado_iiee.xls')[0]
    
    # 2. Cargar shapefile de distritos
    maps = gpd.read_file('../shape_file/DISTRITOS.shp')
    
    # 3. Ajustes de formato
    maps = maps[['IDDIST', 'geometry']]
    maps = maps.rename({'IDDIST':'UBIGEO'}, axis=1)
    maps['UBIGEO'] = maps['UBIGEO'].astype(str).astype(int)
    
    # 4. Renombrar Ubigeo en cv_data para hacer merge
    cv_data = cv_data.rename({'Ubigeo':'UBIGEO'}, axis=1)
    
    return cv_data, maps

cv_data, maps = load_data()

# --- Función para filtrar escuelas por nivel ---
def filtrar_escuelas_por_nivel(cv_data, nivel):
    if nivel == 'Inicial':
        base_nivel = cv_data[cv_data['Nivel / Modalidad'].str.contains('Inicial')]
    elif nivel == 'Primaria':
        base_nivel = cv_data[cv_data['Nivel / Modalidad'].str.contains('Primaria')]
    elif nivel == 'Secundaria':
        base_nivel = cv_data[cv_data['Nivel / Modalidad'].str.contains('Secundaria')]
    else:
        raise ValueError(f"Nivel '{nivel}' no reconocido.")
    
    return base_nivel

# --- Función para contar escuelas por distrito ---
def contar_escuelas_por_distrito(base_nivel, maps):
    # Agrupar y contar
    conteo = base_nivel.groupby(['Departamento', 'Provincia', 'Distrito', 'UBIGEO']).size().reset_index(name='Cantidad')
    
    # Hacemos el merge para unir la geometría
    conteo_con_geom = maps.merge(conteo, how="inner", on="UBIGEO")
    
    # Calcular total
    total_escuelas = conteo_con_geom['Cantidad'].sum()
    
    return conteo_con_geom, total_escuelas

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🗂️ Descripción de Datos", "🗺️ Mapas Estáticos", "🌍 Mapas Dinámicos"])

# ============================================================
# Tab 1: Descripción de datos
# ============================================================

with tab1:
    st.header("🗂️ Descripción de Datos", help="Información detallada sobre los datos utilizados en este análisis")

    st.subheader("Unidad de Análisis")
    st.write("""
   La unidad de análisis de este proyecto corresponde a cada institución educativa en el territorio peruano.
   Cada una de estas instituciones está individualizada mediante sus coordenadas geográficas (latitud y longitud), 
   así como por el nivel educativo que ofrece (Inicial, Primaria o Secundaria).
   El estudio espacial agrupa las escuelas a nivel distrital con el fin de identificar patrones de acceso a la educación 
   en el ámbito local.
    """)

    st.subheader("Supuestos y Preprocesamiento")
    st.write("""
    - Solo se incluyeron colegios que cuentan con coordenadas geográficas válidas y completas.
    - Se categorizaron los colegios en tres niveles educativos: Inicial, Primaria y Secundaria.
    - Se excluyeron las instituciones sin clasificación de nivel o con errores de geolocalización.
    - La agregación distrital asume que los límites provistos por el INEI son actuales y precisos.
    - Para un mejor rendimiento, se simplificaron las geometrías de los distritos utilizando un factor de tolerancia de 0.01.
    - El análisis de proximidad utiliza un buffer de 5 km alrededor de cada escuela para identificar escuelas secundarias cercanas a las primarias.
    - Se realizó un proceso de limpieza para garantizar la compatibilidad de los formatos entre el shapefile y los datos de los colegios.
    """)


    st.subheader("Fuentes de Datos")
    st.write("""
    - **Base de datos de escuelas**: Ministerio de Educación del Perú (MINEDU). (https://sigmed.minedu.gob.pe/mapaeducativo/)
    - **Shapefile de distritos**: Instituto Nacional de Estadística e Informática (INEI).
    """)

    # Descripción de las variables
    st.subheader("Descripción de Variables")
    st.write("La base de datos 'listado_iiee.xls' contiene las siguientes variables:")
    
    # Crear dataframe con la descripción de cada variable
    variables_desc = pd.DataFrame({
        'Variable': [
            'Código Modular', 
            'Anexo', 
            'Nombre de SS.EE.', 
            'Nivel / Modalidad', 
            'Gestión', 
            'Unidad Ejecutora',
            'Ubigeo',
            'Departamento',
            'Provincia',
            'Distrito',
            'Centro Poblado',
            'Dirección',
            'Latitud',
            'Longitud'
        ],
        'Descripción': [
            'Código único que identifica a la institución educativa',
            'Código de anexo o subdivisión de la institución educativa',
            'Nombre oficial de la institución educativa',
            'Nivel educativo que imparte la institución (Inicial, Primaria, Secundaria, etc.)',
            'Tipo de gestión (Pública o Privada)',
            'Entidad responsable de la administración de la institución',
            'Código de ubicación geográfica de 6 dígitos (concatenación de departamento, provincia y distrito)',
            'Departamento donde se ubica la institución',
            'Provincia donde se ubica la institución',
            'Distrito donde se ubica la institución',
            'Centro poblado donde se ubica la institución',
            'Dirección física de la institución',
            'Coordenada geográfica de latitud',
            'Coordenada geográfica de longitud'
        ]
    })
    
    # Mostrar tabla con descripción de variables con un zoom más grande
    st.dataframe(variables_desc, use_container_width=True, height=450)
    
    # Mostrar una muestra de los datos
    with st.expander("Ver muestra de datos"):
        st.dataframe(cv_data.head(10), use_container_width=True)

    # Estadísticas básicas
    st.subheader("Estadísticas Básicas")
    
    # CORRECCIÓN: Conteo correcto de escuelas por nivel
    # Contar directamente cada nivel correctamente
    total = len(cv_data)
    iniciales = len(cv_data[cv_data['Nivel / Modalidad'].str.contains('Inicial', case=False, na=False)])
    primarias = len(cv_data[cv_data['Nivel / Modalidad'].str.contains('Primaria', case=False, na=False)])
    secundarias = len(cv_data[cv_data['Nivel / Modalidad'].str.contains('Secundaria', case=False, na=False)])
    
    
    # Crear columnas para métricas corregidas
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Colegios", total)
    col2.metric("Iniciales", iniciales)
    col3.metric("Primarios", primarias)
    col4.metric("Secundarios", secundarias)

    # Gráfico de distribución de colegios por departamento
    st.subheader("Distribución de Colegios por Departamento")
    dept_counts = cv_data['Departamento'].value_counts().reset_index()
    dept_counts.columns = ['Departamento', 'Cantidad']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(dept_counts['Departamento'], dept_counts['Cantidad'], color='skyblue')
    ax.set_xlabel('Número de Colegios')
    ax.set_title('Distribución de Colegios por Departamento')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# Tab 2: Mapas Estáticos
# ============================================================

with tab2:
    st.header("🗺️ Mapas Estáticos")

    # Poner los tres mapas en una fila
    col1, col2, col3 = st.columns(3)
    niveles = ['Inicial', 'Primaria', 'Secundaria']
    colores = ['Blues', 'Reds', 'Greens']

    for i, (nivel, color) in enumerate(zip(niveles, colores)):
        # Obtener dataframe filtrado y con conteo por distrito
        base_nivel = filtrar_escuelas_por_nivel(cv_data, nivel)
        conteo_con_geom, total_escuelas = contar_escuelas_por_distrito(base_nivel, maps)

        # Plot estático con tamaño más pequeño
        fig, ax = plt.subplots(figsize=(7, 7))
        conteo_con_geom.plot(
            column='Cantidad',
            cmap=color,
            linewidth=0.3,
            edgecolor='gray',
            legend=True,
            ax=ax,
            legend_kwds={'shrink': 0.6}
        )
        ax.set_title(f'Nivel {nivel}\n(Total: {total_escuelas})', fontsize=12)
        ax.axis('off')
        
        # Mostrar en la columna correspondiente
        with [col1, col2, col3][i]:
            st.subheader(f"Nivel {nivel}")
            st.pyplot(fig)


# ============================================================
# Tab 3: Mapas Dinámicos
# ============================================================

with tab3:
    st.header("🌍 Mapas Dinámicos")

    st.subheader("Distribución de colegios por nivel educativo (mapa interactivo)")

    niveles = ['Inicial', 'Primaria', 'Secundaria']
    conteos = {}

    # Preparar datos para cada nivel
    for nivel in niveles:
        base_nivel = filtrar_escuelas_por_nivel(cv_data, nivel)
        conteo = base_nivel.groupby('UBIGEO').size().reset_index(name='Total_Colegios')
        conteos[nivel] = conteo

    # Simplificar geometría para mejor rendimiento
    maps_simplified = maps.copy()
    maps_simplified = maps_simplified.to_crs(epsg=4326)
    maps_simplified['geometry'] = maps_simplified['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    
    # Convertir a formato GeoJSON
    maps_geojson = maps_simplified.to_json()

    # Crear el mapa base centrado en Perú
    m = folium.Map(location=[-12.04318, -77.02824], zoom_start=6, tiles='OpenStreetMap')

    # Colores para cada nivel
    colors = {
        'Inicial': 'BuPu',
        'Primaria': 'OrRd',
        'Secundaria': 'GnBu'
    }

    # Agregar capas choropleth para cada nivel
    for nivel, conteo in conteos.items():
        Choropleth(
            geo_data=maps_geojson,
            data=conteo,
            columns=['UBIGEO', 'Total_Colegios'],
            key_on='feature.properties.UBIGEO',
            fill_color=colors[nivel],
            fill_opacity=0.7,
            line_opacity=0.2,
            name=f"{nivel}",
            legend_name=f"Colegios de {nivel}"
        ).add_to(m)

    # Añadir control de capas
    LayerControl(collapsed=False).add_to(m)

    # Mostrar el mapa folium
    folium_static(m, width=1200, height=700)
    
    # Agregar análisis específico para HUANCAVELICA y AYACUCHO
    st.header("📊 Análisis de Proximidad: Escuelas Primarias y Secundarias")
    
    st.write("""
    A continuación, analizamos la proximidad entre escuelas primarias y secundarias en dos regiones: 
    HUANCAVELICA y AYACUCHO. Para cada escuela primaria, identificamos cuántas escuelas secundarias 
    se encuentran en un radio de 5 km.
    """)
    
    # Función para crear y mostrar el mapa de proximidad
    def analizar_region(region_name):
        # Crear copia del dataset para procesamiento
        school_data = cv_data.copy()
        
        # Crear una columna para el nivel escolar
        school_data['nivel_escolar'] = np.nan
        school_data.loc[school_data['Nivel / Modalidad'].str.contains('Inicial', case=False, na=False), 'nivel_escolar'] = 'Inicial'
        school_data.loc[school_data['Nivel / Modalidad'].str.contains('Primaria', case=False, na=False), 'nivel_escolar'] = 'Primaria'
        school_data.loc[school_data['Nivel / Modalidad'].str.contains('Secundaria', case=False, na=False), 'nivel_escolar'] = 'Secundaria'
        
        # Filtrar solo para la región especificada
        regional_schools = school_data[school_data['Departamento'] == region_name].copy()
        
        # Convertir latitud y longitud a valores numéricos
        regional_schools['Latitud'] = pd.to_numeric(regional_schools['Latitud'], errors='coerce')
        regional_schools['Longitud'] = pd.to_numeric(regional_schools['Longitud'], errors='coerce')
        
        # Eliminar filas con coordenadas faltantes
        regional_schools = regional_schools.dropna(subset=['Latitud', 'Longitud']).copy()
        
        # Crear puntos de geometría a partir de coordenadas
        from shapely.geometry import Point
        regional_schools['school_point'] = regional_schools.apply(
            lambda row: Point(row['Longitud'], row['Latitud']), axis=1
        )
        
        # Crear GeoDataFrame
        school_gdf = gpd.GeoDataFrame(regional_schools, geometry='school_point', crs="EPSG:4326")
        
        # Proyectar a UTM para cálculos de distancia precisos
        school_gdf = school_gdf.to_crs(epsg=32718)  # UTM Zone 18S para Perú
        
        # Filtrar por nivel
        primarias = school_gdf[school_gdf['nivel_escolar'] == 'Primaria'].copy()
        secundarias = school_gdf[school_gdf['nivel_escolar'] == 'Secundaria'].copy()
        
        # Estadísticas básicas
        st.write(f"**Número de escuelas primarias en {region_name}:** {len(primarias)}")
        st.write(f"**Número de escuelas secundarias en {region_name}:** {len(secundarias)}")
        
        # Crear buffer de 5km alrededor de cada escuela primaria
        primarias['buffer_5km'] = primarias['school_point'].buffer(5000)
        
        # Contar escuelas secundarias dentro de cada buffer
        nearby_counts = []
        for idx, primary in primarias.iterrows():
            buffer = primary['buffer_5km']
            count = sum(secundarias['school_point'].within(buffer))
            nearby_counts.append(count)
        
        primarias['secondary_count'] = nearby_counts
        
        # Mostrar estadísticas
        st.write(f"**Promedio de escuelas secundarias en un radio de 5km:** {primarias['secondary_count'].mean():.2f}")
        st.write(f"**Máximo de escuelas secundarias cercanas:** {primarias['secondary_count'].max()}")
        st.write(f"**Mínimo de escuelas secundarias cercanas:** {primarias['secondary_count'].min()}")
        
        # Encontrar escuelas con mínimo y máximo de secundarias cercanas
        min_school = primarias.loc[primarias['secondary_count'].idxmin()]
        max_school = primarias.loc[primarias['secondary_count'].idxmax()]
        
        st.write("**Escuela primaria con menos escuelas secundarias cercanas:**")
        st.write(f"  - Nombre: {min_school['Nombre de SS.EE.']}")
        st.write(f"  - Distrito: {min_school['Distrito']}")
        st.write(f"  - Escuelas secundarias en 5km: {min_school['secondary_count']}")
        
        st.write("**Escuela primaria con más escuelas secundarias cercanas:**")
        st.write(f"  - Nombre: {max_school['Nombre de SS.EE.']}")
        st.write(f"  - Distrito: {max_school['Distrito']}")
        st.write(f"  - Escuelas secundarias en 5km: {max_school['secondary_count']}")
        
        # Convertir de nuevo a latitud/longitud para el mapa
        primarias = primarias.to_crs(epsg=4326)
        secundarias = secundarias.to_crs(epsg=4326)
        
        # Crear mapa interactivo
        center_lat = primarias['Latitud'].mean()
        center_lon = primarias['Longitud'].mean()
        region_map = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='CartoDB positron')
        
        # Añadir escuelas con máximo y mínimo de secundarias cercanas
        
        # Añadir escuela primaria con mínimo
        min_lat, min_lon = min_school['Latitud'], min_school['Longitud']
        folium.Marker(
            location=[min_lat, min_lon],
            popup=f"<b>Primaria con MENOS secundarias:</b><br>{min_school['Nombre de SS.EE.']}<br>Secundarias cercanas: {min_school['secondary_count']}",
            icon=folium.Icon(color='red', icon='arrow-down', prefix='fa')
        ).add_to(region_map)
        
        # Añadir un círculo de 5km alrededor
        folium.Circle(
            location=[min_lat, min_lon],
            radius=5000,
            color='red',
            fill=True,
            fill_opacity=0.2
        ).add_to(region_map)
        
        # Añadir escuela primaria con máximo
        max_lat, max_lon = max_school['Latitud'], max_school['Longitud']
        folium.Marker(
            location=[max_lat, max_lon],
            popup=f"<b>Primaria con MÁS secundarias:</b><br>{max_school['Nombre de SS.EE.']}<br>Secundarias cercanas: {max_school['secondary_count']}",
            icon=folium.Icon(color='green', icon='arrow-up', prefix='fa')
        ).add_to(region_map)
        
        # Añadir un círculo de 5km alrededor
        folium.Circle(
            location=[max_lat, max_lon],
            radius=5000,
            color='green',
            fill=True,
            fill_opacity=0.2
        ).add_to(region_map)
        
        # Añadir todas las escuelas secundarias
        for idx, row in secundarias.iterrows():
            folium.CircleMarker(
                location=[row['Latitud'], row['Longitud']],
                radius=3,
                color='blue',
                fill=True,
                fill_opacity=0.7,
                popup=f"Secundaria: {row['Nombre de SS.EE.']}"
            ).add_to(region_map)
        
        # Añadir leyenda
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; width: 250px; z-index: 1000; background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <p><b>Leyenda:</b></p>
            <p><i class="fa fa-arrow-down" style="color:red"></i> Primaria con menos secundarias</p>
            <p><i class="fa fa-arrow-up" style="color:green"></i> Primaria con más secundarias</p>
            <p><span style="color:blue; font-size: 15px;">●</span> Escuelas secundarias</p>
            <p><span style="display:inline-block; height:10px; width:10px; background-color:red; opacity:0.2;"></span> Buffer 5km</p>
        </div>
        '''
        region_map.get_root().html.add_child(folium.Element(legend_html))
        
        return region_map
    
    # Crear y mostrar mapas para HUANCAVELICA y AYACUCHO
    tab_huancavelica, tab_ayacucho = st.tabs(["HUANCAVELICA", "AYACUCHO"])
    
    with tab_huancavelica:
        st.subheader("Análisis de Proximidad en HUANCAVELICA")
        huancavelica_map = analizar_region("HUANCAVELICA")
        folium_static(huancavelica_map, width=1000, height=600)
    
    with tab_ayacucho:
        st.subheader("Análisis de Proximidad en AYACUCHO")
        ayacucho_map = analizar_region("AYACUCHO")
        folium_static(ayacucho_map, width=1000, height=600)
