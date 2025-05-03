import matplotlib.pyplot as plt
import folium
from folium import Circle, CircleMarker
import geopandas as gpd
import io
import base64

def create_choropleth_map(maps, inicial_count, primaria_count, secundaria_count):
    """Create a folium choropleth map with three layers for school levels"""
    # Merge counts with district geometries
    distrito_map = maps.copy()
    distrito_map = distrito_map.merge(inicial_count, on='UBIGEO', how='left')
    distrito_map = distrito_map.merge(primaria_count, on='UBIGEO', how='left')
    distrito_map = distrito_map.merge(secundaria_count, on='UBIGEO', how='left')
    
    # Fill NaN values with 0
    distrito_map[['inicial_count', 'primaria_count', 'secundaria_count']] = distrito_map[['inicial_count', 'primaria_count', 'secundaria_count']].fillna(0)
    
    # Simplify geometry to improve performance
    distrito_map = distrito_map.to_crs(epsg=4326)
    distrito_map['geometry'] = distrito_map['geometry'].simplify(tolerance=0.01)
    
    # Convert to GeoJSON format
    distrito_geojson = distrito_map.to_json()
    
    # Create base map
    peru_map = folium.Map(location=[-9.19, -75.0152], zoom_start=5, tiles='cartodbpositron')
    
    # Add choropleth layer for inicial schools
    folium.Choropleth(
        geo_data=distrito_geojson,
        name='Inicial',
        data=distrito_map,
        columns=['UBIGEO', 'inicial_count'],
        key_on='feature.properties.UBIGEO',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Número de Escuelas Inicial'
    ).add_to(peru_map)
    
    # Add choropleth layer for primaria schools
    folium.Choropleth(
        geo_data=distrito_geojson,
        name='Primaria',
        data=distrito_map,
        columns=['UBIGEO', 'primaria_count'],
        key_on='feature.properties.UBIGEO',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Número de Escuelas Primaria'
    ).add_to(peru_map)
    
    # Add choropleth layer for secundaria schools
    folium.Choropleth(
        geo_data=distrito_geojson,
        name='Secundaria',
        data=distrito_map,
        columns=['UBIGEO', 'secundaria_count'],
        key_on='feature.properties.UBIGEO',
        fill_color='Reds',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Número de Escuelas Secundaria'
    ).add_to(peru_map)
    
    # Add layer control
    folium.LayerControl().add_to(peru_map)
    
    return peru_map

def create_proximity_map(proximity_data):
    """Create a folium map showing proximity analysis for a region"""
    region = proximity_data['region']
    primaria = proximity_data['primaria']
    secundaria = proximity_data['secundaria']
    min_school = proximity_data['min_school']
    max_school = proximity_data['max_school']
    
    # Create a map centered on the region
    center = primaria.geometry.centroid.unary_union.centroid
    region_map = folium.Map(
        location=[center.y, center.x],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add terrain layer
    folium.TileLayer('Stamen Terrain', name='Terrain View').add_to(region_map)
    
    # Add minimum school (fewest nearby secondary schools)
    min_lat, min_lon = min_school.geometry.y, min_school.geometry.x
    folium.Marker(
        location=[min_lat, min_lon],
        popup=f"Primary school with fewest secondary schools:<br>{min_school['Nombre de SS.EE.']}<br>Count: {min_school['secondary_count']}",
        icon=folium.Icon(color='red', icon='home', prefix='fa')
    ).add_to(region_map)
    
    # Add 5km radius around minimum school
    Circle(
        location=[min_lat, min_lon],
        radius=5000,
        color='red',
        fill=True,
        fill_opacity=0.2
    ).add_to(region_map)
    
    # Add maximum school (most nearby secondary schools)
    max_lat, max_lon = max_school.geometry.y, max_school.geometry.x
    folium.Marker(
        location=[max_lat, max_lon],
        popup=f"Primary school with most secondary schools:<br>{max_school['Nombre de SS.EE.']}<br>Count: {max_school['secondary_count']}",
        icon=folium.Icon(color='green', icon='home', prefix='fa')
    ).add_to(region_map)
    
    # Add 5km radius around maximum school
    Circle(
        location=[max_lat, max_lon],
        radius=5000,
        color='green',
        fill=True,
        fill_opacity=0.2
    ).add_to(region_map)
    
    # Add secondary schools
    for idx, school in secundaria.iterrows():
        CircleMarker(
            location=[school.geometry.y, school.geometry.x],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"Secondary School: {school['Nombre de SS.EE.']}"
        ).add_to(region_map)
    
    # Add layer control
    folium.LayerControl().add_to(region_map)
    
    return region_map

def create_static_map(dataset, level, cmap='Reds', title=None):
    """Create a static choropleth map using GeoPandas for a specific school level"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the map
    dataset.plot(
        column=f'{level.lower()}_count',
        cmap=cmap,
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True
    )
    
    # Set title
    if title:
        ax.set_title(title, fontsize=15)
    else:
        ax.set_title(f'Distribution of {level} Schools by District', fontsize=15)
    
    # Remove axis
    ax.axis('off')
    
    # Convert plot to a base64 encoded image for Streamlit
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return image_data