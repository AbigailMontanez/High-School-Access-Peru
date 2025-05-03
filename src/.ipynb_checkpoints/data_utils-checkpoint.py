import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def load_data():
    """Load and prepare the educational institutions dataset and shapefiles"""
    try:
        # Load school data
        cv_data = pd.read_html('listado_iiee.xls')[0]
        # Load district shapefiles
        maps = gpd.read_file('DISTRITOS.shp')
        
        # Rename Ubigeo column for consistency
        cv_data = cv_data.rename({'Ubigeo':'UBIGEO'}, axis=1)
        
        # Keep relevant columns from maps and rename IDDIST to match cv_data
        maps = maps[['IDDIST', 'NOMBDIST', 'NOMBPROV', 'NOMBDEP', 'geometry']]
        maps = maps.rename({'IDDIST':'UBIGEO'}, axis=1)
        
        # Convert UBIGEO to same type for merging
        maps['UBIGEO'] = maps['UBIGEO'].astype(str)
        cv_data['UBIGEO'] = cv_data['UBIGEO'].astype(str)
        
        # Merge datasets
        dataset_cv = pd.merge(maps, cv_data, how="inner", on="UBIGEO")
        
        # Create a field for educational level classification
        dataset_cv['nivel_educativo'] = np.nan
        dataset_cv.loc[dataset_cv['Nivel / Modalidad'].str.contains('Inicial', case=False, na=False), 'nivel_educativo'] = 'Inicial'
        dataset_cv.loc[dataset_cv['Nivel / Modalidad'].str.contains('Primaria', case=False, na=False), 'nivel_educativo'] = 'Primaria'
        dataset_cv.loc[dataset_cv['Nivel / Modalidad'].str.contains('Secundaria', case=False, na=False), 'nivel_educativo'] = 'Secundaria'
        
        return dataset_cv, maps, cv_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def prepare_points_data(dataset_cv):
    """Create point geometry from school coordinates"""
    # Filter schools with valid coordinates
    valid_coords = dataset_cv.dropna(subset=['Latitud', 'Longitud']).copy()
    
    # Convert to numeric
    valid_coords['Latitud'] = pd.to_numeric(valid_coords['Latitud'], errors='coerce')
    valid_coords['Longitud'] = pd.to_numeric(valid_coords['Longitud'], errors='coerce')
    
    # Create point geometry
    valid_coords['school_point'] = valid_coords.apply(
        lambda row: Point(row['Longitud'], row['Latitud']), axis=1
    )
    
    # Create a proper GeoDataFrame
    school_gdf = gpd.GeoDataFrame(valid_coords, geometry='school_point', crs="EPSG:4326")
    
    return school_gdf

def count_schools_by_district(dataset_cv):
    """Count schools by district for each educational level"""
    # Filter schools by level
    inicial = dataset_cv[dataset_cv['nivel_educativo'] == 'Inicial']
    primaria = dataset_cv[dataset_cv['nivel_educativo'] == 'Primaria']
    secundaria = dataset_cv[dataset_cv['nivel_educativo'] == 'Secundaria']
    
    # Count schools per district
    inicial_count = inicial.groupby('UBIGEO').size().reset_index(name='inicial_count')
    primaria_count = primaria.groupby('UBIGEO').size().reset_index(name='primaria_count')
    secundaria_count = secundaria.groupby('UBIGEO').size().reset_index(name='secundaria_count')
    
    return inicial_count, primaria_count, secundaria_count

def get_proximity_analysis(school_gdf, region):
    """Perform proximity analysis for primary and secondary schools in a region"""
    # Convert to UTM for accurate distance calculations
    school_utm = school_gdf.to_crs(epsg=32718)  # UTM Zone 18S for Peru
    
    # Filter for the region
    region_schools = school_utm[school_utm['Departamento'] == region].copy()
    
    # Filter by educational level
    region_primaria = region_schools[region_schools['nivel_educativo'] == 'Primaria'].copy()
    region_secundaria = region_schools[region_schools['nivel_educativo'] == 'Secundaria'].copy()
    
    # Create 5km buffer around each primary school
    region_primaria['buffer_5km'] = region_primaria['school_point'].buffer(5000)
    
    # Count secondary schools within each buffer
    nearby_counts = []
    
    for idx, primary in region_primaria.iterrows():
        buffer = primary['buffer_5km']
        # Count secondary schools within the buffer
        count = sum(region_secundaria['school_point'].within(buffer))
        nearby_counts.append(count)
    
    region_primaria['secondary_count'] = nearby_counts
    
    # Find schools with minimum and maximum counts
    min_school = region_primaria.loc[region_primaria['secondary_count'].idxmin()]
    max_school = region_primaria.loc[region_primaria['secondary_count'].idxmax()]
    
    # Convert back to WGS84 for visualization
    region_primaria_wgs84 = region_primaria.to_crs(epsg=4326)
    region_secundaria_wgs84 = region_secundaria.to_crs(epsg=4326)
    
    return {
        'primaria': region_primaria_wgs84,
        'secundaria': region_secundaria_wgs84,
        'min_school': min_school,
        'max_school': max_school,
        'region': region
    }

def get_dataset_stats(dataset_cv):
    """Calculate basic statistics about the dataset"""
    total_schools = len(dataset_cv)
    total_districts = dataset_cv['UBIGEO'].nunique()
    
    level_counts = dataset_cv['nivel_educativo'].value_counts().to_dict()
    
    region_counts = dataset_cv.groupby(['Departamento', 'nivel_educativo']).size().reset_index(name='count')
    
    return {
        'total_schools': total_schools,
        'total_districts': total_districts,
        'level_counts': level_counts,
        'region_counts': region_counts
    }