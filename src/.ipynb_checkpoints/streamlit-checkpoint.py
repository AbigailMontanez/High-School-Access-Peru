import streamlit as st
import pandas as pd
from data_utils import load_data, prepare_points_data, count_schools_by_district, get_proximity_analysis, get_dataset_stats
from map_utils import create_choropleth_map, create_proximity_map, create_static_map

# Set page configuration
st.set_page_config(
    page_title="Educational Infrastructure in Peru",
    page_icon="🏫",
    layout="wide"
)

# Load and prepare data
dataset_cv, maps, cv_data = load_data()
if dataset_cv is not None:
    school_gdf = prepare_points_data(dataset_cv)
    inicial_count, primaria_count, secundaria_count = count_schools_by_district(dataset_cv)
    huancavelica_analysis = get_proximity_analysis(school_gdf, 'HUANCAVELICA')
    ayacucho_analysis = get_proximity_analysis(school_gdf, 'AYACUCHO')
    stats = get_dataset_stats(dataset_cv)

    # Create maps
    peru_choropleth = create_choropleth_map(maps, inicial_count, primaria_count, secundaria_count)
    huancavelica_map = create_proximity_map(huancavelica_analysis)
    ayacucho_map = create_proximity_map(ayacucho_analysis)
    
    # Create static maps
    inicial_map_img = create_static_map(
        maps.merge(inicial_count, on='UBIGEO', how='left').fillna(0),
        'inicial',
        'Blues',
        'Distribution of Initial Education Schools'
    )
    primaria_map_img = create_static_map(
        maps.merge(primaria_count, on='UBIGEO', how='left').fillna(0),
        'primaria',
        'Greens',
        'Distribution of Primary Schools'
    )
    secundaria_map_img = create_static_map(
        maps.merge(secundaria_count, on='UBIGEO', how='left').fillna(0),
        'secundaria',
        'Reds',
        'Distribution of Secondary Schools'
    )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Description", "Static Maps", "Dynamic Maps"])
    
    with tab1:
        st.title("Educational Infrastructure in Peru")
        st.markdown("### Data Description and Analysis")
        
        st.markdown("#### Unit of Analysis")
        st.markdown("""
        The unit of analysis for this study is **educational institutions** in Peru, specifically focusing on:
        - **Initial Education Schools** (preschool/kindergarten)
        - **Primary Schools** (elementary education)
        - **Secondary Schools** (high schools)
        
        The dataset includes information about the schools' locations, administrative divisions, and educational levels.
        """)
        
        st.markdown("#### Data Sources")
        st.markdown("""
        The data used in this analysis comes from two main sources:
        1. **School Directory** (`listado_iiee.xls`): Contains information about educational institutions including their coordinates, administrative details, and educational levels.
        2. **District Shapefile** (`DISTRITOS.shp`): Contains geographic boundaries for districts in Peru.
        
        These sources were combined to enable both spatial and statistical analysis of educational infrastructure distribution.
        """)
        
        st.markdown("#### Data Preprocessing")
        st.markdown("""
        The following preprocessing steps were performed:
        - Merging the schools dataset with district shapefiles using UBIGEO (district code)
        - Classifying schools into three main levels: Initial, Primary, and Secondary
        - Creating point geometries from latitude and longitude coordinates
        - Generating 5km buffers around primary schools to analyze proximity to secondary schools
        - Converting between coordinate reference systems for accurate distance calculations
        """)
        
        st.markdown("#### Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Schools", stats['total_schools'])
        with col2:
            st.metric("Initial Schools", stats['level_counts'].get('Inicial', 0))
        with col3:
            st.metric("Primary Schools", stats['level_counts'].get('Primaria', 0))
        with col4:
            st.metric("Secondary Schools", stats['level_counts'].get('Secundaria', 0))
        
        st.markdown("#### School Distribution by Region")
        st.dataframe(stats['region_counts'])
        
    with tab2:
        st.title("Static Maps: Spatial Distribution of Schools")
        st.markdown("""
        These maps show the distribution of schools by district across Peru,
        with color intensity representing the number of schools in each district.
        """)
        
        st.markdown("### Initial Education Schools")
        st.image(f"data:image/png;base64,{inicial_map_img}", use_column_width=True)
        
        st.markdown("### Primary Schools")
        st.image(f"data:image/png;base64,{primaria_map_img}", use_column_width=True)
        
        st.markdown("### Secondary Schools")
        st.image(f"data:image/png;base64,{secundaria_map_img}", use_column_width=True)
        
    with tab3:
        st.title("Dynamic Maps: Interactive Visualizations")
        
        st.markdown("### National Overview: Schools by Level")
        st.markdown("""
        This interactive map shows the distribution of schools by educational level across all districts in Peru.
        Use the layer control in the top right to toggle between different school levels.
        """)
        st.components.v1.html(peru_choropleth._repr_html_(), height=500)
        
        st.markdown("### Proximity Analysis: Huancavelica")
        st.markdown(f"""
        This map shows the proximity analysis for Huancavelica region:
        - Red marker: Primary school with the fewest secondary schools within 5km ({huancavelica_analysis['min_school']['secondary_count']})
        - Green marker: Primary school with the most secondary schools within 5km ({huancavelica_analysis['max_school']['secondary_count']})
        - Blue dots: Secondary schools
        """)
        st.components.v1.html(huancavelica_map._repr_html_(), height=500)
        
        st.markdown("### Proximity Analysis: Ayacucho")
        st.markdown(f"""
        This map shows the proximity analysis for Ayacucho region:
        - Red marker: Primary school with the fewest secondary schools within 5km ({ayacucho_analysis['min_school']['secondary_count']})
        - Green marker: Primary school with the most secondary schools within 5km ({ayacucho_analysis['max_school']['secondary_count']})
        - Blue dots: Secondary schools
        """)
        st.components.v1.html(ayacucho_map._repr_html_(), height=500)
        
        st.markdown("### Geographic Context Analysis")
        st.markdown("""
        #### Huancavelica Region
        
        The analysis of high school proximity in Huancavelica reveals significant geographic disparities. 
        This mountainous region presents challenging terrain with elevations ranging from 2,000 to 4,000 meters, 
        which directly impacts educational access. Primary schools with few nearby secondary schools 
        tend to be located in remote rural areas with limited transportation infrastructure, 
        whereas schools with better access to secondary education are typically situated in more 
        densely populated areas with better road connectivity.
        
        #### Ayacucho Region
        
        Similarly in Ayacucho, geography plays a decisive role in educational access. The region's 
        varied terrain, from high Andean zones to lower valleys, creates natural barriers to educational 
        infrastructure development. Urban centers show clustering of both primary and secondary schools, 
        while rural communities often have primary schools with limited or no nearby secondary education options. 
        This urban-rural divide highlights the challenges in providing equitable educational access across 
        diverse geographical contexts.
        """)
else:
    st.error("Failed to load data. Please check the data files and paths.")