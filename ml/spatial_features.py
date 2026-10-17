import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def generate_spatial_features():
    print("Loading processed earthquake data and spatial layers...")
  
    df_quakes = pd.read_csv("data/processed/cleaned_quakes.csv")
  
    geometry = [Point(xy) for xy in zip(df_quakes['longitude'], df_quakes['latitude'])]
    gdf_quakes = gpd.GeoDataFrame(df_quakes, geometry=geometry, crs="EPSG:4326")
    
    fault_lat, fault_lon = 37.7749, -122.4194 
    fault_point = gpd.GeoSeries([Point(fault_lon, fault_lat)], crs="EPSG:4326")
    
    gdf_quakes_proj = gdf_quakes.to_crs(epsg=32610)
    fault_point_proj = fault_point.to_crs(epsg=32610)
    
    gdf_quakes['dist_to_fault_km'] = gdf_quakes_proj.geometry.distance(fault_point_proj.iloc[0]) / 1000.0
    
    output_path = "data/processed/features_quakes.csv"
    gdf_quakes.drop(columns=['geometry']).to_csv(output_path, index=False)
    print(f"Spatial feature engineering complete! Saved to {output_path}")

if __name__ == "__main__":
    generate_spatial_features()
