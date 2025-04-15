import torch
from torch.utils.data import Dataset
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path

class EuroSatDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Convert labels to numerical format
        self.label_to_idx = {label: idx for idx, label 
                            in enumerate(self.data_frame['label'].unique())}
        
        # Get the names of non-image features
        self.feature_columns = [col for col in self.data_frame.columns 
                              if col not in ['image_path', 'label', 'country_id', 'country']]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and load image
        img_path = self.data_frame.iloc[idx]['image_path']
        with rasterio.open(img_path) as src:
            # Load all bands and stack them
            image = np.stack([src.read(i) for i in range(1, src.count + 1)])
            
        # Get label
        label = self.label_to_idx[self.data_frame.iloc[idx]['label']]
        
        # Get non-image features
        features = self.data_frame.iloc[idx][self.feature_columns].values.astype(np.float32)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': torch.FloatTensor(image),
            'features': torch.FloatTensor(features),
            'label': torch.tensor(label)
        }

# Example usage:
def load_nonimage_data():
    """
    Load and merge all non-image data from CSV files
    """
    data_dir = Path('/Users/jakepappo/LocalDocuments/Stat288/Project/eurosat/nonimage_data')
    dfs = {}
    
    # Load each CSV file
    for csv_file in data_dir.glob('*.csv'):
        metric_name = csv_file.stem  # filename without extension
        df = pd.read_csv(csv_file)
        dfs[metric_name] = df
    
    # Merge all dataframes on country
    merged_df = None
    for name, df in dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on='country', how='outer')
    merged_df = merged_df.rename(columns={
            'Unnamed: 0': 'country_id',
            'gdi_gdi_2017': 'gdi_2017',
            'gii_gii_2017': 'gii_2017',
            'gnipc_gnipc_2017': 'gnipc_2017',
            'abr_abr_2017': 'abr_2017',
            'mmr_mmr_2017': 'mmr_2017',
            'mys_mys_2017': 'mys_2017',
            'hdi_hdi_2017': 'hdi_2017',
            'co2_prod_co2_prod_2017': 'co2_prod_2017',
            'pop_total_pop_total_2017': 'pop_total_2017',
            'le_le_2017': 'le_2017'
        })
    
    return merged_df

def create_dataset_index():
    """
    Create the CSV file that indexes all the data
    """
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Load non-image data first
    print("Loading non-image data...")
    nonimage_df = load_nonimage_data()
    
    # Load world boundaries for country determination
    print("Loading world boundaries...")
    world_shapefile = Path('/Users/jakepappo/LocalDocuments/Stat288/Project/eurosat/nonimage_data/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    world = gpd.read_file(world_shapefile)
    
    data = []
    root_dir = Path('/Users/jakepappo/LocalDocuments/Stat288/Project/eurosat/EuroSAT_MS')
    
    print("Processing satellite images...")
    # First pass: collect all image paths and labels
    image_count = 0
    max_images = 50  # Limit for testing
    
    for class_dir in root_dir.iterdir():
        if class_dir.is_dir() and image_count < max_images:
            class_name = class_dir.name
            for img_path in class_dir.glob('*.tif'):
                if image_count >= max_images:
                    break
                # Get coordinates and country
                with rasterio.open(img_path) as src:
                    bounds = src.bounds
                    center_x = (bounds.left + bounds.right) / 2
                    center_y = (bounds.bottom + bounds.top) / 2
                    
                    # Create point geometry
                    point = Point(center_x, center_y)
                    point_gdf = gpd.GeoDataFrame(
                        geometry=[point], 
                        crs=src.crs
                    )
                    
                    # Transform to lat/lon (EPSG:4326)
                    point_gdf_latlon = point_gdf.to_crs('EPSG:4326')
                    lon, lat = point_gdf_latlon.geometry.iloc[0].xy
                    
                    # Transform to world CRS for country determination
                    if point_gdf.crs != world.crs:
                        point_gdf = point_gdf.to_crs(world.crs)
                    
                    # Find which country contains this point
                    country = None
                    for idx, country_row in world.iterrows():
                        if country_row.geometry.contains(point_gdf.geometry.iloc[0]):
                            country = country_row['ADMIN']
                            break
                
                data.append({
                    'image_path': str(img_path),
                    'label': class_name,
                    'latitude': float(lat[0]),
                    'longitude': float(lon[0]),
                    'country': country
                })
                image_count += 1
                if image_count % 10 == 0:
                    print(f"Processed {image_count}/{max_images} images...")
    
    # Create initial DataFrame
    print("Creating final dataset...")
    df = pd.DataFrame(data)
    
    # One-hot encode the country column
    print("One-hot encoding countries...")
    country_dummies = pd.get_dummies(df['country'], prefix='country').astype(int)
    df = pd.concat([df, country_dummies], axis=1)
    
    # Merge with non-image data
    df = df.merge(nonimage_df, on='country', how='left')
    
    # Save the final index
    output_path = Path('/Users/jakepappo/LocalDocuments/Stat288/Project/eurosat/data/dataset_index.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset index saved to {output_path}")
    return df

# Create train/val/test splits
def create_data_splits(csv_path, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits and save separate CSV files
    """
    df = pd.read_csv(csv_path)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=288).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    # Split the data
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]
    
    # Save splits (from src)
    train_df.to_csv('../data/train_index.csv', index=False)
    val_df.to_csv('../data/val_index.csv', index=False)
    test_df.to_csv('../data/test_index.csv', index=False)

# Create DataLoaders
def get_dataloaders(batch_size=32):
    """
    Create DataLoaders for train/val/test sets
    """
    from torch.utils.data import DataLoader
    
    # Define any transforms you want to apply
    transform = None  # Add your transforms here
    
    # Create datasets
    train_dataset = EuroSatDataset('../data/train_index.csv', transform=transform)
    val_dataset = EuroSatDataset('../data/val_index.csv', transform=transform)
    test_dataset = EuroSatDataset('../data/test_index.csv', transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
