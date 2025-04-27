import torch
from torch.utils.data import Dataset
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from PIL import Image

class EuroSatDataset(Dataset):
    def __init__(self, csv_file, transform=None, root_dir=Path("."), is_tif=False, label_to_idx=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on images.
            root_dir (string): Directory with all the images.
            is_tif (bool): Flag indicating if the images are .tif files.
            label_to_idx (dict, optional): Mapping from labels to indices.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = Path(root_dir)
        self.is_tif = is_tif
        
        # Use provided label_to_idx or create a new one
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            self.label_to_idx = {label: idx for idx, label 
                                in enumerate(self.data_frame['label'].unique())}
        
        # Get the names of non-image features
        self.feature_columns = [
            'latitude', 'longitude', 'elevation_m', 'humidity_pct', 'ndvi',
            'night_lights', 'pop_density', 'slope_deg', 'soil_moisture', 'temperature_c'
        ]
        
        # Apply scaling
        scaler = StandardScaler()
        self.data_frame[self.feature_columns] = scaler.fit_transform(
            self.data_frame[self.feature_columns]
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and load image
        img_path = self.root_dir / self.data_frame.iloc[idx]['image_path']
        
        if self.is_tif:
            with rasterio.open(img_path) as src:
                # Load all bands and stack them
                image = np.stack([src.read(i) for i in range(1, src.count + 1)])
                
                # Convert to float32 and normalize to [0, 1]
                image = image.astype(np.float32)
                for i in range(image.shape[0]):  # Normalize each band independently
                    band = image[i]
                    if band.max() > band.min():
                        image[i] = (band - band.min()) / (band.max() - band.min())
        else:
            # Load .jpg image using PIL
            image = Image.open(img_path).convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
            image = image.transpose((2, 0, 1))  # Convert to CxHxW format
            
        # Get label
        label = self.label_to_idx[self.data_frame.iloc[idx]['label']]
        
        # Get non-image features
        features = self.data_frame.iloc[idx][self.feature_columns].values.astype(np.float32)
        
        # Get country index if available
        country_idx = self.data_frame.iloc[idx]['country_id'] if 'country_id' in self.data_frame.columns else None

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': torch.FloatTensor(image),
            'features': torch.FloatTensor(features),
            'country_idx': torch.tensor(country_idx).long(),
            'label': torch.tensor(label)
        }

# Example usage:
def load_nonimage_data(nonimage_data_dir):
    """
    Load and merge all non-image data from CSV files
    """
    dfs = {}
    
    # Load each CSV file
    for csv_file in nonimage_data_dir.glob('*.csv'):
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

def create_dataset_index(data_dir):
    """
    Create the CSV file that indexes all the data
    """
    csv_data_dir = data_dir / 'csv_data'
    output_path = csv_data_dir / 'dataset_index.csv'
    
    # Check if the dataset index already exists
    if output_path.exists():
        print(f"Dataset index already exists at {output_path}")
        return pd.read_csv(output_path)

    import geopandas as gpd
    from shapely.geometry import Point
    
    # Load non-image data first
    print("Loading non-image data...")
    nonimage_data_dir = data_dir / 'ne_10m_admin_0_countries'
    nonimage_df = load_nonimage_data(nonimage_data_dir)
    
    print("Loading world boundaries...")
    world_shapefile = nonimage_data_dir / 'ne_10m_admin_0_countries.shp'
    world = gpd.read_file(world_shapefile)
    
    data = []
    root_dir = data_dir / 'EuroSAT_MS'
    
    print("Processing satellite images...")
    # First pass: collect all image paths and labels
    image_count = 0
    class_counts = {}  # Track number of images per class
    
    for class_dir in root_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            class_counts[class_name] = 0
            for img_path in class_dir.glob('*.tif'):
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
                    'image_path': str(img_path.relative_to(data_dir)),
                    'label': class_name,
                    'latitude': float(lat[0]),
                    'longitude': float(lon[0]),
                    'country': country
                })
                class_counts[class_name] += 1
                image_count += 1
                if image_count % 10 == 0:
                    print(f"Processed {image_count} images...")
                    print("Class counts:", class_counts)
    
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
    df.to_csv(output_path, index=False)
    print(f"Dataset index saved to {output_path}")
    return df


def create_data_splits(data_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits and save separate CSV files
    """
    csv_data_dir = data_dir / 'csv_data'
    train_path = csv_data_dir / 'train_index.csv'
    val_path = csv_data_dir / 'val_index.csv'
    test_path = csv_data_dir / 'test_index.csv'

    # Check if the data splits already exist
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Data splits already exist.")
        return

    csv_path = csv_data_dir / 'dataset_index.csv'
    df = pd.read_csv(csv_path)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Calculate split indices
    train_end = int(train_ratio * len(df))
    val_end = train_end + int(val_ratio * len(df))
    
    # Split the data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Save the splits
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Data splits saved to {csv_data_dir}")


# Create DataLoaders
def get_dataloaders(data_dir, batch_size=32):
    """
    Create DataLoaders for train/val/test sets
    """
    from torch.utils.data import DataLoader
    
    # Define any transforms you want to apply
    transform = None  # Add your transforms here
    
    # Create datasets
    csv_data_dir = data_dir / 'csv_data'
    train_dataset = EuroSatDataset(csv_data_dir / 'train_index.csv', transform=transform, root_dir=data_dir)
    val_dataset = EuroSatDataset(csv_data_dir / 'val_index.csv', transform=transform, root_dir=data_dir)
    test_dataset = EuroSatDataset(csv_data_dir / 'test_index.csv', transform=transform, root_dir=data_dir)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EuroSAT dataset.')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory where the EuroSAT data is stored')
    args = parser.parse_args()

    # Use the provided directory
    data_dir = Path(args.data_dir)
    root_dir = data_dir/'EuroSAT_MS'
    
    # Create the main dataset index with all features
    df = create_dataset_index(data_dir)

    # Split into train/val/test
    create_data_splits(data_dir)
